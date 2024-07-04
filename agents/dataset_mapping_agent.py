#!/usr/bin/env python3
import argparse
import os
import random
from typing import Dict, Optional, Any, List
import math
import time

import numba
import numpy as np
import torch
import torchvision
import open3d as o3d
import threading

from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from skimage import measure
import skimage.morphology
from PIL import Image
import yaml

import numpy as np
import cv2

from collections import Counter

import open_clip
import supervision as sv
from tqdm import trange
from utils.vis import vis_result_fast, vis_result_slow_caption
import utils.depth_utils as du
import utils.pose as pu
from utils.fmm_planner import FMMPlanner
from utils.mapping import (
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
)
from utils.slam_classes import DetectionList, MapObjectList

from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim
)
from constants import color_palette

import ast
import openai
from openai.error import OpenAIError
openai.api_key = "jXDhEcEr8IdAU0hhERXIsPnAFy31QqbB"
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"

from agents.system_prompt import Instruction_system_prompt, Grounding_system_prompt

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e
from ultralytics import YOLO
from ultralytics import SAM



# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "Tag2Text")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)
try:
    from ram.models import tag2text, ram
    from ram import inference_tag2text, inference_ram
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

from scipy.spatial.transform import Rotation as R
import quaternion
from yacs.config import CfgNode as CN

BG_CLASSES = ["wall", "floor", "ceiling"]


def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
    elif variant == "fastsam":
        # raise NotImplementedError
        from ultralytics import YOLO
        # from FastSAM.tools import *
        FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
        model = YOLO(FASTSAM_CHECKPOINT_PATH)
        return model
    else:
        raise NotImplementedError

def compute_clip_features(image, detections, clip_model, clip_preprocess, device):
 
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_ids = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        image_crops.append(cropped_image)

        # class_id = detections.class_id[idx]
        # text_ids.append(classes[class_id])

    # # Create a batch of images
    image_batch = torch.stack([clip_preprocess(image) for image in image_crops]).to(device)
        
    # image_feats = np.concatenate(image_feats, axis=0)
    with torch.no_grad():
        image_feats = clip_model.encode_image(image_batch)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)


    # return image_crops, image_feats.cpu().numpy(), text_feats.cpu().numpy()
    return image_crops, image_feats.cpu().numpy()

# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    variant:str, model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if variant == "sam":
        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        # print(results)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
        # raise NotImplementedError
    else:
        raise NotImplementedError

# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class Mapping_Agent(Agent):
    def __init__(self, args, follower=None) -> None:
        self.args = args
        self.episode_n = 0
        self.l_step = 0

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        # ------------------------------------------------------------------
        ##### Initialize the SAM model
        # ------------------------------------------------------------------
        # self.mask_generator = get_sam_mask_generator(args.sam_variant, self.device)

        if self.args.detector == "dino":
            self.sam_predictor = get_sam_predictor(args.sam_variant, self.device)
            ## Initialize the Grounding DINO model ###
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
                device=self.device
            )
        elif self.args.detector == "yolo":
            self.sam_predictor = SAM('mobile_sam.pt').to(self.device)

            # Initialize a YOLO-World model
            self.yolo_model_w_classes = YOLO('yolov8l-world.pt').to(self.device)
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "bricks"
            ]
            bg_classes = ["wall", "floor", "ceiling"] 
            fileName = 'data/matterport_category_mappings.tsv'
            lines = []
            items = []
            self.classes=[]
            with open(fileName, 'r') as f:
                text = f.read()
            lines = text.split('\n')[1:]
            for l in lines:
                items.append(l.split('    '))   

            for i in items:
                if len(i) > 3 and i[-1] not in self.classes:
                    self.classes.append(i[-1]) 

            self.classes = [cls for cls in self.classes if cls not in bg_classes]
            self.classes = [cls for cls in self.classes if cls not in remove_classes]
            self.yolo_model_w_classes.set_classes(self.classes)


        # Initialize the CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Initialize the tagging model
        tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
        
        self.tagging_model = tagging_model.eval().to(self.device)

        # initialize Tag2Text
        self.tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        self.global_classes = set()

        self.init_sim_position = None
        self.init_sim_rotation = None


        # 3D mapping
        # !!!! desampling
        self.point_sum = o3d.geometry.PointCloud()
        self.camera_K = du.get_camera_K(
            336, 336, self.args.hfov)

        self.Open3D_traj = []
        self.init_map_and_pose()

        fcfg = open('configs/mapping_base.yaml')
        self.cfg = CN.load_cfg(fcfg)
        self.objects = MapObjectList(device=self.device)
        # print(self.cfg)
    
        if not self.cfg.skip_bg:
            # Handle the background detection separately 
            # Each class of them are fused into the map as a single object
            self.bg_objects = {
                c: None for c in BG_CLASSES
            }
        else:
            self.bg_objects = None

        
        # ------------------------------------------------------------------
            
        # ------------------------------------------------------------------
        ##### Initialize navigation
        # ------------------------------------------------------------------
        if follower != None:
            self.follower = follower
        self.selem = skimage.morphology.disk(3)

        self.text_queries = 'it is a soft chair.'
        self.user_prompt = None

        response_message = self.chat_with_gpt(Instruction_system_prompt, self.text_queries)

        ground_json = ast.literal_eval(response_message)
        self.target_data = ground_json.pop("target", None)
        self.landmark_data = ground_json
        print("target_data: ", self.target_data)
        print("landmark_data: ", self.landmark_data)
        # ------------------------------------------------------------------


    def init_map_and_pose(self):
        # local map
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        vh = int(self.args.map_height_cm / self.args.map_resolution)
        self.local_w, self.local_h = self.map_size, self.map_size
        self.explored_map = np.zeros((self.local_w, self.local_h))
        self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.frontier_map = np.zeros((self.local_w, self.local_h))
        self.visited_vis = np.zeros((self.local_w, self.local_h))
        self.goal_map = np.zeros((self.local_w, self.local_h))
        self.collision_map = np.zeros((self.local_w, self.local_h))
        self.last_pose = [self.map_size/2, self.map_size/2]
        self.local_position = np.zeros(3)
        self.local_rotation = np.zeros(3)
        self.z_min = 0
        self.z_max = 0
        self.target_point_list = []
        self.relative_angle = 0
        self.eve_angle = 0

        self.found_goal = False

        self.origins = [self.args.map_size_cm / 100.0 / 2.0,
                         self.args.map_size_cm / 100.0 / 2.0]

    def reset(self) -> None:
        self.episode_n += 1
        self.l_step = 0

        # self.classes = None
        # self.global_classes = set()

        self.point_sum = o3d.geometry.PointCloud()
        self.init_sim_position = None
        self.init_sim_rotation = None
        self.init_agent_positio = None
        self.Open3D_traj = []

        self.init_map_and_pose()

        self.objects = MapObjectList(device=self.device)
        self.open3d_reset = True


    def mapping(self, observations: Observations, agent_state, window):
        time_step_info = 'Mapping time (s): \n'

        preprocess_s_time = time.time()
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.init_sim_position = agent_state[:3, 3]
            self.init_agent_position = agent_state[:3, 3]
            self.init_sim_rotation = agent_state[:3, :3]

        # print("current position: ", agent_state.sensor_states["depth"].position)
        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------
            
        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        image_pil = Image.fromarray(image_rgb)
        annotated_image = image

        get_results = False
        if self.args.detector == "dino":

            raw_image = image_pil.resize((384, 384))
            raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device)
            
            res = inference_ram(raw_image , self.tagging_model)
            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            text_prompt=res[0].replace(' |', ',')
            # print(text_prompt)
            
            # Add "other item" to capture objects not in the tag2text captions. 
            # Remove "xxx room", otherwise it will simply include the entire image
            # Also hide "wall" and "floor" for now...
            add_classes = ["other item"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "bricks"
            ]
            bg_classes = ["wall", "floor", "ceiling"]

            if self.args.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            self.classes = process_tag_classes(
                text_prompt, 
                add_classes = add_classes,
                remove_classes = remove_classes,
            )

            # add classes to global classes
            self.global_classes.update(self.classes)

            if self.args.accumu_classes:
                # Use all the classes that have been seen so far
                self.classes = list(self.global_classes)


            # ------------------------------------------------------------------
            ##### Detection and segmentation
            # ------------------------------------------------------------------
            # Using GroundingDINO to detect and SAM to segment
            detections = self.grounding_dino_model.predict_with_classes(
                image=image, # This function expects a BGR image...
                classes=self.classes,
                box_threshold=self.args.box_threshold,
                text_threshold=self.args.text_threshold,
            )
            
            if len(detections.class_id) > 0:
                ### Non-maximum suppression ###
                # print(f"Before NMS: {len(detections.xyxy)} boxes")
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    self.args.nms_threshold
                ).numpy().tolist()
                # print(f"After NMS: {len(detections.xyxy)} boxes")

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
                
                # Somehow some detections will have class_id=-1, remove them
                valid_idx = detections.class_id != -1
                detections.xyxy = detections.xyxy[valid_idx]
                detections.confidence = detections.confidence[valid_idx]
                detections.class_id = detections.class_id[valid_idx]
                
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=self.sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                get_results = True  

        elif self.args.detector == "yolo":

            # UltraLytics YOLO
            yolo_s_time = time.time()
            yolo_results_w_classes = self.yolo_model_w_classes.predict(image, conf=0.1, verbose=False)
            # print(self.yolo_model_w_classes.device.type)
            yolo_e_time = time.time()

            confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
            detection_class_ids = yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int)
            xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()
            print('yolo: %.3f秒'%(yolo_e_time - yolo_s_time)) 

            detections = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=None,
            )

            if len(confidences) > 0:

                # UltraLytics SAM
                sam_s_time = time.time()
                sam_out = self.sam_predictor.predict(image, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
                
                detections = sv.Detections(
                    xyxy=xyxy_np,
                    confidence=confidences,
                    class_id=detection_class_ids,
                    mask=masks_np,
                )
                sam_e_time = time.time()
                print('sam: %.3f秒'%(sam_e_time - sam_s_time)) 

                get_results = True  
 
        if get_results:
            clip_s_time = time.time()
            image_crops, image_feats = compute_clip_features(
                image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)
            
            ### Visualize results ###
            annotated_image = vis_result_fast(image, detections, self.classes, draw_bbox=True, draw_mask=True)
            clip_e_time = time.time()
            # print('clip: %.3f秒'%(clip_e_time - clip_s_time)) 


            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                "image_crops": image_crops,
                "image_feats": image_feats
            }

        else:
            results = None
                
        preprocess_e_time = time.time()
        time_step_info += 'Preprocess time:%.3fs\n'%(preprocess_e_time - preprocess_s_time)

        # ------------------------------------------------------------------
        ##### 3D Projection
        # ------------------------------------------------------------------
        v_time = time.time()

        cfg = self.cfg
        depth = self._preprocess_depth(depth)
        
        camera_matrix, camera_pose = self.get_transform_matrix(agent_state)
        self.Open3D_traj.append(camera_matrix)

        self.relative_angle = np.arctan2(camera_matrix[0][0], camera_matrix[2][0])* 57.29577951308232 + 180
        # print("self.relative_angle: ", self.relative_angle)

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth,
            cam_K = self.camera_K,
            idx = self.l_step,
            gobs = results,
            trans_pose = camera_matrix,
            class_names = self.classes,
            BG_CLASSES = BG_CLASSES,
        )

        
        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if self.bg_objects[class_name] is None:
                    self.bg_objects[class_name] = detected_object
                else:
                    matched_obj = self.bg_objects[class_name]
                    matched_det = detected_object
                    self.bg_objects[class_name] = merge_obj2_into_obj1(self.cfg, matched_obj, matched_det, run_dbscan=False)       

        
        obj_time = time.time()
        objv_time = obj_time - v_time
        # print('build objects: %.3f秒'%objv_time) 
        time_step_info += 'build objects time:%.3fs\n'%(objv_time)

        if len(fg_detection_list) > 0 and len(self.objects) > 0 :
            spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
            visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

            # Threshold sims according to cfg. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')

            self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)



        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (self.l_step+1) % cfg.denoise_interval == 0:
            self.objects = denoise_objects(cfg, self.objects)
        if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
            self.objects = filter_objects(cfg, self.objects)
        if cfg.merge_interval > 0 and (self.l_step+1) % cfg.merge_interval == 0:
            self.objects = merge_objects(cfg, self.objects)

        sim_time = time.time()
        sim_obj_time = sim_time - obj_time 
        # print('calculate merge: %.3f秒'%sim_obj_time) 
        time_step_info += 'calculate merge time:%.3fs\n'%(sim_obj_time)


        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

        # ------------------------------------------------------------------
        
        # text_queries = ['sofa chair']
        if not window.send_queue.empty():
            self.text_queries = window.send_queue.get()
            # print("self.text_queries: ", self.text_queries)

        clip_time = time.time()
        candidate_objects = []
        similarity_threshold = 0.26
        if len(self.objects) > 0:
            if len(self.landmark_data) == 0:
                self.objects, similarities = color_by_clip_sim(self.text_queries, 
                                                            self.objects, 
                                                            self.clip_model, 
                                                            self.clip_tokenizer)

                candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                    if similarities[i] > similarity_threshold]
        #     else:
        #         self.user_prompt = self.Bbox_prompt(similarity_threshold)

        #         if self.user_prompt != None:
        #             generated_text = self.chat_with_gpt(Grounding_system_prompt, self.user_prompt)
        #             ground_json = ast.literal_eval(generated_text)
        #             if ground_json['result'] != None:
        #                 candidate_objects = [self.objects[int(ground_json['result'])]]
                


        f_clip_time = time.time()
        # print('calculate clip sim: %.3f秒'%(f_clip_time - clip_time)) 
        time_step_info += 'calculate clip sim time:%.3fs\n'%(f_clip_time - clip_time)

        # ------------------------------------------------------------------
        ##### 2D Obstacle Map
        # ------------------------------------------------------------------
        f_map_time = time.time()
        current_scene_pcd = self.build_full_scene_pcd(depth, image_rgb)
        current_scene_pcd.transform(camera_matrix)
        self.point_sum += current_scene_pcd

        # self.explored_map = np.zeros((self.local_w, self.local_h))
        # self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.explored_map[self.map_building(current_scene_pcd, camera_pose)] = 1
        self.obstacle_map[self.map_building(current_scene_pcd, camera_pose, self.args.map_height_cm / 100.0 /2)] = 1

        target_score, target_edge_map, self.target_point_list = self.Frontier(self.explored_map, self.obstacle_map, camera_pose, 5)

        v_map_time = time.time()
        # print('voxel map: %.3f秒'%(v_map_time - f_map_time)) 

        self.goal_map = np.zeros((self.local_w, self.local_h))
        if len(candidate_objects) > 0:
            self.goal_map[self.goal_map_building(candidate_objects[0]['pcd'])] = 1
            self.found_goal = True
        else:
            if len(self.target_point_list) > 0:
                self.goal_map[self.target_point_list[0][0], self.target_point_list[0][1]] = 1
            else:
                goal_pose_x = int(np.random.rand() * self.map_size)
                goal_pose_y = int(np.random.rand() * self.map_size)
                self.goal_map[goal_pose_x, goal_pose_y] = 1
        
        vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map)

        if np.sum(self.goal_map) == 1:
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
        else: 
            stg = self._get_stg(self.explored_map, self.last_pose, self.goal_map)
        # stg = np.argwhere(self.goal_map == 1)
        
        x = (stg[1] - int(self.map_size / 2)) * self.args.map_resolution / 100.0
        y = 0
        z = (stg[0] - int(self.map_size / 2)) * self.args.map_resolution / 100.0

        # Open3d_goal_pose = np.stack((x, y, z), axis=-1)
        Open3d_goal_pose = [x, y, z]
        Rx = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        R_habitat2open3d = self.init_sim_rotation @ Rx
        self.habitat_goal_pose = np.dot(R_habitat2open3d.T, Open3d_goal_pose) + self.init_agent_position
        habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

        plan_path = []
        # plan_path = self.follower.get_path_points(
        #     habitat_final_pose
        # )

        # if len(plan_path) > 0:
        #     plan_path = np.dot(R_habitat2open3d, (np.array(plan_path) - self.init_agent_position).T).T

        dd_map_time = time.time()
        time_step_info += '2d map building time:%.3fs\n'%(dd_map_time - f_map_time)

        # ------------------------------------------------------------------
        ##### Send to Open3D visualization thread
        # ------------------------------------------------------------------
        window.receive_queue.put([image_rgb, 
                              depth, 
                              annotated_image, 
                              self.objects.to_serializable(), 
                              np.asarray(self.point_sum.points), 
                              np.asarray(self.point_sum.colors), 
                              self.Open3D_traj,
                              self.open3d_reset,
                              plan_path,
                              transform_rgb_bgr(vis_image),
                              time_step_info]
                              )    
        self.open3d_reset = False

        return annotated_image
    
    def chat_with_gpt(self, system_prompt, user_prompt):

        message_list=[]
        message_list.append({"role": "system", "content": system_prompt})
        message_list.append({"role": "user", "content": user_prompt})

        retries = 10    
        while retries > 0:  
            try: 
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo', 
                    messages=message_list,
                )

                response_message = response.choices[0].message['content']
                break
            except OpenAIError as e:
                if e:
                    print(e)
                    print('Timeout error, retrying...')    
                    retries -= 1
                    time.sleep(5)
                else:
                    raise e
        
        print(response_message)
        return response_message


    def Bbox_prompt(self, similarity_threshold = 0.28):
        # Target object
        self.objects, target_similarities = color_by_clip_sim(self.target_data["phrase"], 
                                                                self.objects, 
                                                                self.clip_model, 
                                                                self.clip_tokenizer)
        
        target_bbox = [{"centroid": [round(ele, 1) for ele in self.objects[i]['bbox'].get_center()], 
                        "extent": [round(ele, 1) for ele in self.objects[i]['bbox'].get_extent()]} 
                        for i in range(len(self.objects)) if target_similarities[i] > similarity_threshold]

        # Landmark objects
        landmark_bbox = {}
        for value in self.landmark_data.values():
            if value["phrase"] is not None: 
                self.objects, landmark_similarities = color_by_clip_sim(value["phrase"], 
                                                                        self.objects, 
                                                                        self.clip_model, 
                                                                        self.clip_tokenizer, 
                                                                        color_set = False)
                
                landmark_bbox[value["phrase"]] = \
                    [{"centroid": [round(ele, 1) for ele in self.objects[i]['bbox'].get_center()], 
                        "extent": [round(ele, 1) for ele in self.objects[i]['bbox'].get_extent()]}  
                        for i in range(len(self.objects)) 
                        if landmark_similarities[i] > similarity_threshold]

        # inference the target if all the objects are found
        observation = {self.text_queries + '\n'}
        evaluation = {}
        if len(target_bbox) > 0 and all(landmark_bbox[key] for key in landmark_bbox.keys()):
            evaluation = {
                "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])": {
                    str(i): bbox for i, bbox in enumerate(target_bbox)
                },
                "Target Candidate BBox Volume (meter^3)": {
                    str(i): round(bbox["extent"][0] * bbox["extent"][1] * bbox["extent"][2], 3)
                    for i, bbox in enumerate(target_bbox)
                },
            }

            evaluation["Landmark BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])"] = {
                phrase: {str(i): bbox for i, bbox in enumerate(landmark)}  
                for phrase, landmark in landmark_bbox.items()
            }
                        
            return str(observation) + str(evaluation)
        else:
            return None
            

    def map_building(self, point_sum, camera_pose, height_diff = 0):
        # height range (z is down in Open3D)
        z_min = camera_pose[1] + 0.1
        z_max = camera_pose[1] 

        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 1] <= z_min) & (points[:, 1] >= z_max) & \
                (points[:, 0] >= -self.origins[0]) & (points[:, 0] <= self.origins[0]) & \
                (points[:, 2] >= -self.origins[0]) & (points[:, 2] <= self.origins[0])

        points_filtered = points[mask]
        colors_filtered = colors[mask]

        # 计算二维地图的索引ww
        i_values = np.floor((points_filtered[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.map_size / 2)
        j_values = np.floor((points_filtered[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.map_size / 2)

        return j_values, i_values
    
    def goal_map_building(self, point_sum):

        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 0] >= -self.origins[0]) & (points[:, 0] <= self.origins[0]) & \
                (points[:, 2] >= -self.origins[0]) & (points[:, 2] <= self.origins[0])

        points_filtered = points[mask]
        colors_filtered = colors[mask]

        # 计算二维地图的索引ww
        i_values = np.floor((points_filtered[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.map_size / 2)
        j_values = np.floor((points_filtered[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.map_size / 2)

        return j_values, i_values

    def Frontier(self, explored_map, obstacle_map, pose, threshold_point):
        # ------------------------------------------------------------------
        ##### Get the frontier map and score
        # ------------------------------------------------------------------
        edge_map = np.zeros((self.map_size, self.map_size))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        obstacle_map = cv2.dilate(obstacle_map, kernel)

        kernel = np.ones((5, 5), dtype=np.uint8)
        show_ex = cv2.inRange(explored_map,0.1,1)
        free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)
        contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            contour = max(contours, key = cv2.contourArea)
            cv2.drawContours(edge_map,contour,-1,1,1)

        # clear the boundary
        edge_map[0:2, 0:self.map_size]=0.0
        edge_map[self.map_size-2:self.map_size, 0:self.map_size-1]=0.0
        edge_map[0:self.map_size, 0:2]=0.0
        edge_map[0:self.map_size, self.map_size-2:self.map_size]=0.0

        target_edge = edge_map - obstacle_map

        target_edge[target_edge>0.8]=1.0
        target_edge[target_edge!=1.0]=0.0

        img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

        
        local_pose = [pose[2]*100/self.args.map_resolution + int(self.map_size / 2), 
                      pose[0]*100/self.args.map_resolution + int(self.map_size / 2)]
        pose_x = int(local_pose[0]) if int(local_pose[0]) < self.map_size-1 else self.map_size-1
        pose_y = int(local_pose[1]) if int(local_pose[1]) < self.map_size-1 else self.map_size-1
        
        # selem = skimage.morphology.disk(1)
        # traversible = skimage.morphology.binary_dilation(
        #     obstacle_map, selem) != True
        # # traversible = 1 - traversible
        # planner = FMMPlanner(traversible)
        # goal_pose_map = np.zeros((obstacle_map.shape))
        # goal_pose_map[pose_x, pose_y] = 1
        # planner.set_multi_goal(goal_pose_map)

        current_pose = [pose_x, pose_y]
        def draw_line(start, end, mat, steps=25, w=1):
            for i in range(steps + 1):
                x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
                y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
                mat[x - w:x + w, y - w:y + w] = 1
            return mat
        self.visited_vis = draw_line(self.last_pose, current_pose, self.visited_vis)
        self.last_pose = current_pose

        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = []
        Goal_area_list = []
        dict_cost = {}
        for i in range(1, len(props)):
            if props[i].area > threshold_point:
                # dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
                dict_cost[i] = props[i].area
                # dict_cost[i] = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])]

        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

            for i, (key, value) in enumerate(dict_cost):
                Goal_edge[img_label == key + 1] = 1
                Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
                Goal_area_list.append(value)
                if i == 5:
                    break

        return  Goal_area_list, Goal_edge, Goal_point
    
    # def act(self):
    #     # ------------------------------------------------------------------
    #     ##### Update long-term goal if target object is found
    #     ##### Otherwise, use the LLM to select the goal
    #     # ------------------------------------------------------------------
    #     keystroke = cv2.waitKey(0)
    #     action = None
    #     if keystroke == ord(FORWARD_KEY):
    #         action = HabitatSimActions.MOVE_FORWARD
    #         print("action: FORWARD")
    #     elif keystroke == ord(LEFT_KEY):
    #         action = HabitatSimActions.TURN_LEFT
    #         print("action: LEFT")
    #         self.relative_angle -= 15
    #     elif keystroke == ord(RIGHT_KEY):
    #         action = HabitatSimActions.TURN_RIGHT
    #         print("action: RIGHT")
    #         self.relative_angle += 15
    #     elif keystroke == ord(UP_KEY):
    #         action = HabitatSimActions.LOOK_UP
    #         print("action: UP")
    #         self.eve_angle += 15
    #     elif keystroke == ord(DOWN_KEY):
    #         action = HabitatSimActions.LOOK_DOWN
    #         print("action: DOWN")
    #         self.eve_angle -= 15
    #     elif keystroke == ord(FINISH):
    #         action = HabitatSimActions.STOP
    #         print("action: FINISH")
    #     else:
    #         print("INVALID KEY")
        
    #     self.l_step += 1
    #     # self.relative_angle = self.relative_angle % 360.0
    #     # if self.relative_angle > 180:
    #     #         self.relative_angle -= 360

    #     return action
    
    # def act(self):
        
    #     action_s_time = time.time()

    #     action = self.follower.get_next_action(
    #         self.habitat_goal_pose
    #     )

    #     if not self.found_goal and action == 0:
    #         action = 2

    #     eve_start_x = int(5 * math.sin(self.relative_angle) + self.last_pose[0])
    #     eve_start_y = int(5 * math.cos(self.relative_angle) + self.last_pose[1])
    #     if eve_start_x >= self.map_size: eve_start_x = self.map_size-1
    #     if eve_start_y >= self.map_size: eve_start_y = self.map_size-1 
    #     if eve_start_x < 0: eve_start_x = 0 
    #     if eve_start_y < 0: eve_start_y = 0 
    #     if self.explored_map[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
    #         action = 5
    #         self.eve_angle -= 30
    #     elif self.explored_map[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
    #         action = 4
    #         self.eve_angle += 30

    #     # keystroke = cv2.waitKey(0)
    #     # if keystroke == ord(FINISH):
    #     #     action = HabitatSimActions.STOP
    #     #     print("action: FINISH")

    #     action_e_time = time.time()

    #     # print('acton: %.3f秒'%(action_e_time - action_s_time)) 
    #     self.l_step += 1
    #     return action
    
    
    def _get_stg(self, grid, start, goal):
        """Get short-term goal"""

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        # print("grid: ", grid.shape)

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[x1:x2, y1:y2] == 1] = 0
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        mask = traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
        # state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        # stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        # return (stg_x, stg_y), stop
    
    def _preprocess_depth(self, depth, min_d=0.5, max_d=5.0):
        

        return depth / 1000.0


    
    def get_transform_matrix(self, agent_state):
        """
        transform the habitat-lab space to Open3D space (initial pose in habitat)
        habitat-lab space need to rotate camera from x,y,z to  x, -y, -z
        Returns Pose_diff, R_diff change of the agent relative to the initial timestep
        """

        h_camera_matrix = agent_state

        habitat_camera_self = np.eye(4)
        habitat_camera_self[:3, :3] = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        
        R_habitat2open3d = np.eye(4)
        R_habitat2open3d[:3, :3] = self.init_sim_rotation
        R_habitat2open3d[:3, 3] = self.init_sim_position

        O_camera_matrix = habitat_camera_self @ h_camera_matrix @ habitat_camera_self
        # O_camera_matrix = habitat_camera_self @ R_habitat2open3d.T @ h_camera_matrix @ habitat_camera_self

        return O_camera_matrix, O_camera_matrix[:3, 3]
    
    
    def build_full_scene_pcd(self, depth, image):
        height, width = depth.shape

        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.
        fx = (width / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))
        # fy = (height / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))
 
        x = np.arange(0, width, 1.0)
        y = np.arange(0, height, 1.0)
        u, v = np.meshgrid(x, y)
        
        # Apply the mask, and unprojection is done only on the valid points
        masked_depth = depth # (N, )
        u = u # (N, )
        v = v # (N, )

        # Convert to 3D coordinates
        x = (u - cx) * masked_depth / fx
        y = (v - cy) * masked_depth / fx
        z = masked_depth

        # Stack x, y, z coordinates into a 3D point cloud
        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)
        
        # Perturb the points a bit to avoid colinearity
        points += np.random.normal(0, 4e-3, points.shape)

        image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

        colors = image / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        camera_object_pcd = pcd.voxel_down_sample(0.05)

        return camera_object_pcd
    
    def _visualize(self, map_pred, exp_pred, map_edge, goal_map):

        # start_x, start_y, start_o = pose

        sem_map = np.zeros((self.local_w, self.local_h))

        # no_cat_mask = sem_map == 20
        map_mask = map_pred == 1
        exp_mask = exp_pred == 1
        vis_mask = self.visited_vis == 1
        edge_mask = map_edge == 1

        # sem_map[no_cat_mask] = 0
        # m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[exp_mask] = 2

        # m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[map_mask] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_map, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        if np.sum(goal_map) == 1:
            f_pos = np.argwhere(goal_map == 1)
            # fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]))
            # goal_fmb = skimage.draw.circle_perimeter(int((fmb[0]+fmb[1])/2), int((fmb[2]+fmb[3])/2), 23)
            goal_fmb = skimage.draw.circle_perimeter(f_pos[0][0], f_pos[0][1], int(self.map_size/8 -1))
            goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
            goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        vis_image = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)

       
        def get_contour_points(pos, origin, size=20):
            x, y, o = pos
            pt1 = (int(x) + origin[0],
                int(y) + origin[1])
            pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
            pt3 = (int(x + size * np.cos(o)) + origin[0],
                int(y + size * np.sin(o)) + origin[1])
            pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

            return np.array([pt1, pt2, pt3, pt4])

        pos = [self.last_pose[1], int(self.map_size)-self.last_pose[0], np.deg2rad(self.relative_angle)]
        agent_arrow = get_contour_points(pos, origin=(0, 0), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        # cv2.imshow("episode_n {}".format(self.episode_n), vis_image)

        return vis_image

