import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import torchvision
import supervision as sv

from PIL import Image
from utils.vis import vis_result_fast


from utils.model_utils import(
    get_sam_predictor,
    process_tag_classes,
    get_sam_segmentation_from_xyxy
)

from ultralytics import YOLO
from ultralytics import SAM
from groundingdino.util.inference import Model

 
# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
# GroundingDINO config and checkpoint
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


# Tag2Text checkpoint
# TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")


GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")


class Object_Detection_and_Segmentation():
    r""" YOLO or DINO
    
    Args:
    """
    def __init__(self, args, classes, device):
        self.args = args
        self.device = device
        if self.args.detector == "dino":
            self.sam_predictor = get_sam_predictor(args.sam_variant, self.device)
            ## Initialize the Grounding DINO model ###
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
                device=self.device
            )
            # # Initialize the tagging model
            # tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
            #                                 image_size=384,
            #                                 vit='swin_l')
            
            # self.tagging_model = tagging_model.to(self.device)

            # # initialize Tag2Text
            # self.tagging_transform = TS.Compose([
            #     TS.Resize((384, 384)),
            #     TS.ToTensor(), 
            #     TS.Normalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225]),
            # ])

            # self.global_classes = set()
        elif self.args.detector == "yolo":
            self.sam_predictor = SAM('mobile_sam.pt').to(self.device)

            # Initialize a YOLO-World model
            self.yolo_model_w_classes = YOLO('yolov8l-world.pt').to(self.device)
            
            if self.args.task_config == "vlobjectnav_hm3d.yaml":
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
                self.classes = []
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
            self.yolo_model_w_classes.set_classes(classes)
        
    def detect(self, image, image_rgb, classes, save_image=False):
        
        get_results = False
        if self.args.detector == "dino":

            # raw_image = image_pil.resize((384, 384))
            # raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device)
            
            # res = inference_ram(raw_image , self.tagging_model)
            # # Currently ", " is better for detecting single tags
            # # while ". " is a little worse in some case
            # text_prompt=res[0].replace(' |', ',')
            # print(text_prompt)
            
            # # Add "other item" to capture objects not in the tag2text captions. 
            # # Remove "xxx room", otherwise it will simply include the entire image
            # # Also hide "wall" and "floor" for now...
            # add_classes = ["other item", "staircase", self.text_queries]
            # remove_classes = [
            #     "room", "kitchen", "office", "house", "home", "building", "corner",
            #     "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
            #     "apartment", "image", "city", "blue", "skylight", "hallway", 
            #     "bureau", "modern", "salon", "doorway", "wall lamp", "bricks"
            # ]
            # bg_classes = ["wall", "floor", "ceiling"]

            # if self.args.add_bg_classes:
            #     add_classes += bg_classes
            # else:
            #     remove_classes += bg_classes

            # self.classes = process_tag_classes(
            #     text_prompt, 
            #     add_classes = add_classes,
            #     remove_classes = remove_classes,
            # )

            # # add classes to global classes
            # self.global_classes.update(self.classes)

            # if self.args.accumu_classes:
            #     # Use all the classes that have been seen so far
            #     self.classes = list(self.global_classes)


            # ------------------------------------------------------------------
            ##### Detection and segmentation
            # ------------------------------------------------------------------
            # Using GroundingDINO to detect and SAM to segment
            detections = self.grounding_dino_model.predict_with_classes(
                image=image, # This function expects a BGR image...
                classes=classes,  # TODO 是通过提示文本的label然后对图像进行检测
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
                
                # TODO 这里才进行segementation
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=self.sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                get_results = True  

        elif self.args.detector == "yolo":

            # UltraLytics YOLO
            yolo_s_time = time.time()
            with torch.no_grad():
                # yolo_results_w_classes = self.yolo_model_w_classes(image, conf=0.1, verbose=False)
                yolo_results_w_classes = self.yolo_model_w_classes.predict(image, conf=0.1, verbose=False)
            # print(yolo_results_w_classes)
            yolo_e_time = time.time()

            confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
            detection_class_ids = yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int)
            xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()
            # print('yolo: %.3f秒'%(yolo_e_time - yolo_s_time)) 

            detections = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=None,
            )

            if len(confidences) > 0:

                # UltraLytics SAM
                sam_s_time = time.time()
                with torch.no_grad():
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
                # print('sam: %.3f秒'%(sam_e_time - sam_s_time)) 

                get_results = True  
 

        # 其实这个时候就应该把检测结果给标在图上然后返回当前的image
        result_image = vis_result_fast(image, detections, classes, draw_bbox=False, draw_mask=False, draw_bbox_id=True)
        if save_image:
            pil_image = Image.fromarray(np.uint8(result_image))
            output_file = "/home/rickyyzliu/workspace/embodied-AI/habitat/detect.jpg" 
            pil_image.save(output_file)
            
            raw_file = "/home/rickyyzliu/workspace/embodied-AI/habitat/raw_img.jpg" 
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(raw_file)

        return get_results, detections, result_image



