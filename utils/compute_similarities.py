import torch
import torch.nn.functional as F
import matplotlib
import open3d as o3d
import numpy as np
import time
from scipy.stats import entropy

from utils.slam_classes import MapObjectList, DetectionList
from utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accuracte_batch, 
    compute_3d_giou_accurate_batch,
)
from utils.mapping import (
    merge_obj2_into_obj1, 
    compute_overlap_matrix_2set
)

def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    # torch.cuda.synchronize()
    det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    # print("visual_sim shape: ", visual_sim.shape)
    return visual_sim.cpu().numpy()

def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return sims

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    
    # 定义一个
    detection_to_object = {}

    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
            
            detection_to_object[detection_list[i]["mask_idx"][0]] = len(objects)-1

        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj

            detection_to_object[detection_list[i]["mask_idx"][0]] = j
            
    return objects, detection_to_object

cmap = matplotlib.colormaps.get_cmap("turbo")
def color_by_clip_sim(text_queries, objects, clip_model, clip_tokenizer, color_set = True) :
    clip_model_device = next(clip_model.parameters()).device
    
    text_queries_tokenized = clip_tokenizer(text_queries).to(clip_model_device)
    text_query_ft = clip_model.encode_text(text_queries_tokenized)
    text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
    text_query_ft = text_query_ft.squeeze()
    
    # similarities = objects.compute_similarities(text_query_ft)
    objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
    objects_clip_fts = objects_clip_fts.to(clip_model_device)
    similarities = torch.cosine_similarity(
        text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
    )
    
    max_value = similarities.max()
    min_value = similarities.min()
    # print("max similarities: ", max_value)
    # print("max text_similarities: ", text_similarities.max())

    if color_set:
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]
        
        for i in range(len(objects)):
            objects[i]['pcd'].colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(objects[i]['pcd'].points), 1)
                )
            )

    return objects, similarities

cmap = matplotlib.colormaps.get_cmap("turbo")
def cal_clip_sim(text_queries, feats, clip_model, clip_tokenizer):
    clip_model_device = next(clip_model.parameters()).device
    
    text_queries_tokenized = clip_tokenizer(text_queries).to(clip_model_device)
    text_query_ft = clip_model.encode_text(text_queries_tokenized)
    text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
    text_query_ft = text_query_ft.squeeze()
    
    similarities = torch.cosine_similarity(
        text_query_ft.unsqueeze(0), feats.unsqueeze(0), dim=-1
    )
    
    return similarities