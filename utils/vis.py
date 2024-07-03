import copy
from typing import Iterable
import dataclasses
from PIL import Image
import cv2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import open3d as o3d

import supervision as sv
from supervision.draw.color import Color, ColorPalette

# Copied from https://github.com/concept-graphs/concept-graphs/     
def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: list[str], 
    color: Color | ColorPalette = ColorPalette.default(), 
    instance_random_color: bool = False,
    draw_bbox: bool = False,
    draw_mask: bool = False
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color = color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color = color
    )
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ ,_
        in detections]
    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
    
    annotated_image = scene=image.copy()
    if draw_mask:
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    

    draw_bbox_id = True
    if draw_bbox_id:
        annotated_image = box_annotator.annotate_bbox_id(scene=annotated_image, detections=detections, labels=labels)


    return annotated_image


def init_vis_image(goal_name, action = 0):
    vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Find {}  Action {}".format(goal_name, str(action))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color


#     # draw legend
#     lx, ly, _ = legend.shape
#     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat
