import os
import numpy as np 

from PIL import Image, ImageDraw
import easyocr
from paddleocr import PaddleOCR
import supervision as sv

def draw_bounding_box(point1, point2, image):
    # draw
    color = (0, 255, 0)
    thickness = 2
    draw = ImageDraw.Draw(image)
    draw.rectangle([point1, point2], outline=color, width=thickness)
    return image

def run_easy_ocr(image, bounding_box_annotator, label_annotator):
    # easy ocr https://github.com/JaidedAI/EasyOCR
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    outputs = reader.readtext(image)
    xyxy, label_confidence = [], []
    for out in outputs:
        xyxy.append([*out[0][0], *out[0][2]])
        label_confidence.append(f"{out[1]} {out[2]:.2f}")
    
    detections = sv.Detections(np.array(xyxy), class_id=np.arange(len(xyxy)))
    annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=np.array(label_confidence),
    )   
    # annotated_frame.show()
    return annotated_frame

def run_paddle_ocr(image, bounding_box_annotator, label_annotator):
    # https://github.com/PaddlePaddle/PaddleOCR/tree/main
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False) # need to run only once to download and load model into memory
    outputs = ocr.ocr(np.array(image), cls=True)
    xyxy, label_confidence = [], []
    for out in outputs:
        x_min = np.min([a[0] for a in out[0][0]])
        x_max = np.max([a[0] for a in out[0][0]])
        y_min = np.min([a[1] for a in out[0][0]])
        y_max = np.max([a[1] for a in out[0][0]])
        xyxy.append([x_min, y_min, x_max, y_max])
        label_confidence.append(f"{out[0][1][0]} {out[0][1][1]:.2f}")
    detections = sv.Detections(np.array(xyxy), class_id=np.arange(len(xyxy)))
    annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=np.array(label_confidence),
    )   
    # annotated_frame.show()
    return annotated_frame

def main():
    image_file = "/home/data/teaganli/test_image/"
    image_names = [f for f in sorted(os.listdir(image_file)) if f.endswith(".jpg")]
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=10)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, 
                                        text_scale=3.0,
                                        text_thickness=3)
    
    for f in image_names:
        image = Image.open(os.path.join(image_file, f))
        img_easy_ocr = run_easy_ocr(image, bounding_box_annotator, label_annotator)
        img_paddle_ocr = run_paddle_ocr(image, bounding_box_annotator, label_annotator)

        img_easy_ocr.save(os.path.join(image_file, "easy_ocr_" + f))
        img_paddle_ocr.save(os.path.join(image_file, "paddle_ocr_" + f))


if __name__ == "__main__":
    main()
    