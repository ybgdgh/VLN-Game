from http import HTTPStatus
import dashscope
import re
from PIL import Image
import supervision as sv
import numpy as np
import easyocr

def draw_bounding_box(image, response, bounding_box_annotator, label_annotator):
    
    box_text = response.output.choices[0].message['content']
    w, h = image.size[0], image.size[1]   

    # iterate all boxes
    coordinates, labels = [], []
    for i in range(len(box_text)):
        if 'box' not in box_text[i]:
            continue
        coordinate = re.findall(r'<box>(.*?)</box>', box_text[i]['box'])
        label = re.findall(r'<ref>(.*?)</ref>', box_text[i]['box'])
        coordinate = [tuple(map(int, coord.replace('(', '').replace(')', '').split(','))) for coord in coordinate]
        x1, y1, x2, y2 = coordinate[0]
        x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
        
        coordinates.append([x1, y1, x2, y2])
        labels.append(label[0])    
    
    if len(coordinates) == 0:
        print(response)
        print("Not detected")
        exit()
        
    detections = sv.Detections(np.array(coordinates), class_id=np.arange(len(coordinates)))
    annotated_frame = bounding_box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=np.array(labels)
    )
    annotated_frame.show()
    

def simple_multimodal_conversation_call():
    """Simple single round multimodal conversation call.
    """
    
    local_image_path1 = "file:///home/data/teaganli/test_image/indoor_raw.jpeg"
    local_image_path2 = "file:///home/data/teaganli/test_image/indoor_detect.jpg"
    local_image_path3 = "file:///home/data/teaganli/test_image/IMG_8120.jpg"
    
    image_path = local_image_path1
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, 
                                        text_scale=0.5,
                                        text_thickness=1)
    
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"text":  f"Instruction: {instruction}. Here are two images. The first image shows what the robot sees, and the second image shows object segmentation annotations."},
    #             {"image": local_image_path1},
    #             {"image": local_image_path2},
    #             {"text": "Please identify the obj_i in the images that corresponds to the target object described in the instruction and provide a reason. Please respond in the format: 'Answer: obj_i, Reason: ...'. Note: 1. If the target object is in the image but not marked by a bounding box, respond with 'Answer: false_1, Reason: object hear, bbox not hear'. 2. If the target object is not in the image at all, respond with 'Answer: false_2, Reason: object not hear'."}
    #         ]
    #     }
    # ]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text":  f"please draw the bounding boxes of the table in front of the sofa and the small sofa."},
                {"image": local_image_path1},
            ]
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             # {"text":  f"你现在看到的是电梯井，请在图中画出E3电梯门。要求在途中画出E3不锈钢材质的电梯门的bounding box"},
    #             {"text":  f"你现在看到的是电梯井，请告诉我你现在在几楼，并且在图中画出楼层标志的bounding box，楼层标志通常是不锈钢材质挂在大理石墙面上"},
    #             {"image": image_path},
    #         ]
    #     }
    # ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',  # qwen-vl-max or qwen-vl-plus
                                                     messages=messages)
    image = Image.open(image_path.split("//")[1])
    draw_bounding_box(image, response, bounding_box_annotator, label_annotator)
    
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.


def ocr_qwen():
    local_image_path1 = "file:///home/data/teaganli/test_image/indoor_raw.jpeg"
    local_image_path2 = "file:///home/data/teaganli/test_image/indoor_detect.jpg"
    local_image_path3 = "file:///home/data/teaganli/test_image/IMG_8120.jpg"
    
    image_path = local_image_path3
    image = Image.open(image_path.split("//")[1])
    
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    outputs = reader.readtext(image)
    w, h = image.size[0], image.size[1]   
    xyxy, label, confidence = [], [], []
    for out in outputs:
        if out[2] > 0.7:
            # xyxy.append([*out[0][0], *out[0][2]])
            # convert to qwen bbox format
            x1, y1, x2, y2 = [*out[0][0], *out[0][2]]
            x1, y1, x2, y2 = (int(x1 / w * 1000), int(y1 / h * 1000), int(x2 / w * 1000), int(y2 / h * 1000))
            xyxy.append([[x1, y1], [x2, y2]])
            label.append(out[1])
            confidence.append(out[2])
    print(label, confidence)
    
    messages = [
        {
            "role": "user",
            "content": [ 
                # {"text":  f"你现在看到的是电梯井，请在图中画出E3电梯门。要求在途中画出E3不锈钢材质的电梯门的bounding box"},
                {"text":  f"你现在看到的是电梯井，我们先进行了OCR识别，识别到的字符：{label}，并且他们的bounding box在图上的位置是{xyxy}。通常电梯编号是一个英文字母与一个数字结合，比如A1，并且电梯编号通常在电梯门的上面。楼层的编号是二位数字，比如03.请告诉我你看到的电梯编号与楼层信息，并在图中画出E8电梯的位置"},
                {"image": image_path},
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',  # qwen-vl-max or qwen-vl-plus
                                                     messages=messages)
    print(response)


if __name__ == '__main__':
    simple_multimodal_conversation_call()
    # ocr_qwen()