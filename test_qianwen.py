from http import HTTPStatus
import dashscope
import os

import base64
import os
import requests
import io
import json
from PIL import Image, ImageDraw, ImageFont
import re


api_key = os.environ.get('DASHSCOPE_API_KEY')

def ask_VLM():
    """Simple single round multimodal conversation call.
    """
    
    local_image_path1 = "file:///home/rickyyzliu/workspace/embodied-AI/habitat/2.jpeg"
    local_image_path2 = "file:///home/rickyyzliu/workspace/embodied-AI/habitat/output_image.jpg"
    # instruction = "A standing man wearing blue clothes."
    # instruction = "The pillow on the sofa."
    instruction = "The table in front of the sofa."
    # instruction = "The side table near the sofa."
    # instruction = "The side table behand the sofa."
    # instruction = "Table."
    # instruction = "Desk."
    text = (
        f"Here are two images. "
        "The first image shows what the robot sees, and the second image shows object segmentation annotations. "
        "Based on this information, please identify the object in the second image that corresponds to the target object "
        "described in the instruction and provide a reason, respon with: 'Answer: obj_i. Reason: ...'. "
        "Note: Use object IDs('obj_#') to describe the objects in the image instead of their actual names. "
        "Note: 1. If the target object is in the first image but not marked by a bounding box in the second image, "
        "respond with 'Answer: false_1. Reason: ...'. 2. If the "
        "target object is not in the image at all, respond with 'Answer: false_2, Reason: ... "
        f"Instruction: {instruction}."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"image": local_image_path1},
                {"image": local_image_path2},
                {"text": text}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',  # qwen-vl-max or qwen-vl-plus
                                                     messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.


def ask_VLM_coord(result_image=None, instruction=None):
    pil_image = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/habitat.png")
    image_size = pil_image.size
    w = image_size[0]
    h = image_size[1]

    local_image_path1 = "file:///home/rickyyzliu/workspace/embodied-AI/habitat/habitat.png"
    text = (
        # "Frame the table in front of the sofa."
        # "Frame the table behind the sofa."
        # "Frame E5 elevator door in the image using bbox."
        # "框出E8电梯门."
        # "如果图像中有行人, 请在图像上将该目标用BBox框出来; 如果没有, 则输出文本: “没有行人”。"
        # "图片里有行人吗"
        "Frame the TV. "
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"image": local_image_path1},
                {"text": text}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',  # qwen-vl-max or qwen-vl-plus
                                                     messages=messages)
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.

    box_str = response["output"].choices[0].message.content[0]["box"]
    match = re.search(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', box_str)
    x1, y1, x2, y2 = map(int, match.groups())
    x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
    print(f"{x1}, {y1}, {x2}, {y2}")
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)

    pil_image.show()
    print("finished!")


if __name__ == '__main__':
    # ask_VLM()
    ask_VLM_coord()