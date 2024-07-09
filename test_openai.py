import base64
import os
from PIL import Image
import requests
import io
import json
from PIL import Image, ImageDraw, ImageFont

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

def ask_VLM(result_image=None, instruction=None):
    pil_image = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/2.jpeg")
    pil_segmented_image = Image.open(
        "/home/rickyyzliu/workspace/embodied-AI/habitat/output_image.jpg"
    )

    # pil_image = Image.fromarray(np.uint8(result_image))
    # pil_image.show()

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    segmented_buffered = io.BytesIO()
    pil_segmented_image.save(segmented_buffered, format="JPEG")
    segmented_img_str = base64.b64encode(segmented_buffered.getvalue()).decode("utf-8")

    api_key = os.getenv("OPENAI_API_KEY")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


    # instruction = "A standing man wearing blue clothes."
    instruction = "The pillow on the sofa."
    # instruction = "The table in front of the sofa."
    # instruction = "The side table near the sofa."
    # instruction = "Table."
    # instruction = "Desk."
    text = (
        f"Here are two images. "
        "The first image shows what the robot sees, and the second image shows object segmentation annotations. "
        "Based on this information, please identify the object in the second image that corresponds to the target object "
        "described in the instruction and provide a reason. "
        "Please respond in the format: 'Answer: obj_i. Reason: ...'. "
        "Note: Use object IDs('obj_#') to describe the objects in the image instead of their actual names. "
        "If the target object is in the image but not marked by a bounding box, "
        "respond with 'Answer: false_1. Reason: ...'. If the "
        "target object is not in the image at all, respond with 'Answer: false_2, Reason: ... "
        f"Instruction: {instruction}."
    )

    payload = {
        "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{segmented_img_str}"
                        },
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
        "max_tokens": 100,  # 修改为适当的值
    }

    response = requests.post(
        "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"{answer}")
    answer_parts = answer.split(". Reason: ")
    obj_i = answer_parts[0].split(": ")[1]
    reason = answer_parts[1]

    if obj_i == "false_1":
        print("false_1")
    elif obj_i == "false_2":
        print("false_2")
    else:
        index = int(obj_i.split("_")[1])
        print(f"index: {index}")


    print(f"reason: {reason}")

    return obj_i


def ask_VLM_coord(result_image=None, instruction=None):
    pil_image = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/2.jpeg")

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")


    api_key = os.getenv("OPENAI_API_KEY")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # instruction = "A standing man wearing blue clothes."
    # instruction = "The pillow on the sofa."
    instruction = "The table in front of the sofa."
    # instruction = "The side table near the sofa."
    # instruction = "Table."
    # instruction = "Desk."
    image_size = pil_image.size
    print(image_size)
    text = (
        "Assume you are the navigation unit of a robot. The image shows what the robot sees. "
        "Your task is to identify the object in the image that corresponds to the target object described in the instruction. "
        "If you find the target object in the image, please accurately provide its coordinates (x, y) in the image, representing the center of the object. "
        f"Note: The coordinates of the image top left corner is (0, 0), and the image size (width, height) is {image_size}. "
        "Please ensure the accuracy of the coordinates and provide the output in JSON format, without any Markdown syntax, such as "
        '{"found_object": "True", "object_coordinate": ..., "reason": ...}'
        f"\n\nInstruction: {instruction}"
    )
    payload = {
        "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
        "max_tokens": 100,  # 修改为适当的值
    }

    response = requests.post(
        "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"{answer}")

    # 解析 JSON
    output_dict = json.loads(answer)
    found_object = output_dict["found_object"]
    object_coordinate = tuple(output_dict["object_coordinate"])
    reason = output_dict["reason"]
    # 输出解析结果
    print(f"Found object: {found_object}")
    print(f"Object coordinate: {object_coordinate}")
    print(f"Reason: {reason}")

    # 在图像上绘制坐标
    if found_object == "True":
        draw = ImageDraw.Draw(pil_image)
        radius = 10
        x, y = object_coordinate
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))

    pil_image.show()

    print("finished!")

def ask_VLM_patch(result_image=None, instruction=None):
    pil_image = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/2.jpeg")
    n, m = 8, 8
    width, height = pil_image.size
    patch_width, patch_height = width // n, height // m

    numbered_image = pil_image.copy()
    draw = ImageDraw.Draw(numbered_image)
    font = ImageFont.load_default()

    for i in range(n):
        for j in range(m):
            patch_id = f"{i+1}-{j+1}"
            x, y = i * patch_width, j * patch_height
            draw.rectangle([(x, y), (x + patch_width, y + patch_height)], outline="white", width=1)
            draw.text((x + patch_width // 2, y + patch_height // 2), patch_id, font=font, fill="white", anchor="mm")

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    segmented_buffered = io.BytesIO()
    numbered_image.save(segmented_buffered, format="JPEG")
    segmented_img_str = base64.b64encode(segmented_buffered.getvalue()).decode("utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


    # instruction = "The pillow on the sofa."
    # instruction = "A standing man wearing blue clothes."
    # instruction = "chair."
    # instruction = "The pillow on the sofa."
    # instruction = "The table in front of the sofa."
    instruction = "The side table behind the sofa."
    # instruction = "Table."
    # instruction = "Desk."
    text = (
        f"Here are two images. "
        "The first image shows what the robot sees, and the second image shows that this image is divided into n*m patches. "
        "Your task is to identify the patch in the second image that "
        "main contains the target object described in the instruction. If you find the target object, "
        "please accurately provide the identifier of the patch. If the target object is not in the image, "
        "suggest the most promising patch where further exploration is likely to reveal the target object. "
        "Please provide the output in JSON format, without any Markdown syntax, such as "
        '{"found_object": "True", "patch_id": "i-j", "reason": ...} or '
        '{"found_object": "False", "explore_patch_id": "i-j", "reason": ...}.'
        f"\n\nInstruction: {instruction}"
    )

    payload = {
        "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{segmented_img_str}"
                        },
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
        "max_tokens": 100,  # 修改为适当的值
    }

    response = requests.post(
        "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"{answer}")
    numbered_image.show(title="Numbered Image")

if __name__ == "__main__":
    ask_VLM_patch()

