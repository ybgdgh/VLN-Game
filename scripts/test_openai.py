import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key )
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1" #只增加这一行即可

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ]
# )

# print(response)


import base64
import os
from PIL import Image
import requests



# Function to encode the image
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# # Path to your image
# image_path = "/home/rickyyzliu/workspace/embodied-AI/habitat/2.jpeg"

# # Getting the base64 string
# base64_image = encode_image(image_path)

# headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# payload = {
#     "model": "gpt-4o", # gpt-4o gpt-4-vision-preview
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What's in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                 },
#             ],
#         }
#     ],
#     "max_tokens": 8,  # 300
# }

# response = requests.post(
#     "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
#     headers=headers,
#     json=payload,
# )

# print(response.json())


def ask_VLM(result_image=None, instruction=None):
        
    # instruction = "A standing man wearing blue clothes."
    # instruction = "The pillow on the sofa."
    instruction = "Please tell me how many elevators and their elevator ID, and mark the bounding box of the elevator door."

    pil_image = Image.open("/home/data/teaganli/test_image/IMG_8119.jpg")
    pil_segmented_image = Image.open("/home/data/teaganli/test_image/IMG_8119.jpg")
    
    
    
    import io
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

    # payload = {
    #     "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "What's in this image?"},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{img_str}"
    #                     },
    #                 },
    #             ],
    #         }
    #     ],
    #     "max_tokens": 8,  # 300
    # }

    # payload = {
    #     "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"Instruction: {instruction}. Here is an image:"},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{img_str}"
    #                     },
    #                 },
    #                 {"type": "text", "text": "Please identify the obj_i in the image that corresponds to the object described in the instruction and provide a reason. Please respond in the format: 'Answer: obj_i, Reason: ...'."}
    #             ],
    #         }
    #     ],
    #     "max_tokens": 20,  # 修改为适当的值
    # }

    # payload = {
    #     "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"Instruction: {instruction}. Here is an image:"},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{img_str}"
    #                     },
    #                 },
    #                 {"type": "text", "text": "Please identify the obj_i in the image that corresponds to the target object described in the instruction and provide a reason. If the target object is in the image but not marked by a bounding box, respond with 'Answer: false_1, Reason: object hear, bbox not hear'. If the target object is not in the image at all, respond with 'Answer: false_2, Reason: object not hear'. Otherwise, respond in the format: 'Answer: obj_i, Reason: ...'."}
    #             ],
    #         }
    #     ],
    #     "max_tokens": 20,  # 修改为适当的值
    # }


    # payload = {
    #     "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"Instruction: {instruction}. Here are two images. The first image shows what the robot sees, and the second image shows object segmentation annotations."},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{img_str}"
    #                     },
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{segmented_img_str}"
    #                     },
    #                 },
    #                 {"type": "text", "text": "Please identify the obj_i in the images that corresponds to the target object described in the instruction and provide a reason. Please respond in the format: 'Answer: obj_i, Reason: ...'. Note: 1. If the target object is in the image but not marked by a bounding box, respond with 'Answer: false_1, Reason: object hear, bbox not hear'. 2. If the target object is not in the image at all, respond with 'Answer: false_2, Reason: object not hear'."}
    #             ],
    #         }
    #     ],
    #     "max_tokens": 20,  # 修改为适当的值
    # }
    payload = {
        "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Instruction: {instruction}. "},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        },
                    },                    
                ],
            }
        ],
        "max_tokens": 20,  # 修改为适当的值
    }



    response = requests.post(
        "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    print(response.json())


if __name__ == "__main__":
    ask_VLM()