from http import HTTPStatus
import dashscope
import os

api_key = os.environ.get('DASHSCOPE_API_KEY')

def simple_multimodal_conversation_call():
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


if __name__ == '__main__':
    simple_multimodal_conversation_call()