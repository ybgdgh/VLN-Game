from http import HTTPStatus
import dashscope


def simple_multimodal_conversation_call():
    """Simple single round multimodal conversation call.
    """
    
    local_image_path1 = "file:///home/data/teaganli/test_image/indoor_raw.jpeg"
    local_image_path2 = "file:///home/data/teaganli/test_image/indoor_detect.jpg"
    instruction = "table"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text":  f"Instruction: {instruction}. Here are two images. The first image shows what the robot sees, and the second image shows object segmentation annotations."},
                {"image": local_image_path1},
                {"image": local_image_path2},
                {"text": "Please identify the obj_i in the images that corresponds to the target object described in the instruction and provide a reason. Please respond in the format: 'Answer: obj_i, Reason: ...'. Note: 1. If the target object is in the image but not marked by a bounding box, respond with 'Answer: false_1, Reason: object hear, bbox not hear'. 2. If the target object is not in the image at all, respond with 'Answer: false_2, Reason: object not hear'."}
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