import numpy as np
import ast
import jsonschema
from jsonschema import validate
from utils.json_validate import schema
import time
import requests
import json
from arguments import get_args

import openai
from openai import OpenAIError
# openai.organization = "org-ZCl0HLFc8HhQCrrtvdiPccEz"
# openai.api_key = "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"
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1" #只增加这一行即可

gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo',  # 'gpt-3.5-turbo-0125'
            'gpt-4o',
            'gpt-4-turbo'
        ]           

args = get_args()

def chat_with_gpt(chat_history, gpt_type = args.gpt_type):
    retries = 10    
    while retries > 0:  
        try: 
            response = openai.ChatCompletion.create(
                model=gpt_name[args.gpt_type], 
                messages=chat_history,
                temperature=1.0,
            )

            response_message = response.choices[0].message['content']
            print(gpt_name[args.gpt_type] + " response: ")
            try:
                ground_json = ast.literal_eval(response_message)
                break
                
            except (SyntaxError, ValueError) as e:
                print(e)
        except OpenAIError as e:
            if e:
                print(e)
                print('Timeout error, retrying...')    
                retries -= 1
                time.sleep(5)
            else:
                raise e
            
    print(response_message)
    return response_message

def chat_with_llama3(user_prompt, chat_history):

    chat_history.append({"role": "user", "content": user_prompt})

    retries = 10    
    while retries > 0:  
        
        url = 'http://localhost:11434/api/chat'
        data = {
            "model": "llama3",
            "messages": chat_history,
            "stream": False,
            "format": "json",
            "options": {
                "seed": 101,
                "temperature": 0
            }
        }
        
        response = requests.post(url, json=data)
        decoded_string = response.content.decode('utf-8')
        response_message = json.loads(decoded_string)["message"]["content"]
        ground_json = ast.literal_eval(response_message)
                
        try: 
            validate(instance = ground_json, schema = schema)
        except jsonschema.exceptions.ValidationError as err:
            print("JSON data is invalid", err)
            print('Json format error, retrying...')    
            retries -= 1

            
    chat_history.append({"role": "assistant", "content": response_message})
    print(response_message)
    return response_message

