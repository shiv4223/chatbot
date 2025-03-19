import os
import json
import requests
from groq import Groq 
from openai import OpenAI 
import re, json

#OpenRouter
def query_meta_llama_vision_openrouter(prompt_text, image_url):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        # Optionally include HTTP-Referer and X-Title if desired
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    if image_url:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }

    else:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": {prompt_text}
                }
            ]
        }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    try:
        return response["choices"][0]["message"]["content"]
    except:
        return "Server Error"
    


#OpenRouter
def query_google_gemini_flash_openrouter(prompt_text, image_url):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        # Optionally include HTTP-Referer and X-Title if desired
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    if image_url:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }

    else:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": {prompt_text}
                }
            ]
        }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    try:
        return response["choices"][0]["message"]["content"]
    except:
        return "Server Error"

#OpenRouter
def query_qwen_2_5_vl_72b_instruct_openrouter(prompt_text, image_url):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        # Optionally include HTTP-Referer and X-Title if desired
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    if image_url:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }

    else:
        data = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": {prompt_text}
                }
            ]
        }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    try:
        return response["choices"][0]["message"]["content"]
    except:
        return "Server Error"

#GROQ
def query_llama_3_3_70b_versatile(prompt_text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


#GROQ
def query_llama_3_2_1b(prompt_text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        model="llama-3.2-1b-preview",
    )
    return chat_completion.choices[0].message.content


#GROQ
def query_llama_3_2_3b(prompt_text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        model="llama-3.2-3b-preview",
    )
    return chat_completion.choices[0].message.content



# OpenRouter
def query_deepseek_r1_zero(prompt_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    data = {
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    content = response["choices"][0]['message'].get("content", "").strip() 
    
    cleaned_string = re.sub(r"\\boxed\{(.*)\}", r"\1", content, flags=re.DOTALL)

    json_string = re.sub(r"```json|```", "", cleaned_string).strip()
    print(json_string)
    # try:
    #     data_dict = json.loads(json_string)
    #     answer = data_dict.get("answer", "No answer found")
    # except json.JSONDecodeError as e:
    #     answer = f"JSON Parsing Error: {e}"
    
    try:
        return [json_string,response["choices"][0]['message']["reasoning"]]
        # return answer
    except:
        "Server Error"

def query_qwen_2_5_coder_32b(prompt, temperature=0.8, max_completion_tokens=1024, top_p=1, stop=None, stream=False):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    data =  [{"role": "user", "content": prompt}]
    
    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages= data,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=stream,
        stop=stop,
    )
    
    return completion.choices[0].message.content


def query_gemma_3_1b(prompt):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="google/gemma-3-1b-it:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content


def query_gemma_3_4b(prompt):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="google/gemma-3-4b-it:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content


def query_olympic_coder_7B(prompt):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="open-r1/olympiccoder-7b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content

def query_olympic_coder_32B(prompt):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>",
    },
    extra_body={},
    model="open-r1/olympiccoder-32b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content

# OpenRouter
def query_deepseek_r1(prompt_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    try:
        return [response["choices"][0]["message"]["content"], response["choices"][0]["message"]["reasoning"]]
        # return response["choices"][0]["message"]["content"]
    except:
        return "Server Error"


# OpenRouter
def query_deepseek_chat(prompt_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('SITE_URL', ''),
        "X-Title": os.getenv('SITE_NAME', '')
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    try:
        return response["choices"][0]["message"]['content']
    except:
        return "Server Error"

# OpenRouter
def query_google_gemini_flash(prompt_text, image_url=None):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": os.getenv('SITE_URL', ''),
            "X-Title": os.getenv('SITE_NAME', '')
        },
        extra_body={},
        model="google/gemini-2.0-flash-thinking-exp:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )
    return completion.choices[0].message.content

#CloudFlare
def query_meta_llama_vision_cf(prompt_text):
    # ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/a1dcb00fafe16f342a49368e2ccb1145/ai/run/@cf/meta/llama-3.2-11b-vision-instruct"
    headers = {"Authorization": f"Bearer GtqCJ3w5SQzajqNh00k1U7IjyZRAVI95ea1vTKjA"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": prompt_text}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    return response

#CloudFlare
def query_m2m100_translation(text, source_lang, target_lang):
    # ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/a1dcb00fafe16f342a49368e2ccb1145/ai/run/"
    headers = {"Authorization": f"Bearer GtqCJ3w5SQzajqNh00k1U7IjyZRAVI95ea1vTKjA"}
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    url = API_BASE_URL + "@cf/meta/m2m100-1.2b"
    response = requests.post(url, headers=headers, json=payload)
    response = response.json()
    try:
        return response["result"]['translated_text']
    except: 
        return str(response)


#CloudFlare
def query_black_forest_flux_schnell(prompt_text):
    # ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/a1dcb00fafe16f342a49368e2ccb1145/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {
        "Authorization": f"Bearer GtqCJ3w5SQzajqNh00k1U7IjyZRAVI95ea1vTKjA",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt_text}
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    try:
        return response["result"]['image']
    except:
        return response["result"]


#CloudFlare
def query_meta_llama_3_1_8B(prompt_text):
    # ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/a1dcb00fafe16f342a49368e2ccb1145/ai/run/@cf/meta/llama-3.1-8b-instruct"
    headers = {"Authorization": f"Bearer GtqCJ3w5SQzajqNh00k1U7IjyZRAVI95ea1vTKjA"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": prompt_text}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    try:
        return response['result']['response']
    except:
        return "Server Error"

#CloudFlare
def query_deepseek_r1_distill_qwen(prompt_text):
    # ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/a1dcb00fafe16f342a49368e2ccb1145/ai/run/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
    headers = {"Authorization": f"Bearer GtqCJ3w5SQzajqNh00k1U7IjyZRAVI95ea1vTKjA"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": prompt_text}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    print(response)
    try:
        return response["result"]["response"]
    except: 
        return "Server Error"


#GROQ
def query_whisper_transcription(file_path):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with open(file_path, "rb") as file:
        file_contents = file.read()
    
    transcription = client.audio.transcriptions.create(
        file=(file_path, file_contents),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    
    return transcription.text

#GROQ
def query_mixtral_chat(messages, temperature=1, max_completion_tokens=1024, top_p=1, stop=None, stream=False):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=stream,
        stop=stop,
    )
    if stream:
        output = ""
        for chunk in completion:
            output += chunk.choices[0].delta.content or ""
        return output
    else:
        return completion.choices[0].message.content
    


def query_mistral(payload):
    MISTRAL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    default_headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    response = requests.post(MISTRAL_ENDPOINT, headers=default_headers, json=payload)
    print("Hugging Face Response:", response.text)
    response = response.json()
    return response[0]['generated_text']


def query_sdxl_realistic(prompt):
    data = {
    "prompt": prompt,
    "samples": 1,
    "scheduler": "DPM++ 2M Karras",
    "num_inference_steps": 21,
    "guidance_scale": 7,
    "seed": 565481734,
    "img_width": 640,
    "img_height": 720,
    "base64": True
        }

    url = "https://api.segmind.com/v1/sdxl1.0-yamers-realistic"
    api_key = "SG_5653225ab519aa54"
    headers = {'x-api-key': api_key}

    response = requests.post(url, json=data, headers=headers)
    response = response.content
    json_str = response.decode("utf-8")
    data = json.loads(json_str)
    image_data = data["image"]
    print(image_data)
    return image_data


def query_grok2_vision(prompt, image_data):
    api_key = "SG_56223ff5b3948309"
    # api_key = "SG_5653225ab519aa54"
    url = "https://api.segmind.com/v1/grok-2-vision"

    
    data = {
            "messages": []
            }
    
    if image_data:
        data["messages"].append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        })
    
    else:
        data = {
            "messages": [
            {
                "role": "user",
                    "content": prompt
                }
            ]}


    response = requests.post(url, json=data, headers={'x-api-key': api_key})
    response = response.json()
    return response['choices'][0]["message"]["content"]