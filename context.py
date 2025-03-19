import requests
import numpy as np
import os
from datetime import datetime
from llm_inferences import query_llama_3_2_1b
import json

# Endpoints and API keys


def get_embedding(text):
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
    model = "sentence-transformers/all-MiniLM-L6-v2"
    endpoint = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    embed_headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    response = requests.post(endpoint, headers=embed_headers, json=payload)
    if response.status_code != 200:
        raise Exception("Embedding API error: " + response.text)
    embedding_data = response.json()
    
    if isinstance(embedding_data, list) and isinstance(embedding_data[0], list):
        embedding = [sum(x)/len(embedding_data) for x in zip(*embedding_data)]
    else:
        embedding = embedding_data
    return embedding

# import os
# import requests

# def get_embedding(text):
#     # Get API key from environment variables
#     segmind_api_key = "SG_56223ff5b3948309"

#     if not segmind_api_key:
#         raise Exception("SEGMIND_API_KEY is not set in environment variables.")

#     # API endpoint
#     model = "text-embedding-3-small"
#     endpoint = f"https://api.segmind.com/v1/{model}"

#     # API headers
#     headers = {
#         "x-api-key": segmind_api_key,
#         "Content-Type": "application/json"
#     }

#     # Request payload
#     payload = {"inputs": text}

#     response = requests.post(endpoint, headers=headers, json=payload)

#     if response.status_code != 200:
#         raise Exception("Embedding API error: " + response.text)

#     embedding_data = response.json()

#     # Ensure response is a valid embedding
#     if isinstance(embedding_data, dict) and "embedding" in embedding_data:
#         embedding = embedding_data["embedding"]
#     else:
#         raise Exception("Unexpected response format from Segmind API")

#     return embedding



def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def select_relevant_history(history, current_embedding, threshold = 0.5, max_tokens=500):
    scored_messages = []
    now = datetime.now()
    for msg in history:
        # Compute semantic similarity for the message
        try:

            hist_embedding = msg["embeddings"]
            hist_embedding = json.loads(hist_embedding)
            hist_embedding = hist_embedding["data"]
            sim = cosine_similarity(current_embedding, hist_embedding)
        except Exception as e:
            print("Error computing history embedding:", e)
            sim = 0.0

        # Compute recency factor using the created_at timestamp
        created_at = msg.get('created_at')
        try:
            dt = datetime.fromisoformat(created_at)
        except Exception as e:
            dt = now

        delta_minutes = (now - dt).total_seconds() / 60.0
        recency_factor = 1 / (1 + delta_minutes / 10)
        score = sim * recency_factor
        scored_messages.append((score, msg))
    

    scored_messages.sort(key=lambda x: x[0], reverse=True)
    
    # Select messages until reaching max_tokens
    selected = []
    token_count = 0
    for score, msg in scored_messages:
        ## Come back here to check
        print(score, msg["message"])
        if score < threshold:
            break
        tokens = len(msg['message'].split()) + len(msg['response'].split())
        if token_count + tokens > max_tokens + 30:
            break
        selected.append(msg)
        token_count += tokens
    return selected


def summarize_history(messages, message):
    if not messages:
        return ""
    text = ""
    for msg in messages:
        text += f"User: {msg['message']}\nAssistant: {msg['response']}\n"
    summary_prompt = f'''You are a summarizer. You need to summarize the conversation following the instructions.
    ##Text to Summarize: 
    {text}

    ##Message: 
    {message}

    ##Instructions: 
    Generate the summary for text given under Text to Summarize such that it would be helpful for generating next response for the message given under Message. 
    Also make sure to generate the summary relevant to our message. 
    '''

    summary_response = query_llama_3_2_1b(summary_prompt)
    return summary_response


def build_summary_context(history, message, recent_count=3, token_limit=500):
    if len(history) <= recent_count:
        prompt_context = "##Conversation history:\n"
        for entry in history:
            prompt_context += f"User: {entry['message']}\nAssistant: {entry['response']}\n"
        return prompt_context

    # Split history into older messages and recent messages
    older_messages = history[:-recent_count]
    recent_messages = history[-recent_count:]
    
    # Get summary of older messages
    summary = summarize_history(older_messages, message)
    print(summary)
    # Build prompt context
    prompt_context = "##Conversation summary:\n" + summary + "\n\n##Recent conversation:\n"
    token_count = 0
    for entry in recent_messages:
        tokens = len(entry['message'])
        if len(entry['response']) > 800:
            token_count += tokens
            prompt_context += f"User: {entry['message']}\nAssistant: Generated Image\n"
            continue
        token_count += tokens + len(entry['response'])
        if token_count > token_limit:
            break
        prompt_context += f"User: {entry['message']}\nAssistant: {entry['response']}\n"
    return prompt_context

def summarize_history_TtI(messages, message):
    if not messages:
        return ""
    text = ""
    for msg in messages:
        text += f"Question: {msg['message']}\n"
    summary_prompt = f'''You are a summarizer. You need to summarize the conversation following the instructions.
    ##Text to Summarize: 
    {text}

    ##Message: 
    {message}

    ##Instructions: 
    Generate the summary for text given under Text to Summarize such that it would be helpful for generating next response for the message given under Message. 
    Also make sure to generate the summary relevant to our message. 
    '''
    summary_response = query_llama_3_2_1b({
        "inputs": summary_prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5,
        }
    })
    print(summary_response)
    summary_response = summary_response[0]['generated_text'].split("Assistant:")[-1].strip()
    summary = summary_response.split("Summary:")[-1].strip()
    return summary

def build_text_to_image_context(history, message, current_embedding, recent_count=2, token_limit=500):    
    scored_messages = []
    now = datetime.now()
    for msg in history:
        # Compute semantic similarity for the message
        try:
            hist_embedding = get_embedding(msg['message'])
            sim = cosine_similarity(current_embedding, hist_embedding)
        except Exception as e:
            print("Error computing history embedding:", e)
            sim = 0.0

        # Compute recency factor using the created_at timestamp
        created_at = msg.get('created_at')
        try:
            dt = datetime.fromisoformat(created_at)
        except Exception as e:
            dt = now

        delta_minutes = (now - dt).total_seconds() / 60.0
        recency_factor = 1 / (1 + delta_minutes / 10)
        score = sim * recency_factor
        scored_messages.append((score, msg))
    

    scored_messages.sort(key=lambda x: x[0], reverse=True)
    
    # Select messages until reaching max_tokens
    selected = []
    token_count = 0
    for score, msg in scored_messages:

        ## Come back here to check
        print(score, msg["message"])
        if score < 0.40:
            break
        tokens = len(msg['message'].split())
        if token_count + tokens > token_limit:
            break
        selected.append(msg)
        token_count += tokens

    # Get summary of relevant messages
    summary = summarize_history_TtI(selected, message)
    # Build prompt context
    prompt_context = "##Conversation summary:\n" + summary + "\n"
    return prompt_context


def truncate_history(history, max_tokens=500):
    truncated = []
    token_count = 0
    for msg in reversed(history):
        token_count += len(msg['message'].split())  # Approximate token count
        if token_count > max_tokens:
            break
        truncated.insert(0, msg)
    return truncated
