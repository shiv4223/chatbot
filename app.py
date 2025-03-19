import os
import uuid
import json
import base64
import tempfile
import http.client
import urllib.parse
import requests
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from supabase import create_client
from dotenv import load_dotenv
from flask_cors import CORS
from image_url import upload_file_to_google_drive
import redis



from helper_functions_for_chat import(
    no_model_selection, 
    best_quality,
    get_context_mode,
    voice_processing,
    process_context,
    process_mode,
    process_model,
)

from context import (
    get_embedding,
    cosine_similarity
)

# Initialize Flask app and environment variables
app = Flask(__name__)
CORS(app, origins=["*"], methods=["POST", "GET"], allow_headers=["*"])
load_dotenv()

# Initialize Supabase and Redis clients
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

redis_client = redis.Redis.from_url(os.getenv('REDIS_URL'))


def store_at_database(user_id, conversation_id, message, llm_response, mode, image_data, current_embedding = None, reasoning= None):
    if llm_response == "Server Error":
        return jsonify({"response": llm_response, "conversation_id": conversation_id})
    
    chat_res = supabase.table('chats').insert({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message": message,
        "response": llm_response,
        "created_at": datetime.utcnow().isoformat(),
        "mode": mode,
        "image": image_data,
        "embeddings": current_embedding,
        "reasoning" : reasoning
    }).execute()
    if not chat_res.data:
        print("Error saving chat:", chat_res)

    return jsonify({"response": llm_response, "conversation_id": conversation_id, "reasoning": reasoning})
    

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    res = supabase.table("users").select("*").eq("username", username).execute()
    if res.data:
        return jsonify({"error": "User already exists"}), 400

    insert_res = supabase.table("users").insert({
        "username": username,
        "password": password
    }).execute()

    if not insert_res.data:
        print("Registration Error:", insert_res)
        return jsonify({"error": "Registration failed"}), 500

    user = insert_res.data[0]
    print("User registered successfully:", user)
    return jsonify({"message": "Registration successful", "user_id": user.get("id")}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    res = supabase.table("users").select("*").eq("username", username).execute()
    if not res.data:
        return jsonify({"error": "User not found"}), 404

    user = res.data[0]
    if user.get("password") != password:
        return jsonify({"error": "Incorrect password"}), 401

    print("User logged in:", user)
    return jsonify({"message": "Login successful", "user_id": user.get("id")}), 200

@app.route('/conversations/<user_id>', methods=['GET'])
def get_conversations(user_id):
    res = supabase.table('conversations').select('id, title, created_at') \
           .eq('user_id', user_id) \
           .order("created_at.desc") \
           .execute()
    return jsonify(res.data)

@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    res = supabase.table('chats').select('message, response, created_at, mode, image, reasoning') \
           .eq('conversation_id', conversation_id) \
           .order("created_at.asc") \
           .execute()
    return jsonify(res.data)


@app.route('/chat', methods=['POST'])
def chat_handler():
    # if request.content_type.startswith("multipart/form-data"):
    #     # Get non-file fields from request.form
    #     user_id = request.form.get("user_id")
    #     message = request.form.get("message")
    #     context_mode = request.form.get("context_mode", "relevant")
    #     conversation_id = request.form.get("conversation_id")
    #     model_name = request.form.get("model")
    #     mode = request.form.get("mode", context_mode)
    #     source_lang = request.form.get("source_lang")
    #     target_lang = request.form.get("target_lang")
        
    #     # Get file fields from request.files
    #     image_file = request.files.get("image")
    #     voice_file = request.files.get("voice")
        
    #     # Read file content if file is provided
    #     image_data = image_file.read() if image_file else None
    #     voice_data = voice_file.read() if voice_file else None
    data = request.json
    user_id = data.get("user_id")
    message = data.get("message")
    conversation_id = data.get("conversation_id")
    model_name = data.get("model")
    mode = data.get("mode")
    context_mode = get_context_mode(mode)
    source_lang = data.get("source_lang")
    target_lang = data.get("target_lang")
    image_data = data.get("image")
    voice_data = data.get("voice")

    # if image_data:
    #     print(image_file)
    #     image_url = upload_file_to_google_drive(image_file)

    initial_message = message

    voice_info = None
    if voice_data:
        transcript = voice_processing(voice_data)
        if not message:
            message = transcript
            initial_message = "**Voice Message: **" + transcript

        else:
            voice_info = transcript
            message += "\n Voice Info: " + voice_info
            initial_message += "\n **Voice Message:** " + voice_info
        
        if mode == "Voice Transcript":
            message = "**Voice Message :**" + (message if voice_info is None else voice_info)
            transcript = "**Your Voice Transcript is:**" + transcript
            return store_at_database(user_id, conversation_id, message, transcript, mode, image_data)
    

    print(initial_message, message)

    if not user_id:
        return jsonify({"error": "Your Authentication Error"}), 400

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        title = initial_message if len(initial_message) < 50 else initial_message[:47] + "..."
        conv_res = supabase.table('conversations').insert({
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        if not conv_res.data:
            print("Error creating conversation:", conv_res)
            return jsonify({"error": "Failed to create conversation"}), 500
    
    full_history = []
    if mode != "Translation" and mode != "Voice Transcript":
        try:
            history_response = supabase.table('chats').select("id",'message, response, created_at',"embeddings") \
                .eq('conversation_id', conversation_id) \
                .order("created_at.asc") \
                .execute()
        except Exception as e:
            print("Supabase Error:", e)
            return jsonify({"error": "Database error"}), 500

        full_history = history_response.data if history_response.data else []

    ##Finding in Cache for match: 
    try:
        current_embedding = get_embedding(message)
        current_embedding = current_embedding.tolist()
    except Exception as e:
        print("Embedding error:", e)
        current_embedding = None
    if mode != 'Translation' and mode != "Voice Transcript" and mode != "Text to Image" and not image_data:
        print("finding in cache")
        cached_response = None
        if current_embedding is not None:
            cache_keys = redis_client.keys("cache:*")
            similarity_threshold = 0.92
            for key in cache_keys:
                cached_data = redis_client.hgetall(key)
                cached_embedding_str = cached_data.get(b'embedding')
                if not cached_embedding_str:
                    continue
                cached_embedding = [float(x) for x in cached_embedding_str.decode('utf-8').split(',')]
                sim = cosine_similarity(current_embedding, cached_embedding)
                if sim >= similarity_threshold:
                    cached_response = cached_data.get(b'response').decode('utf-8')
                    print(f"Cache hit with similarity {sim}")
                    break

        if cached_response:
            store_at_database(user_id, conversation_id, initial_message, cached_response, mode, image_data, current_embedding)
            return jsonify({"response": cached_response, "conversation_id": conversation_id})
        
    elif mode == "Text to Image":
        try:
            current_embedding = get_embedding(message)
            current_embedding = current_embedding.tolist()
        except Exception as e:
            print("Embedding error:", e)
            current_embedding = None
    else:
        current_embedding = None

    print("In context Processing")
    #Context processing:
    if mode != "Translation" and mode != "Voice Transcript":
        message = process_context(context_mode, mode, full_history, current_embedding, message, image_data)
    
    print("In Mode Processing")
    ## Mode processing:
    prompt = process_mode(mode, message, source_lang, target_lang)

    reasoning = None
    if mode == "Best Quality":
        prompt = best_quality(prompt, model_name)
        reasoning = prompt[1]
        prompt = prompt[0]


    if model_name == "No Selection":
        model_name = no_model_selection(mode)

    print(reasoning)
    llm_response = process_model(mode, model_name, prompt, source_lang, target_lang, image_data, voice_data)
    # Robust extraction of generated text
    # if isinstance(inference_result, list):
    #     first_item = inference_result[0]
    #     if isinstance(first_item, dict) and "generated_text" in first_item:
    #         llm_text = first_item["generated_text"]
    #     else:
    #         llm_text = first_item
    # elif isinstance(inference_result, dict) and "generated_text" in inference_result:
    #     llm_text = inference_result["generated_text"]
    # elif isinstance(inference_result, str):
    #     llm_text = inference_result
    # else:
    #     llm_text = str(inference_result)

    # llm_response = llm_text.split("Assistant:")[-1].strip()
    if model_name == "Deepseek R1 zero" or model_name == "Deepseek R1":
        reasoning = llm_response[1]
        llm_response = llm_response[0]


    if current_embedding is not None and mode != "Text to Image" and message is not None and llm_response and llm_response != 'Server Error':
        cache_id = "cache:" + str(uuid.uuid4())
        embedding_str = ",".join(map(str, current_embedding))
        redis_client.hset(cache_id, mapping={
            "message": str(message),
            "embedding": embedding_str,
            "response": str(llm_response)
        })
        redis_client.expire(cache_id, 3600)

    if mode == "Text to Image" and len(llm_response) < 1000:
        print(llm_response)


    if current_embedding:
        current_embedding = {"data": current_embedding}
        current_embedding = json.dumps(current_embedding, indent=1)

    print(reasoning)
    return store_at_database(user_id, conversation_id, initial_message, llm_response, mode, image_data, current_embedding, reasoning)

    

if __name__ == '__main__':
    app.run()
