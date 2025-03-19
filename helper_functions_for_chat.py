import os
import base64
import tempfile
import requests
import numpy as np
from datetime import datetime
from flask import jsonify

from wikipedia_data_extraction import collection

from llm_inferences import (
    query_meta_llama_vision_openrouter,
    query_llama_3_3_70b_versatile,
    query_deepseek_r1_zero,
    query_deepseek_r1,
    query_deepseek_chat,
    query_google_gemini_flash,
    query_meta_llama_vision_cf,
    query_m2m100_translation,
    query_deepseek_r1_distill_qwen,
    query_mixtral_chat,
    query_whisper_transcription,
    query_llama_3_2_1b,
    query_llama_3_2_3b,
    query_google_gemini_flash_openrouter,
    query_qwen_2_5_vl_72b_instruct_openrouter,
    query_black_forest_flux_schnell,
    query_meta_llama_3_1_8B,
    query_sdxl_realistic,
    query_grok2_vision,
    query_olympic_coder_32B,
    query_olympic_coder_7B,
    query_gemma_3_4b,
    query_gemma_3_1b,
    query_qwen_2_5_coder_32b
)

from context import (
    get_embedding,
    cosine_similarity,
    select_relevant_history,
    summarize_history,
    build_summary_context,
    truncate_history,
    build_text_to_image_context
)

from prompts import(
    prompt_translation,
    prompt_coding,
    prompt_summarizing,
    prompt_reasoning,
    prompt_best_mode,
    prompt_general_mode
)


MISTRAL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
default_headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query_mistral(payload):
    response = requests.post(MISTRAL_ENDPOINT, headers=default_headers, json=payload)
    response = response.json()
    return response[0]['generated_text']


def no_model_selection(mode):
    mode_dic = {
        "General": "Google Gemma 3 4B",
        "Translation": "Llama 3.2 1B",
        "Coding": "Qwen 2.5 Coder 32B",
        "Summarizing": "Llama 3.2 1B",
        "WebSearch": "Meta Llama 3.1 8B",
        "Reasoning and Research": "Deepseek Chat",
        "Text to Image": "Black Forest Flux Schnell",
        "Best Quality": "Mixtral Chat"
    }
    return mode_dic.get(mode, "Llama 3.2 1B")

def get_context_mode(mode):
    context_mode_map = {"General": "relevant",
        "Translation": "summary",
        "Coding": "relevant",
        "Summarizing": "summary",
        "WebSearch": "summary",
        "Reasoning and Research": "relevant",
        "Text to Image": "summary",
        "Best Quality": "relevant"}
    try:
        return context_mode_map[mode]
    except:
        return 'None'

def best_quality(prompt, model_name):
    message, prompt = prompt[0],prompt[1]
    retrieved_results = collection.query(
        query_texts=[message],
        n_results=1
    )

    documents = retrieved_results["documents"][0]
    doc_text = ""
    for doc in documents:
        doc_text += doc

    print(doc_text)
    reasoning = None
    if model_name not in ["Deepseek R1",'Deepseek R1 zero']:
        reasoning = query_deepseek_r1_distill_qwen(prompt)
    
    if reasoning:
        prompt = f'''You are chatbot which has vast knowledge of every field. You will be provided with New Message (that is user query), Conversation History, Reasoning and Knowledge base. 
        Your task is generate State of Art response for the New Message considering Conversation history if relevant to the New Message. You must use Reasoning provided and Knowledge base to make your response more informative.   
        
        {message}

        ##Reasoning:
        {reasoning}

        ##Knowledge base:
        {doc_text}
        '''
    else:
        prompt = f'''You are chatbot which has vast knowledge of every field. You will be provided with New Message (that is user query), Conversation History and Knowledge base. 
        Your task is generate State of Art response for the New Message considering Conversation history if relevant to the New Message. You must use Knowledge base if relevant to the New Message to make your response more informative.   
        
        {message}

        ##Knowledge base:
        {doc_text}
        '''
    return [prompt, reasoning]


def voice_processing(voice_data):
    try:
        if isinstance(voice_data, str) and voice_data.startswith("data:"):
            voice_data = voice_data.split(",")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(base64.b64decode(voice_data))
            temp_audio_path = temp_audio.name
            
       
        transcript = query_whisper_transcription(temp_audio_path)
        if isinstance(transcript, list):
            first_item = transcript[0]
            if isinstance(first_item, dict) and "generated_text" in first_item:
                transcript = first_item["generated_text"]
            else:
                transcript = first_item
        elif isinstance(transcript, dict) and "generated_text" in transcript:
            transcript = transcript["generated_text"]
        elif isinstance(transcript, str):
            transcript = transcript
        else:
            transcript = str(transcript)
        
        return transcript

    except:
        return jsonify({"error": "Error getting Voice Data"}), 400


def process_context(context_mode, mode, full_history, current_embedding, message, image_data):
    prompt = ""
    if mode == "Text to Image":
        prompt = build_text_to_image_context(full_history, message, current_embedding, recent_count=3, token_limit=500)
        prompt += f"\n##New message: {message}\nAssistant:"
        return prompt
    if image_data:
        prompt = build_summary_context(full_history, message, recent_count=1, token_limit=500)
        prompt += f"\n##New message: {message}\nAssistant:"
        return prompt

    if mode == "Summarizing":
        prompt = build_summary_context(full_history, message, recent_count=1, token_limit=500)

    elif context_mode == "summary":
        prompt = build_summary_context(full_history, message, recent_count=3, token_limit=500)
    elif context_mode == "relevant":
        if current_embedding is not None:
            selected_history = select_relevant_history(full_history[2:], message, current_embedding, max_tokens=500)
        else:
            selected_history = full_history[2:4]
        prompt = "##Conversation history:\n"
        for entry in full_history[:2]:
            if len(entry['response']) > 10000: # Not adding image
                prompt += f"User: {entry['message']}\nAssistant: Generated Image\n"
                continue
            prompt += f"User: {entry['message']}\nAssistant: {entry['response']}\n"

        for entry in selected_history:
            if len(entry['response']) > 10000: # Not adding image
                prompt += f"User: {entry['message']}\nAssistant: Generated Image\n"
                continue
            prompt += f"User: {entry['message']}\nAssistant: {entry['response']}\n"
    else:
        if current_embedding is not None:
            selected_history = select_relevant_history(full_history, current_embedding, threshold = 0.4, max_tokens=700)
        else:
            selected_history = full_history[:7]
        prompt = build_summary_context(selected_history, message, recent_count=2, token_limit=700)
    
    prompt += f"\n##New message: {message}\nAssistant:"
    return prompt


def process_mode(mode, message, source_lang, target_lang):
    if mode == "Coding":
        prompt = prompt_coding(message)
    
    elif mode == "Summarizing":
        prompt = prompt_summarizing(message)
    
    elif mode == "Reasoning and Research":
        prompt = prompt_reasoning(message)
    elif mode == "General":
        prompt = prompt_general_mode(message)
    elif mode == "Best Quality":
        prompt = prompt_best_mode(message)
        print("coming here")
        return [message, prompt]
    else:
        prompt = message

    return prompt    


def process_model(mode, model_name, prompt, source_lang, target_lang, image_data, voice_data):
    if model_name == "M2M 100 translation":
        inference_result = query_m2m100_translation(prompt, source_lang=source_lang or "en", target_lang=target_lang or "hi")
    elif model_name == "Llama 3.3 70B versatile":
        inference_result = query_llama_3_3_70b_versatile(prompt)
    elif model_name == "Deepseek R1 zero":
        inference_result = query_deepseek_r1_zero(prompt)
    elif model_name == "Deepseek R1":
        inference_result = query_deepseek_r1(prompt)
    elif model_name == "Deepseek R1 Distill Qwen":
        inference_result = query_deepseek_r1_distill_qwen(prompt)
    elif model_name == "Mixtral Chat":
        inference_result = query_mixtral_chat([{"role": "user", "content": prompt}], stream=False)
    elif model_name == "Deepseek Chat":
        inference_result = query_deepseek_chat(prompt)
    elif model_name == "Google Gemini Flash":
        inference_result = query_google_gemini_flash(prompt, image_data)
    elif model_name == "Meta Llama Vision Openrouter":
        inference_result = query_meta_llama_vision_openrouter(prompt, image_data)
    elif model_name == "Meta Llama Vision CF":
        inference_result = query_meta_llama_vision_cf(prompt)
    elif model_name == "Whisper Transcription":
        if voice_data:
            if isinstance(voice_data, str) and voice_data.startswith("data:"):
                voice_data = voice_data.split(",")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(base64.b64decode(voice_data))
                temp_audio_path = temp_audio.name
            inference_result = query_whisper_transcription(temp_audio_path)
        else:
            return jsonify({"error": "No voice data provided"}), 400
    elif model_name == "Llama 3.2 1B":
        inference_result = query_llama_3_2_1b(prompt)
    elif model_name == "Qwen 2.5 Coder 32B":
        inference_result = query_qwen_2_5_coder_32b(prompt)
    elif model_name == "Olypic Coder 32B":
        inference_result = query_olympic_coder_32B(prompt)
    elif model_name == "Olypic Coder 7B":
        inference_result = query_olympic_coder_7B(prompt)
    elif model_name == "SDXL 1.0 Yamers Realistic":
        inference_result = query_sdxl_realistic(prompt)
    elif model_name == "Google Gemma 3 1B":
        inference_result = query_gemma_3_1b(prompt)
    elif model_name == "Google Gemma 3 4B":
        inference_result = query_gemma_3_4b(prompt)
    elif model_name == "Llama 3.2 3B":
        inference_result = query_llama_3_2_3b(prompt)
    elif model_name == "Google Gemini Flash Openrouter":
        inference_result = query_google_gemini_flash_openrouter(prompt, image_data)
    elif model_name == "Qwen 2.5 VL 72B Instruct Openrouter":
        inference_result = query_qwen_2_5_vl_72b_instruct_openrouter(prompt, image_data)
    elif model_name == "Meta Llama 3.1 8B":
        inference_result = query_meta_llama_3_1_8B(prompt)
    elif model_name == "Grok 2 Vision":
        inference_result = query_grok2_vision(prompt, image_data)
    elif model_name == "Black Forest Flux Schnell":
        inference_result = query_black_forest_flux_schnell(prompt)
    else:
        inference_result = query_mistral({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "temperature": 0.7}
        })
    
    return inference_result

