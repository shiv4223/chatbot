import numpy as np
import json
import spacy
from sklearn.cluster import KMeans
from keybert import KeyBERT
from datetime import datetime, timedelta
from collections import deque
import requests
import os
from llm_inferences import query_llama_3_2_1b
from dateutil import parser
from sentence_transformers import SentenceTransformer

embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    embedding_data = embeddings_model.encode(text) 
    
    if isinstance(embedding_data, list) and isinstance(embedding_data[0], list):
        embedding = [sum(x)/len(embedding_data) for x in zip(*embedding_data)]
    else:
        embedding = embedding_data
    return embedding

# Load NLP model for NER and keywords
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

# Cache for precomputed summaries
context_cache = deque(maxlen=5)  # Stores last 5 precomputed summaries

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Adaptive Similarity Threshold
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.3, alpha=0.05):
        self.threshold = initial_threshold
        self.alpha = alpha

    def update(self, success):
        if success:
            self.threshold = min(0.9, self.threshold + self.alpha)
        else:
            self.threshold = max(0.3, self.threshold - self.alpha)

adaptive_threshold = AdaptiveThreshold()

# Keyword & NER Extraction
def extract_keywords_and_entities(text):
    doc = nlp(text)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in {"ORG", "GPE", "PERSON", "PRODUCT"}]
    return set([kw[0] for kw in keywords] + entities)

# Embedding Clustering for Efficient History Retrieval
def cluster_history_embeddings(history, current_embedding, num_clusters=3):
    if len(history) < num_clusters:
        return history  # Not enough data to cluster

    embeddings = []
    for msg in history:
        emb = json.loads(msg["embeddings"])["data"]
        if emb:
            embeddings.append(emb)
        
    if len(embeddings) == 0:
        return []
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Find cluster closest to current_embedding
    current_cluster = kmeans.predict([current_embedding])[0]

    return [msg for i, msg in enumerate(history) if labels[i] == current_cluster]

#Select Relevant History
def select_relevant_history(history, message, current_embedding, max_tokens=500):
    now = datetime.now()
    scored_messages = []
    
    # Cluster history to improve selection efficiency
    clustered_history = cluster_history_embeddings(history, current_embedding)
    if len(clustered_history) == 0:
        return []
    current_keywords = extract_keywords_and_entities(message)
    for msg in clustered_history:
        try:
            hist_embedding = json.loads(msg["embeddings"])["data"]
            sim = cosine_similarity(current_embedding, hist_embedding)
        except:
            sim = 0.0

        # Compute recency factor
        created_at = msg.get('created_at', now.isoformat())
        try:
            dt = parser.parse(created_at)  # More flexible parsing
        except Exception as e:
            print(f"Error parsing timestamp: {created_at}, defaulting to now. Error: {e}")
            dt = datetime.now()
        delta_minutes = (now - dt).total_seconds() / 60.0
        # recency_factor = np.exp(-delta_minutes / 20)  # Exponential Decay
        recency_factor = 1 / (1 + delta_minutes / 10)

        # Compute keyword & entity overlap
        user_keywords = extract_keywords_and_entities(msg["message"])
        response_keywords = extract_keywords_and_entities(msg["response"])
        keyword_match = len(current_keywords & user_keywords)
        keyword_match += len(current_keywords & response_keywords)

        # Final scoring function
        score = sim + (0.3 * keyword_match) + 0.6 * recency_factor
        scored_messages.append((score, msg))

    # Sort by score and apply adaptive threshold
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    threshold = adaptive_threshold.threshold
    print(threshold)
    selected = []
    token_count = 0

    for score, msg in scored_messages:
        print(score)
        if score < threshold:
            selected.append(msg)
            break
        tokens = len(msg['message'].split()) + len(msg['response'].split())
        if token_count + tokens > max_tokens:
            break
        selected.append(msg)
        token_count += tokens

    return selected

# Hybrid Summarization
def summarize_history(messages, message):
    if not messages:
        return ""

    # Extractive Summary
    extractive_summary = ""
    for msg in messages:
        extractive_summary += f"User: {msg['message']}\nAssistant: {msg['response']}\n"

    # Abstractive Summary Prompt
    summary_prompt = f'''You are a summarizer. Summarize the given conversation to retain key context for the message below.
    ## Conversation:
    {extractive_summary}

    ## Current User Message:
    {message}

    ## Instructions:
    Generate a concise summary relevant to the new message while keeping important past details.
    '''

    # Query LLM
    abstractive_summary = query_llama_3_2_1b(summary_prompt)

    return abstractive_summary.strip()

#Build Context with Precomputed Cache
def build_summary_context(history, message, recent_count=3, token_limit=500):
    # Check cache first
    for cache_entry in context_cache:
        if cache_entry["message"] == message:
            return cache_entry["context"]

    if len(history) <= recent_count:
        prompt_context = "## Conversation history:\n"
        for entry in history:
            prompt_context += f"User: {entry['message']}\nAssistant: {entry['response']}\n"
        return prompt_context

    older_messages = history[:-recent_count]
    recent_messages = history[-recent_count:]

    # Summarize older messages
    summary = summarize_history(older_messages, message)

    # Build context
    prompt_context = "## Conversation summary:\n" + summary + "\n\n## Recent conversation:\n"
    for entry in recent_messages:
        if len(entry['response']) > 10000:
            prompt_context += f"User: {entry['message']}\nAssistant: Generated Image\n"
        else:
            prompt_context += f"User: {entry['message']}\nAssistant: {entry['response']}\n"

    # Store in cache
    context_cache.append({"message": message, "context": prompt_context})

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
            hist_embedding = json.loads(msg["embeddings"])["data"]
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
