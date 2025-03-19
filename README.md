# Chatbot
Chat bot with multiple models, multiple-modes, multi-modal, context management, RAG, Caching. 
Architecture Link: https://www.figma.com/design/z3aN66C4rR3CpOSgkXsBAK/chatbot-arch?node-id=0-1&p=f&t=OnEVYUoE6kzxqm2J-0

## Key Features:
1. Includes multiple model support.
2. Includes task specific models like models specialized for translation, coding, reasoning etc.
3. Includes multiple modes of generation.
4. Includes support for Text, Images, Voice.
5. Includes context management using clustering and semantic based searching, key word based searching and summary of older messages with N recent messages.
6. Includes mode specific processing.
7. Includes conversation management, users management and storage using Supabase.
8. Includes Caching of responses using Redis and similarity based searching for reducing inference cost and token cost.

## Architecture of Chatbot: 
<img width="1293" alt="Screenshot 2025-03-19 at 06 47 22" src="https://github.com/user-attachments/assets/25c91c4b-9464-48ed-a56b-86c41e0a7c80" />

## Key Components of Chatbot: 
1. ### Front-end:
   - <u>__Login Page__</u>: The login page component is built using HTML, Bootstrap, and vanilla JavaScript to create a interface for user authentication. It features separate forms for login and registration, each with input fields for username and password, along with client-side validation. Upon submission, the forms send requests to a backend API, handling authentication, user registration, and redirection based on the response.
   - <u>__Chat Page__</u>: The chat page component, built using HTML, Bootstrap, and JavaScript, this features a dynamic chat UI with a sidebar for managing previous conversations and a main area for displaying interactive messages. It supports various chat modes—including General, Translation, Text-to-Image, and Voice Transcription, Coding, Reasoning etc, with corresponding UI adjustments such as model selection, file uploads, and voice recording.This handles interactions with a backend server for sending and receiving chat data.

2. ### Back-end:
   - User Management & Conversation Storage: The Flask based backend provides endpoints for user registration and login, the app use Supabase to store user details, conversations, and chat messages.
   - Chat Processing & Multi-Modal Support: The /chat endpoint handles different input types (text, voice, images) and supports various modes (translation, voice transcript, text-to-image, coding, etc.) by using various functions to process context, generate prompts, and select appropriate LLM models.
   - Caching & Context Management: Redis is integrated for caching responses based on embedding similarity, enabling efficient retrieval of previous interactions, while context is dynamically processed to enhance conversational quality.
  

## ML System:
1. ### Caching:
   - __Redis Integration:__ Stores chat responses along with their computed embeddings, enabling fast retrieval when a similar request is received.
   - __Adaptive Matching:__ Uses an adaptive similarity threshold to compare new embeddings against cached ones for efficient cache hits. This helps in reducing inference cost. 
   
2. ### Context Processing:
   - __History Retrieval:__ Extracts and clusters conversation history using cosine similarity, recency weighting (based on time), and keyword matching.
   - __Summarization:__ Uses functions like build_summary_context and summarize_history to create concise summaries from older messages.
   - __Tailored Prompts:__ Combines selected historical context with the new message to generate a well-informed prompt for the LLM.
   - There are 3 Modes of Context Processing:
       - Method 1: Summary of Older messages and N recent messages are placed in the context of the new query.
       - Method 2: Search of relevant previous messages using semantic search, keyword based searching and time based decay.
       - Method 3: Search of relevant previous message and then summarize them. 
     
3. ### Model Selection:
   - __Dynamic Mapping:__ Selects an appropriate language model based on the user’s chosen mode and explicit model selection, with fallback defaults via no_model_selection.
   - __Specialized Functions:__ Invokes specific inference functions (e.g., for translation, coding, text-to-image) to match the task requirements.
   - __Mode-Driven:__ Ensures that model selection aligns with the processing mode, whether it be general chat, specialized research, or other domains.

4. ### Mode procoessing:
   - __Prompt Customization:__ Formats the input message according to the selected mode (e.g., Translation, Coding, Summarizing, etc.) using dedicated prompt templates.
   - __Context Augmentation:__ Adjusts the context processing logic based on mode, handling additional data types like images or voice when necessary.
   - __Instruction Guidance:__ Provides clear instructions to the LLM by creating prompts that specify the expected output style and content.

5. ### Prompts: redefined Templates:
   - Uses a created prompt templates (such as prompt_translation, prompt_coding, etc.) to guide the LLM’s responses.
   - Enhanced Instructions: Constructs detailed prompts like in the Best Quality Mode function—that integrate retrieved knowledge, reasoning, and conversation history.
   - Task-Specific Detailing: Ensures each prompt is fine-tuned to the conversation’s context and the specific task at hand.

6. ### RAG Support: Knowledge Base Integration:
   - Leverages Chroma DB to retrieve external information relevant to the current query.
   - Combined Context: Merges the retrieved knowledge, conversation history, and dynamic reasoning into a single prompt for richer responses.
   - Enhanced Response Quality: Uses external data to augment the LLM’s generation capabilities, producing more informed and contextually accurate answers.

## Deployment Instructions: 



### References: 
1. Caching: https://arxiv.org/abs/2406.00025
2. 


