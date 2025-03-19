# Chatbot
Chat bot with multiple models, multi-modes, multi-modal, context management, RAG, Caching. 
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
