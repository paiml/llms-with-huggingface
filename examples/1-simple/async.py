import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
)
model_name = os.getenv("MODEL_NAME", "qwen2.5-coder:7b-instruct")


def ai_chat_stream(user_message):
    message_text = [
        {"role":"system","content":"You are a friendly AI assistant."},
        {"role": "user", "content": user_message}
    ]

    stream = client.chat.completions.create(
        model=model_name,
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        stream=True,
    )
    
    return stream


print("Welcome! How can I help you today?")

while True:
    user_message = input(">> ")
    
    if user_message.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
    
    print("Assistant: ", end="", flush=True)
    
    try:
        stream = ai_chat_stream(user_message)
        
        for chunk in stream:
            # Check if there's content in this chunk
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
    
    except Exception as e:
        print(f"\nError: {e}")
    
    print()  # New line
