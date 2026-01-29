import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
client.api_base = "http://localhost:11434"
model_name = os.getenv("MODEL_NAME")


def ai_chat(user_message):
    message_text = [
       {"role":"system","content":"You are an HTTP API and you respond only in JSON. Avoid niceties or formalities, you only respond in JSON format to user questions. Avoid formatting like backticks ` or trying to format like a markdown document. ONLY respond in pure JSON"},
       {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(model=model_name,
    messages=message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
    return completion


print("Welcome! how can I help you today?")

while True:
    user_message = input(">> ")
    completion = ai_chat(user_message)
    # Completion will return a response that we need to use to get the acctual string
    print(completion.choices[0].message.content)
