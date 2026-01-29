from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from os.path import dirname
from openai import OpenAI
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


model_name = os.getenv("MODEL_NAME", "qwen3-coder")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
)


app = FastAPI()
current_directory = dirname(os.path.realpath(__file__))
root = dirname(dirname(current_directory))
csv_file = os.path.join(root, 'top_rated_wines.csv')
df = pd.read_csv(csv_file)
df = df[df['variety'].notna()]
data = df.sample(700).to_dict('records') # Get only 700 records. More records will make it slower to index


# Load the encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')


# Not for production environments! Don't use in memory databases unless it is for testing and demoing
qdrant = QdrantClient(":memory:")

# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)


class Body(BaseModel):
    text: str


def ai_chat(user_message, extra_context=""):
    user_content = user_message
    if extra_context:
        user_content = f"{user_message}\n\nHere is some relevant wine information:\n{extra_context}"
    message_text = [
       {"role":"system","content":"You are a wine specialist. Your top priority is to help guide users find the best wine. You always come with good suggestions."},
       {"role": "user", "content": user_content}
    ]
    completion = client.chat.completions.create(
      model=model_name,
      messages=message_text,
      temperature=0.5,
      max_tokens=400,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
    )
    return completion


@app.post('/generate')
def generate(body: Body):
    # Search time for awesome wines!
    hits = qdrant.query_points(
        collection_name="top_wines",
        query=encoder.encode(body.text).tolist(),
        limit=4
    )
    # Debug the output from the Vector Database
    print(hits)
    # define a variable to hold the search results
    search_results = [hit.payload for hit in hits.points]
    completion = ai_chat(body.text, str(search_results))
    print(completion.choices)
    return {"text": completion.choices[-1].message.content}
