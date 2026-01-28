# Lab 4: Working with Small Language Models

In this lab, you will learn how to work with small language models that can run efficiently on local hardware. You'll explore model selection, optimization techniques, and when to choose smaller models over larger ones.

## Learning Objectives

By the end of this lab, you will be able to:

- Understand the tradeoffs between model size and capability
- Run small language models locally
- Optimize model inference for better performance
- Choose the right model size for your use case

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install openai fastapi uvicorn
```

You'll also need Ollama or another local LLM server installed.

## Key Concepts

- **Small Language Models**: Models with fewer parameters (1B-7B) that run on consumer hardware
- **Quantization**: Reducing model precision to decrease memory usage
- **Inference Speed**: How quickly a model generates responses
- **Context Window**: Maximum input length a model can process
- **Parameter Count**: Number of learnable weights in the model

## Lab Exercises

### Exercise 1: Understanding Model Sizes

Navigate to the [examples/4-small-lm](../examples/4-small-lm/) directory.

Common model size categories:

| Size | Parameters | RAM Required | Use Case |
|------|------------|--------------|----------|
| Tiny | < 1B | 1-2 GB | Simple tasks, edge devices |
| Small | 1-3B | 2-4 GB | Basic chat, classification |
| Medium | 7B | 8-16 GB | General purpose |
| Large | 13B+ | 16-32 GB | Complex reasoning |

### Exercise 2: Running a Small Model

Study [chat.py](../examples/4-small-lm/chat.py):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai

# Configure for local model
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")

app = FastAPI()

class Body(BaseModel):
    text: str


def ai_chat(user_message):
    """Chat with a small local model."""
    message_text = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
    )
    return completion
```

### Exercise 3: Comparing Model Sizes with Ollama

Install and run different model sizes:

```bash
# Pull models of different sizes
ollama pull qwen2.5:0.5b    # 500M parameters
ollama pull qwen2.5:1.5b    # 1.5B parameters
ollama pull qwen2.5:7b      # 7B parameters

# Run the smallest model
ollama run qwen2.5:0.5b
```

### Exercise 4: Benchmarking Response Quality

Test the same prompt across different model sizes:

```python
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def benchmark_model(model_name, prompt):
    """Measure response time and quality."""
    start = time.time()

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    elapsed = time.time() - start

    return {
        "model": model_name,
        "time": elapsed,
        "response": response.choices[0].message.content,
        "tokens": response.usage.completion_tokens
    }

# Compare models
prompt = "Explain what a REST API is in simple terms."
models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b"]

for model in models:
    result = benchmark_model(model, prompt)
    print(f"\n{result['model']}:")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Tokens: {result['tokens']}")
    print(f"  Response: {result['response'][:100]}...")
```

### Exercise 5: Optimizing for Speed

Techniques to improve inference speed:

```python
# 1. Reduce max_tokens for faster responses
completion = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=messages,
    max_tokens=100,  # Limit output length
)

# 2. Use lower temperature for faster convergence
completion = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=messages,
    temperature=0.1,  # More deterministic
)

# 3. Keep system prompts short
messages = [
    {"role": "system", "content": "Be brief."},  # Short system prompt
    {"role": "user", "content": user_input}
]
```

### Exercise 6: Use Case Selection Guide

Match model sizes to tasks:

```python
# Task: Simple classification (use smallest model)
def classify_sentiment(text):
    response = client.chat.completions.create(
        model="qwen2.5:0.5b",  # Smallest is sufficient
        messages=[
            {"role": "system", "content": "Reply only: positive, negative, or neutral"},
            {"role": "user", "content": f"Classify: {text}"}
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


# Task: Code generation (use medium model)
def generate_code(description):
    response = client.chat.completions.create(
        model="qwen2.5:7b",  # Larger for complex tasks
        messages=[
            {"role": "system", "content": "Write clean, working code."},
            {"role": "user", "content": description}
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# Task: Quick Q&A (use small model)
def quick_answer(question):
    response = client.chat.completions.create(
        model="qwen2.5:1.5b",  # Balance of speed and quality
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content
```

### Exercise 7: Building a Model Router

Create a system that selects the appropriate model:

```python
def get_model_for_task(task_type):
    """Select model based on task complexity."""
    model_map = {
        "classification": "qwen2.5:0.5b",
        "extraction": "qwen2.5:1.5b",
        "summarization": "qwen2.5:1.5b",
        "chat": "qwen2.5:3b",
        "code": "qwen2.5:7b",
        "reasoning": "qwen2.5:7b",
    }
    return model_map.get(task_type, "qwen2.5:1.5b")


def smart_completion(task_type, prompt):
    """Route to appropriate model based on task."""
    model = get_model_for_task(task_type)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "model_used": model,
        "response": response.choices[0].message.content
    }

# Usage
result = smart_completion("classification", "Is this review positive? 'Great product!'")
print(f"Used {result['model_used']}: {result['response']}")
```

## Challenge

1. Benchmark 3 different model sizes on the same set of 10 prompts
2. Create a FastAPI service that automatically routes to different models
3. Implement a fallback system: try small model first, escalate if needed
4. Measure memory usage for different model sizes
5. Find the smallest model that achieves acceptable quality for your use case

## Summary

In this lab, you learned how to:
- Understand tradeoffs between model size and capability
- Run and compare different small language models
- Benchmark model performance (speed and quality)
- Optimize inference for faster responses
- Select appropriate model sizes for different tasks
- Build intelligent model routing systems

## Next Steps

Continue to [Lab 5: Building Agentic Applications](./lab-5.md) to learn how to extend LLMs with tools and create AI agents.
