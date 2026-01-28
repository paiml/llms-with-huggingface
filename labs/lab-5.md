# Lab 5: Building Agentic Applications

In this lab, you will learn how to build agentic applications that extend LLM capabilities through function calling and tool use. You'll create AI agents that can interact with external systems and make decisions based on real-world data.

## Learning Objectives

By the end of this lab, you will be able to:

- Understand the concept of AI agents and tool calling
- Implement function calling with LLMs
- Build agents using the LangChain framework
- Create custom tools for your agents
- Handle the agent execution loop

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install langchain langchain-core langchain-openai openai
```

## Key Concepts

- **Agent**: An LLM that can decide which actions to take and execute them
- **Tool**: A function that an agent can call to interact with external systems
- **Function Calling**: LLM capability to generate structured function calls
- **Agent Loop**: The cycle of thinking, acting, and observing results
- **LangChain**: A framework for building LLM-powered applications

## Lab Exercises

### Exercise 1: Understanding Agents

Navigate to the [examples/5-agentic](../examples/5-agentic/) directory.

Agents follow a loop:
1. **Think**: Analyze the user's request
2. **Act**: Decide which tool to call (if any)
3. **Observe**: Process the tool's result
4. **Respond**: Generate a final answer

### Exercise 2: Creating a Simple Tool

Study [simple.py](../examples/5-agentic/simple.py):

```python
from langchain_core.tools import tool

@tool
def get_weather(day: str = "today") -> str:
    """Get weather conditions. Use 'today' or 'week'."""
    mock_weather = {
        "today": "Sunny, 72°F, Wind: 5mph",
        "week": {
            "Mon": "Sunny, 72°F",
            "Tue": "Partly Cloudy, 68°F",
            "Wed": "Rainy, 62°F",
            "Thu": "Clear, 70°F",
            "Fri": "Cloudy, 65°F",
            "Sat": "Sunny, 75°F",
            "Sun": "Sunny, 78°F"
        }
    }

    if day == "today":
        return f"Today's weather: {mock_weather['today']}"
    return f"Week forecast: {json.dumps(mock_weather['week'])}"
```

Key elements:
- `@tool` decorator marks the function as an agent tool
- Docstring is used by the LLM to understand when to use the tool
- Type hints help the LLM generate correct parameters

### Exercise 3: Building an Agent

Create an agent that uses your tools:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def create_bike_agent():
    """Create and configure a planning agent."""

    # Initialize LLM (local or remote)
    llm = ChatOpenAI(
        model="qwen3-coder",
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
        temperature=0
    )

    # Define available tools
    tools = [get_weather]

    # System prompt guides agent behavior
    system_prompt = """You are a bike ride planning assistant.
    Use available tools to answer:
    1. Check weather for ideal riding conditions
    2. Provide specific recommendations with reasoning"""

    return create_agent(llm, tools, system_prompt=system_prompt)
```

### Exercise 4: Running the Agent

Execute the agent with a query:

```python
if __name__ == "__main__":
    # Create agent
    agent = create_bike_agent()

    # Run query
    query = "When is the best time to ride my bike this week?"

    result = agent.invoke({"messages": [("user", query)]})

    # The agent will:
    # 1. Recognize it needs weather data
    # 2. Call the get_weather tool
    # 3. Analyze the results
    # 4. Provide a recommendation

    print(result["messages"][-1].content)
```

### Exercise 5: Multiple Tools

Study [chat.py](../examples/5-agentic/chat.py) for a more complex agent:

```python
import json
import pandas as pd
from langchain_core.tools import tool

@tool
def get_weather(day: str = "today") -> str:
    """Get weather conditions for planning outdoor activities."""
    # Implementation...

@tool
def get_trail_status() -> str:
    """Get current trail status for mountain biking."""
    trail_data = {
        "rope_mill": {"status": "open", "conditions": "dry"},
        "blankets_creek": {"status": "closed", "conditions": "muddy"}
    }
    return json.dumps(trail_data)

@tool
def get_recent_workouts() -> str:
    """Get user's recent workout history."""
    workouts = [
        {"date": "2024-01-15", "type": "cycling", "duration": 90},
        {"date": "2024-01-13", "type": "running", "duration": 45},
    ]
    return json.dumps(workouts)

# Agent now has multiple tools to make informed decisions
tools = [get_weather, get_trail_status, get_recent_workouts]
```

### Exercise 6: Observing Agent Behavior

Debug and understand the agent's decision process:

```python
result = agent.invoke({"messages": [("user", query)]})

# Examine the message flow
print("\n--- Agent Message Flow ---")
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__

    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        # Agent decided to call tools
        print(f"[{i}] {msg_type}: Calling tools: {[tc['name'] for tc in msg.tool_calls]}")
    elif hasattr(msg, 'name') and msg.name:
        # Tool returned a response
        print(f"[{i}] ToolMessage ({msg.name}): {msg.content[:80]}...")
    else:
        # Regular message
        content = str(msg.content)[:100] if msg.content else "(empty)"
        print(f"[{i}] {msg_type}: {content}...")

print("\nFINAL RECOMMENDATION:")
print(result["messages"][-1].content)
```

### Exercise 7: Using Hugging Face Inference API

Study [hg-inference.py](../examples/5-agentic/hg-inference.py) for remote model usage:

```python
from huggingface_hub import InferenceClient

# Use Hugging Face's hosted models
client = InferenceClient(token="your_hf_token")

# For agents, you need models that support function calling
response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[
        {"role": "user", "content": "What's the weather like?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]
)
```

### Exercise 8: Building a Complete Agentic System

Combine everything into a production-ready agent:

```python
class BikeAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen3-coder",
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
        )
        self.tools = [get_weather, get_trail_status, get_recent_workouts]
        self.agent = create_agent(
            self.llm,
            self.tools,
            system_prompt="You are a bike ride planning assistant."
        )

    def ask(self, question: str) -> str:
        """Ask the assistant a question."""
        result = self.agent.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    def chat(self):
        """Interactive chat loop."""
        print("Bike Planning Assistant")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            response = self.ask(user_input)
            print(f"Assistant: {response}\n")

# Run the assistant
if __name__ == "__main__":
    assistant = BikeAssistant()
    assistant.chat()
```

## Challenge

1. Create a new tool that fetches real data from an API (weather, news, etc.)
2. Build an agent with 5+ tools for a specific domain (travel, cooking, finance)
3. Implement error handling for when tools fail
4. Add conversation memory so the agent remembers previous interactions
5. Create a web API that exposes your agent using FastAPI

## Summary

In this lab, you learned how to:
- Understand the agent loop: think, act, observe, respond
- Create tools using the `@tool` decorator
- Build agents with LangChain
- Combine multiple tools for complex decision making
- Debug and observe agent behavior
- Use both local and remote models for agents

## Course Complete!

Congratulations on completing the Large Language Models with Hugging Face course! You now have the skills to:

1. Build interactive chat applications with LLMs
2. Implement RAG for knowledge-augmented responses
3. Create web APIs for LLM functionality
4. Work efficiently with small language models
5. Build agentic applications with tool use

## Next Steps

- Explore [Hugging Face Hub](https://huggingface.co/models) for more models
- Learn about [LangGraph](https://langchain-ai.github.io/langgraph/) for complex agent workflows
- Try [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) for production deployments
- Experiment with fine-tuning models for your specific use case
