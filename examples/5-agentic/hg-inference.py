# agentic_workflow_hf.py
"""
Agentic Workflow Demo - Bike Ride Planning Assistant
Using Hugging Face Inference API instead of local models

This example demonstrates how to build an agentic workflow using
Hugging Face Inference API with tool-calling capabilities.

Requirements:
- Hugging Face account (free: https://huggingface.co/join)
- HF Token from: https://huggingface.co/settings/tokens
- Install: pip install langchain-huggingface
"""

import json
import os

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set your Hugging Face token (free tier works!)
# Option 1: Set as environment variable
# export HF_TOKEN="your_token_here"
# Option 2: Set directly (not recommended for production)
# os.environ["HF_TOKEN"] = "your_token_here"


# ========== TOOLS ==========
@tool
def get_weather(day: str = "today") -> str:
    """Get weather conditions. Use 'today' or 'week'."""
    try:
        mock_weather = {
            "today": "Sunny, 72°F, Wind: 5mph",
            "week": {
                "Mon": "Sunny, 72°F",
                "Tue": "Partly Cloudy, 68°F",
                "Wed": "Rainy, 62°F",
                "Thu": "Clear, 70°F",
                "Fri": "Cloudy, 65°F",
                "Sat": "Sunny, 75°F",
                "Sun": "Sunny, 78°F",
            },
        }

        if day == "today":
            return f"Today's weather: {mock_weather['today']}"
        return f"Week forecast: {json.dumps(mock_weather['week'])}"
    except:
        return "Weather data unavailable"


@tool
def get_trail_status(trail: str = "both") -> str:
    """Get mountain bike trail status. Options: 'rope_mill', 'blankets', or 'both'."""
    trails = {
        "rope_mill": {
            "status": "Open",
            "conditions": "Good",
            "last_update": "2024-01-20",
            "notes": "Trail is dry and fast",
        },
        "blankets_creek": {
            "status": "Closed",
            "conditions": "Wet",
            "last_update": "2024-01-20",
            "notes": "Closed due to rain",
        },
    }

    if trail == "rope_mill":
        return f"Rope Mill: {json.dumps(trails['rope_mill'])}"
    elif trail == "blankets":
        return f"Blankets Creek: {json.dumps(trails['blankets_creek'])}"
    else:
        return f"All trails: {json.dumps(trails)}"


@tool
def get_recent_workouts(days: int = 7) -> str:
    """Get recent workouts from fitness tracker CSV."""
    try:
        df = pd.DataFrame(
            {
                "date": ["2024-01-15", "2024-01-16", "2024-01-18", "2024-01-20"],
                "workout": [
                    "Bike - 30mi",
                    "Run - 5mi",
                    "Bike - 15mi",
                    "Strength - 60min",
                ],
                "intensity": ["High", "Medium", "Low", "Medium"],
            }
        )

        recent = df.tail(days).to_dict("records")
        return f"Recent workouts: {json.dumps(recent)}"
    except:
        return "No workout data available"


# ========== AGENT SETUP WITH HUGGING FACE ==========
def create_bike_agent_hf():
    """Create and configure the bike planning agent using Hugging Face Inference API."""

    # Check if token is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: HF_TOKEN not found in environment variables.")
        print("   Get a free token from: https://huggingface.co/settings/tokens")
        print("   Then run: export HF_TOKEN='your_token_here'")
        raise ValueError("HF_TOKEN required for tool calling support")

    # Models that support native tool calling via HF Inference API
    # Qwen2.5-72B-Instruct has good tool calling support
    model_id = "Qwen/Qwen2.5-72B-Instruct"

    # Create endpoint
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        temperature=0.1,
    )

    # Wrap in ChatHuggingFace for LangChain compatibility
    chat_llm = ChatHuggingFace(llm=llm)

    # Define tools
    tools = [get_weather, get_trail_status, get_recent_workouts]

    # Bind tools to the chat model for tool calling support
    chat_llm_with_tools = chat_llm.bind_tools(tools)

    # System prompt
    system_prompt = """You are a bike ride planning assistant. You MUST use the available tools to gather real data before answering:
1. Use get_weather to check current weather conditions
2. Use get_trail_status to check if trails are open
3. Use get_recent_workouts to see recent activity

ALWAYS call the relevant tools first, then provide your recommendation based on the actual data."""

    # Create agent with tool-bound model
    return create_agent(chat_llm_with_tools, tools, system_prompt=system_prompt, debug=True)


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("Bike Ride Planning Assistant - Hugging Face Edition")
    print("=" * 50)

    # Create agent (requires HF token for tool calling support)
    try:
        print("Creating agent with tool calling support...")
        agent = create_bike_agent_hf()
        print("Agent ready with HuggingFace model")
    except Exception as e:
        print(f"Failed to create agent: {e}")
        exit(1)

    # Test query
    queries = [
        "Should I ride today? Check the weather and trail status.",
    ]

    for i, query in enumerate(queries):
        print(f"\n{'=' * 60}")
        print(f"Query {i + 1}: {query}")
        print(f"{'=' * 60}")

        try:
            result = agent.invoke({"messages": [("user", query)]})

            # Show message flow
            print("\n--- Agent Message Flow ---")
            for msg in result["messages"]:
                msg_type = type(msg).__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"  {msg_type}: Calling tools: {[tc['name'] for tc in msg.tool_calls]}")
                elif hasattr(msg, "name") and msg.name:
                    print(f"  ToolMessage ({msg.name}): {msg.content[:100]}...")
                elif hasattr(msg, "content") and msg.content:
                    content = str(msg.content)
                    if len(content) > 100:
                        print(f"  {msg_type}: {content[:100]}...")
                    else:
                        print(f"  {msg_type}: {content}")

            # Show final recommendation
            print(f"\n{'=' * 60}")
            print("RECOMMENDATION:")
            if result["messages"]:
                print(result["messages"][-1].content)

        except Exception as e:
            print(f"❌ Error: {e}")
            print(" Tip: Some free models don't support tool calling well.")
            print("    Try getting a free HF token for better models.")
