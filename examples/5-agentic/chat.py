# agentic_workflow.py
"""
Agentic Workflow Demo - Bike Ride Planning Assistant

This example demonstrates how to build an agentic workflow using LangChain's
tool-calling capabilities. The agent can use multiple tools to gather information
and make recommendations.

IMPORTANT: Function/tool calling requires a model that supports it.
Compatible options:
  - OpenAI API: gpt-4, gpt-4o, gpt-3.5-turbo (native support)
  - Ollama: llama3.1, mistral, qwen2.5 (with tool_choice support)
  - LM Studio: Must use a model with function calling support
  - llama-cpp-python: Requires --chat-template chatml-function-calling

If the model doesn't support tool calling, it will describe what it WOULD do
instead of actually calling the tools.
"""

import json
import pandas as pd
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


# ========== TOOLS ==========
@tool
def get_weather(day: str = "today") -> str:
    """Get weather conditions. Use 'today' or 'week'."""
    try:
        # Option 2: Mock weather data
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
    # Mock trail API response
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
        # Read local CSV file (create mock if doesn't exist)
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


# ========== AGENT SETUP ==========
def create_bike_agent():
    """Create and configure the bike planning agent."""

    # Initialize LLM
    # Option 1: Local LLM (requires function calling support)
    llm = ChatOpenAI(
        model="qwen3-coder",  # Model name from your local server
        base_url="http://localhost:8080/v1",  # Local LLM endpoint without tool support
        # base_url="http://localhost:11434/v1",  # Local LLM endpoint
        api_key="not-needed",
        temperature=0,
    )

    # Define tools
    tools = [get_weather, get_trail_status, get_recent_workouts]

    # System prompt for the agent
    system_prompt = """You are a bike ride planning assistant. Use available tools to answer:
    1. Check weather for ideal riding conditions
    2. Check trail status and conditions
    3. Consider recent workout history
    4. Provide specific recommendations with reasoning

    If the weather is at or above 75f it might be too hot, so de-prioritize it.
    """

    # Create agent using the new langchain 1.x API
    return create_agent(llm, tools, system_prompt=system_prompt, debug=False)


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("Bike Ride Planning Assistant")
    print("=" * 40)

    # Create agent
    agent = create_bike_agent()

    # Run query
    query = "When is the best time to ride my bike this week? "
    print(f"\nQuery: {query}\n")
    print("=" * 40)

    result = agent.invoke({"messages": [("user", query)]})

    # Show all messages to see the tool calling flow
    print("\n--- Agent Message Flow ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(
                f"[{i}] {msg_type}: Calling tools: {[tc['name'] for tc in msg.tool_calls]}"
            )
        elif hasattr(msg, "name") and msg.name:  # Tool response
            print(f"[{i}] ToolMessage ({msg.name}): {msg.content[:80]}...")
        else:
            content_preview = str(msg.content)[:100] if msg.content else "(empty)"
            print(f"[{i}] {msg_type}: {content_preview}...")

    print("\n" + "=" * 40)
    print("RECOMMENDATION:")
    print(result["messages"][-1].content)
