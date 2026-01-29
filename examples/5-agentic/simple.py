"""Simple agentic workflow demo for bike ride planning.

This minimal example shows how to create an agent with a single tool
using LangChain's agent framework.
"""

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(day: str = "today") -> str:
    """Get weather conditions. Use 'today' or 'week'.

    Args:
        day: Time period to check weather for.

    Returns:
        Weather information as a formatted string.
    """
    mock_weather: dict[str, Any] = {
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


# ========== AGENT SETUP ==========
def create_bike_agent() -> Any:
    """Create and configure the bike planning agent."""

    # Initialize LLM
    # Option 1: Local LLM (requires function calling support)
    llm = ChatOpenAI(
        model="qwen3-coder",  # Model name from your local server
        base_url="http://localhost:11434/v1",  # Local LLM endpoint
        api_key="not-needed",
        temperature=0,
    )

    # Define tools
    tools = [get_weather]

    # System prompt for the agent
    system_prompt = """You are a bike ride planning assistant. Use available tools to answer:
    1. Check weather for ideal riding conditions
    2. Provide specific recommendations with reasoning"""

    return create_agent(llm, tools, system_prompt=system_prompt, debug=True)


if __name__ == "__main__":
    print("Bike Ride Planning Assistant")
    print("=" * 40)

    # Create agent
    agent = create_bike_agent()

    # Run query
    query = "When is the best time to ride my bike this week?"
    print(f"\nQuery: {query}\n")
    print("=" * 40)

    result = agent.invoke({"messages": [("user", query)]})

    # Show all messages to see the tool calling flow
    print("\n--- Agent Message Flow ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{i}] {msg_type}: Calling tools: {[tc['name'] for tc in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:  # Tool response
            print(f"[{i}] ToolMessage ({msg.name}): {msg.content[:80]}...")
        else:
            content_preview = str(msg.content)[:100] if msg.content else "(empty)"
            print(f"[{i}] {msg_type}: {content_preview}...")

    print("\n" + "=" * 40)
    print("RECOMMENDATION:")
    print(result["messages"][-1].content)
