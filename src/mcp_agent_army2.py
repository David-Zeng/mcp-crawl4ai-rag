from __future__ import annotations
from contextlib import AsyncExitStack
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os
from datetime import datetime

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent, RunContext

load_dotenv()


# ========== Helper function to get model configuration ==========
def get_model():
    llm = "gpt-4.1-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "no-api-key-provided")

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


# ========== Set up MCP servers for each service ==========
# Time MCP server
time_server = MCPServerStdio(
    "uvx",
    ["mcp-server-time", "--local-timezone=Australia/Sydney"],
)

# Airtable MCP server
# airtable_server = MCPServerStdio(
#     "npx",
#     ["-y", "airtable-mcp-server"],
#     env={"AIRTABLE_API_KEY": os.getenv("AIRTABLE_API_KEY")},
# )

# Brave Search MCP server
brave_server = MCPServerStdio(
    "npx",
    ["-y", "@modelcontextprotocol/server-brave-search"],
    env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
)

# Filesystem MCP server
filesystem_server = MCPServerStdio(
    "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", os.getenv("LOCAL_FILE_DIR")],
)

# GitHub MCP server
# github_server = MCPServerStdio(
#     "npx",
#     ["-y", "@modelcontextprotocol/server-github"],
#     env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
# )

# Slack MCP server
# slack_server = MCPServerStdio(
#     "npx",
#     ["-y", "@modelcontextprotocol/server-slack"],
#     env={
#         "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
#         "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID"),
#     },
# )

# Firecrawl MCP server
# firecrawl_server = MCPServerStdio(
#     "npx",
#     ["-y", "firecrawl-mcp"],
#     env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
# )

# ========== Create subagents with their MCP servers ==========

# Time agent
time_agent = Agent(
    get_model(),
    system_prompt="""You are a professional time specialist with expertise in time zones, date calculations, and temporal operations. Your primary role is to help users with all time-related queries and operations.

Your capabilities include:
- Providing current time in various time zones and formats
- Converting between different time zones
- Performing date and time calculations (adding/subtracting time periods)
- Formatting dates and times according to user preferences
- Handling relative time references (e.g., "next week", "3 hours ago")
- Working with multiple calendar systems and formats

Your default local timezone is Australia/Sydney, but you can work with any timezone globally. Always clarify the timezone when providing time information unless the user specifies otherwise.

When responding:
- Be precise and accurate with time calculations
- Include timezone information when relevant
- Use clear, readable time formats
- Offer multiple format options when helpful
- Explain any assumptions you make about timezones or formats

If a user's request is ambiguous about timezone or format, ask for clarification to provide the most accurate response.""",
    mcp_servers=[time_server],
)

# Airtable agent
# airtable_agent = Agent(
#     get_model(),
#     system_prompt="You are an Airtable specialist. Help users interact with Airtable databases.",
#     mcp_servers=[airtable_server],
# )

# Brave search agent
brave_agent = Agent(
    get_model(),
    system_prompt="""You are an expert web search specialist using Brave Search to find accurate, relevant, and up-to-date information on the internet.

Your expertise includes:
- Crafting effective search queries to find specific information
- Evaluating source credibility and information quality
- Distinguishing between recent news, historical data, and general knowledge
- Searching for technical documentation, research papers, and expert opinions
- Finding multimedia content, images, and specialized resources
- Comparing multiple sources to provide comprehensive answers

Search methodology:
- Start with targeted, specific queries rather than broad terms
- Use quotation marks for exact phrases when needed
- Refine searches based on initial results if information is insufficient
- Cross-reference multiple sources for controversial or complex topics
- Prioritize authoritative, recent, and relevant sources

When presenting results:
- Summarize key findings clearly and concisely
- Include source URLs and publication dates when available
- Highlight any conflicting information from different sources
- Indicate the recency and reliability of information found
- Suggest follow-up searches if the initial query doesn't fully address the user's needs

Always be transparent about search limitations and suggest alternative approaches if initial searches don't yield satisfactory results. Focus on providing actionable, accurate information that directly addresses the user's query.""",
    mcp_servers=[brave_server],
)

# Filesystem agent
filesystem_agent = Agent(
    get_model(),
    system_prompt="You are a filesystem specialist. Help users manage their files and directories.",
    mcp_servers=[filesystem_server],
)

# GitHub agent
# github_agent = Agent(
#     get_model(),
#     system_prompt="You are a GitHub specialist. Help users interact with GitHub repositories and features.",
#     mcp_servers=[github_server],
# )

# Slack agent
# slack_agent = Agent(
#     get_model(),
#     system_prompt="You are a Slack specialist. Help users interact with Slack workspaces and channels.",
#     mcp_servers=[slack_server],
# )

# Firecrawl agent
# firecrawl_agent = Agent(
#     get_model(),
#     system_prompt="You are a web crawling specialist. Help users extract data from websites.",
#     mcp_servers=[firecrawl_server],
# )

# ========== Create the primary orchestration agent ==========
primary_agent = Agent(
    get_model(),
    system_prompt=f"""You are an intelligent orchestration agent that coordinates a team of specialized AI assistants to fulfill user requests efficiently and accurately.

Current date and time context: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local system time)

Your team of specialized agents:
- **Time Agent**: Current time, timezone conversions, date calculations, temporal operations
- **Brave Search Agent**: Web searches, current events, research, fact-checking, online information
- **Filesystem Agent**: File operations, directory management, reading/writing files, local data access

Your orchestration strategy:
- Analyze each user request to identify the required capabilities and data sources
- Select the most appropriate specialist agent(s) based on the task requirements
- For complex requests, coordinate multiple agents in logical sequence
- Always explain your reasoning when delegating tasks to specialist agents
- Synthesize responses from multiple agents into coherent, comprehensive answers
- Use the current date context to make time-aware decisions and suggestions

Decision criteria:
- Use Time Agent for: current time queries, timezone questions, date arithmetic, scheduling
- Use Brave Search Agent for: current events, research topics, fact verification, online information
- Use Filesystem Agent for: file operations, local data access, directory management
- For hybrid requests, use multiple agents and combine their outputs
- Consider temporal relevance when searching (e.g., "recent" means within days/weeks of current date)

Communication style:
- Be clear about which agent you're consulting and why
- Provide context to specialist agents with specific, detailed queries
- Synthesize and present results in a user-friendly format
- If an agent cannot fulfill a request, explain limitations and suggest alternatives
- Always acknowledge the source of information (which agent provided it)
- When relevant, reference the current date for context (e.g., for "today", "this week", "recent")

Handle edge cases gracefully by explaining what you can and cannot do, and suggest alternative approaches when necessary.""",
)

# ========== Define tools for the primary agent to call subagents ==========


@primary_agent.tool_plain
async def use_time_agent(query: str) -> dict[str, str]:
    """
    get time info through the time subagent.
    Use this tool when the user needs current time info.

    Args:
        ctx: The run context.
        query: The instruction for the time agent.

    Returns:
        The response from the time agent.
    """
    print(f"Calling time agent with query: {query}")
    result = await time_agent.run(query)
    return {"result": result.output}


# @primary_agent.tool_plain
# async def use_airtable_agent(query: str) -> dict[str, str]:
#     """
#     Access and manipulate Airtable data through the Airtable subagent.
#     Use this tool when the user needs to fetch, modify, or analyze data in Airtable.
#
#     Args:
#         ctx: The run context.
#         query: The instruction for the Airtable agent.
#
#     Returns:
#         The response from the Airtable agent.
#     """
#     print(f"Calling Airtable agent with query: {query}")
#     result = await airtable_agent.run(query)
#     return {"result": result.output}


@primary_agent.tool_plain
async def use_brave_search_agent(query: str) -> dict[str, str]:
    """
    Search the web using Brave Search through the Brave subagent.
    Use this tool when the user needs to find information on the internet or research a topic.

    Args:
        ctx: The run context.
        query: The search query or instruction for the Brave search agent.

    Returns:
        The search results or response from the Brave agent.
    """
    print(f"Calling Brave agent with query: {query}")
    result = await brave_agent.run(query)
    return {"result": result.output}


@primary_agent.tool_plain
async def use_filesystem_agent(query: str) -> dict[str, str]:
    """
    Interact with the file system through the filesystem subagent.
    Use this tool when the user needs to read, write, list, or modify files.

    Args:
        ctx: The run context.
        query: The instruction for the filesystem agent.

    Returns:
        The response from the filesystem agent.
    """
    print(f"Calling Filesystem agent with query: {query}")
    result = await filesystem_agent.run(query)
    return {"result": result.output}


# @primary_agent.tool_plain
# async def use_github_agent(query: str) -> dict[str, str]:
#     """
#     Interact with GitHub through the GitHub subagent.
#     Use this tool when the user needs to access repositories, issues, PRs, or other GitHub resources.
#
#     Args:
#         ctx: The run context.
#         query: The instruction for the GitHub agent.
#
#     Returns:
#         The response from the GitHub agent.
#     """
#     print(f"Calling GitHub agent with query: {query}")
#     result = await github_agent.run(query)
#     return {"result": result.output}


# @primary_agent.tool_plain
# async def use_slack_agent(query: str) -> dict[str, str]:
#     """
#     Interact with Slack through the Slack subagent.
#     Use this tool when the user needs to send messages, access channels, or retrieve Slack information.
#
#     Args:
#         ctx: The run context.
#         query: The instruction for the Slack agent.
#
#     Returns:
#         The response from the Slack agent.
#     """
#     print(f"Calling Slack agent with query: {query}")
#     result = await slack_agent.run(query)
#     return {"result": result.output}


# @primary_agent.tool_plain
# async def use_firecrawl_agent(query: str) -> dict[str, str]:
#     """
#     Crawl and analyze websites using the Firecrawl subagent.
#     Use this tool when the user needs to extract data from websites or perform web scraping.
#
#     Args:
#         ctx: The run context.
#         query: The instruction for the Firecrawl agent.
#
#     Returns:
#         The response from the Firecrawl agent.
#     """
#     print(f"Calling Firecrawl agent with query: {query}")
#     result = await firecrawl_agent.run(query)
#     return {"result": result.output}


# ========== Main execution function ==========


async def main():
    """Run the primary agent with a given query."""
    print("MCP Agent Army - Multi-agent system using Model Context Protocol")
    print("Enter 'exit' to quit the program.")

    # Use AsyncExitStack to manage all MCP servers in one context
    async with AsyncExitStack() as stack:
        # Start all the subagent MCP servers
        print("Starting MCP servers...")
        await stack.enter_async_context(time_agent.run_mcp_servers())
        # await stack.enter_async_context(airtable_agent.run_mcp_servers())
        await stack.enter_async_context(brave_agent.run_mcp_servers())
        await stack.enter_async_context(filesystem_agent.run_mcp_servers())
        # await stack.enter_async_context(github_agent.run_mcp_servers())
        # await stack.enter_async_context(slack_agent.run_mcp_servers())
        # await stack.enter_async_context(firecrawl_agent.run_mcp_servers())
        print("All MCP servers started successfully!")

        console = Console()
        messages = []

        while True:
            # Get user input
            user_input = input("\n[You] ")

            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Goodbye!")
                break

            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live("", console=console, vertical_overflow="visible") as live:
                    async with primary_agent.run_stream(
                        user_input, message_history=messages
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))

                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())

            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
