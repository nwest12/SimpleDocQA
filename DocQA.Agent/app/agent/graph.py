from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.core.config import settings
from app.tools.query_docs import query_sk_docs

_SYSTEM_PROMPT = """\
You are a Semantic Kernel migration assistant.

When given a code snippet that uses Semantic Kernel APIs, you:
1. Identify which SK APIs, classes, or patterns are being used.
2. Query the current SK documentation using the query_sk_docs tool to find up-to-date equivalents.
3. Produce clear migration advice: what changed, why, and a corrected version of the snippet.

Ask targeted questions, e.g. "How do I register a plugin with the Kernel?" or
"What is the current API for KernelFunction?" Query multiple times if the snippet
touches multiple areas.

If the code already matches current SK patterns, say so explicitly."""


def build_agent():
    llm = AzureChatOpenAI(**settings.azure_chat_kwargs)
    return create_react_agent(llm, tools=[query_sk_docs], prompt=_SYSTEM_PROMPT)
