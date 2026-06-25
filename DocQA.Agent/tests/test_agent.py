from unittest.mock import MagicMock, patch

import httpx
import pytest


# --- query_sk_docs tool ---


def test_tool_name():
    from app.tools.query_docs import query_sk_docs

    assert query_sk_docs.name == "query_sk_docs"


def test_tool_description_mentions_semantic_kernel():
    """The docstring is sent to the LLM — verify it references SK so the model
    knows when to use the tool."""
    from app.tools.query_docs import query_sk_docs

    assert "Semantic Kernel" in query_sk_docs.description


def test_tool_posts_to_ask_endpoint():
    from app.tools.query_docs import query_sk_docs

    mock_response = MagicMock()
    mock_response.json.return_value = {"answer": "Use kernel.Plugins.AddFromType<T>()."}
    mock_response.raise_for_status.return_value = None

    with patch("app.tools.query_docs.httpx.post", return_value=mock_response) as mock_post:
        result = query_sk_docs.invoke({"question": "How do I add a plugin?"})

    url = mock_post.call_args[0][0]
    assert url.endswith("/ask")
    assert mock_post.call_args[1]["json"]["question"] == "How do I add a plugin?"
    assert result == "Use kernel.Plugins.AddFromType<T>()."


def test_tool_propagates_http_error():
    from app.tools.query_docs import query_sk_docs

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "502", request=MagicMock(), response=MagicMock()
    )

    with patch("app.tools.query_docs.httpx.post", return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            query_sk_docs.invoke({"question": "test"})


# --- build_agent ---


@patch("app.agent.graph.AzureChatOpenAI")
def test_build_agent_returns_runnable(mock_llm_class):
    from app.agent.graph import build_agent

    agent = build_agent()

    assert agent is not None
    mock_llm_class.assert_called_once()


@patch("app.agent.graph.create_react_agent")
@patch("app.agent.graph.AzureChatOpenAI")
def test_build_agent_registers_query_tool(mock_llm_class, mock_create_agent):
    """Verify query_sk_docs is wired into the agent — if this breaks the agent
    has no way to look up docs."""
    from app.agent.graph import build_agent

    build_agent()

    tools = mock_create_agent.call_args.kwargs["tools"]
    tool_names = [t.name for t in tools]
    assert "query_sk_docs" in tool_names
