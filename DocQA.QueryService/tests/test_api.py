from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --- helpers ---

def _mock_openai(*, answer: str | None = None) -> MagicMock:
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=answer or "The Kernel is the core component [1]."))
    ]
    return mock


def _mock_search(chunks: list[dict] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = chunks or [
        {
            "id": "chunk-0",
            "content": "The Kernel is the central component of Semantic Kernel.",
            "sourceFile": "concepts/kernel.md",
            "@search.score": 0.92,
        }
    ]
    return mock


# --- /health ---

def test_health_returns_200_with_expected_keys():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    for key in ("search_index", "chat_deployment", "embedding_deployment", "retrieval_top_k"):
        assert key in data


# --- /query ---

@patch("app.services.retrieval.get_search_client", return_value=_mock_search())
@patch("app.services.retrieval.get_openai_client", return_value=_mock_openai())
def test_query_returns_chunks(_openai, _search):
    response = client.post("/query", json={"question": "What is the Kernel?"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the Kernel?"
    assert data["count"] == 1
    assert data["results"][0]["id"] == "chunk-0"
    assert data["results"][0]["score"] == 0.92


@patch("app.services.retrieval.get_search_client", return_value=_mock_search())
@patch("app.services.retrieval.get_openai_client", return_value=_mock_openai())
def test_query_empty_question_rejected(_openai, _search):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


@patch("app.services.retrieval.get_search_client", return_value=_mock_search())
@patch("app.services.retrieval.get_openai_client", return_value=_mock_openai())
def test_query_top_k_out_of_range_rejected(_openai, _search):
    response = client.post("/query", json={"question": "What is the Kernel?", "top_k": 99})
    assert response.status_code == 422


# --- /ask ---

@patch("app.services.generation.get_openai_client", return_value=_mock_openai(answer="The Kernel is the core component [1]."))
@patch("app.services.retrieval.get_search_client", return_value=_mock_search())
@patch("app.services.retrieval.get_openai_client", return_value=_mock_openai())
def test_ask_returns_answer_and_sources(_retrieval_openai, _search, _generation_openai):
    response = client.post("/ask", json={"question": "What is the Kernel?"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the Kernel?"
    assert "Kernel" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "chunk-0"


@patch("app.services.generation.get_openai_client", return_value=_mock_openai())
@patch("app.services.retrieval.get_search_client", return_value=_mock_search())
@patch("app.services.retrieval.get_openai_client", return_value=_mock_openai())
def test_ask_empty_question_rejected(_retrieval_openai, _search, _generation_openai):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
