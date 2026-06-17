# DocQA — Documentation Q&A over Semantic Kernel docs (RAG)

A retrieval-augmented generation system that answers natural-language questions
about Microsoft Semantic Kernel, grounded in the official documentation, with
citations and an automated evaluation harness.

Built across two language stacks — C#/.NET for ingestion and Python/FastAPI for
the query API — over a single shared Azure AI Search index.

## What it does

Ask a question about Semantic Kernel and get an answer drawn only from the
official docs, with numbered source citations. If the docs don't cover the
question, it says so rather than guessing.

## Architecture

Two components share a single Azure AI Search index.

```
DocQA.Ingestion  (C#/.NET 8)
  ├── Ingests docs: markdown → clean → chunk → embed → Azure AI Search
  ├── Queries: embed question → vector search → grounded completion
  └── Evaluates: LLM-as-judge harness over a curated question set

DocQA.QueryService  (Python/FastAPI)
  └── Query-only API: exposes /health, /query, /ask over the same index

Shared infrastructure
  ├── Azure AI Search  (vector store)
  └── Azure OpenAI  (text-embedding-3-small + gpt-4o-mini)
```

| Layer | Choice |
|---|---|
| Ingestion + eval | C#/.NET 8, Semantic Kernel |
| Query API | Python 3.12, FastAPI, uv |
| LLM | Azure OpenAI `gpt-4o-mini` |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store | Azure AI Search |

## Why two stacks

The RAG pattern is language-agnostic. Building both components demonstrates
that, and reflects a realistic scenario: a team might own ingestion and
infrastructure in C#/.NET while exposing the search capability as a Python
microservice for the broader AI ecosystem (LangChain, agents, notebooks, etc.).

The .NET component leverages existing backend depth. The Python component is
deliberate skill-building in the Python AI engineering space.

## Getting the docs corpus

Both components read from a local `docs/` folder at the repo root. That folder
is gitignored — the content is the public Microsoft Semantic Kernel
documentation, licensed CC-BY-4.0 (code samples MIT), and belongs to Microsoft.

To populate it:

```bash
git clone https://github.com/MicrosoftDocs/semantic-kernel-docs /tmp/sk-docs
cp -r /tmp/sk-docs/semantic-kernel/concepts docs/semantic-kernel/concepts
# Optionally also copy get-started/ and Frameworks/
```

---

## DocQA.Ingestion (C#/.NET 8)

Ingests the docs corpus into Azure AI Search, provides a query CLI, and runs
an automated LLM-as-judge evaluation harness.

### Setup

Secrets are stored via `dotnet user-secrets` — never committed.

```bash
cd DocQA.Ingestion

dotnet user-secrets set "AzureSearch:Endpoint" "https://<service>.search.windows.net"
dotnet user-secrets set "AzureSearch:Key" "<key>"
dotnet user-secrets set "AzureSearch:IndexName" "<index-name>"
dotnet user-secrets set "AzureOpenAI:Endpoint" "https://<resource>.openai.azure.com"
dotnet user-secrets set "AzureOpenAI:Key" "<key>"
dotnet user-secrets set "AzureOpenAI:ChatDeployment" "<chat-deployment>"
dotnet user-secrets set "AzureOpenAI:EmbeddingDeployment" "<embedding-deployment>"
```

### Running

```bash
dotnet run                  # ingest docs, then run a smoke-test query
dotnet run -- --query-only  # query without re-ingesting
dotnet run -- --eval        # run the LLM-as-judge eval harness
dotnet run -- --reset       # delete + rebuild the index, then re-ingest
```

See [DocQA.Ingestion/README.md](DocQA.Ingestion/README.md) for full detail on
the eval harness, grounding/refusal behavior, and what a measured improvement
looks like in practice.

---

## DocQA.QueryService (Python/FastAPI)

A query-only HTTP API over the same Azure AI Search index. Requires the index
to already be populated by `DocQA.Ingestion`.

### Setup

```bash
cd DocQA.QueryService

cp .env.example .env
# Fill in the Azure values in .env (same search index, same OpenAI resource)

uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### Endpoints

**GET /health** — confirms config loads correctly without exposing secret values.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "search_index": "sk-docs",
  "chat_deployment": "gpt-4o-mini",
  "embedding_deployment": "text-embedding-3-small",
  "retrieval_top_k": 5
}
```

---

**POST /query** — embeds the question and returns the top-k matching chunks with
relevance scores. No generation step.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I register a plugin with the Kernel?"}'
```

```json
{
  "question": "How do I register a plugin with the Kernel?",
  "count": 5,
  "results": [
    {
      "id": "concepts-plugins-adding-native-plugins-chunk-0",
      "content": "...",
      "source": "concepts/plugins/adding-native-plugins.md",
      "score": 0.857
    }
  ]
}
```

---

**POST /ask** — full RAG: retrieves chunks, then calls Azure OpenAI chat
completion with a grounded prompt and returns an answer with citations.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I register a plugin with the Kernel?"}'
```

```json
{
  "question": "How do I register a plugin with the Kernel?",
  "answer": "You can register a plugin by calling `kernel.Plugins.AddFromType<T>()` [1] or by passing a plugin instance to `kernel.Plugins.Add()` [2].",
  "sources": [ ... ]
}
```

The optional `top_k` field overrides how many chunks are retrieved (default 5,
max 20):

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Kernel?", "top_k": 3}'
```

---

## Deploying DocQA.QueryService

The service is a standard ASGI app and can be deployed wherever Python runs.
Three reasonable options for a portfolio context:

**Local only (simplest)**
Run with `uv run uvicorn ...` and demo it live or screen-record it. Zero cost,
zero ops. Sufficient if you're showing the project rather than sharing a URL.

**Azure Container Apps**
Containerize with a simple `Dockerfile` (`python:3.12-slim`, install deps, run
uvicorn), then deploy to Azure Container Apps. Scales to zero when idle so it
costs nothing until a request comes in. Good if you want a live public URL.
Requires Docker and an Azure Container Registry.

**Azure App Service (Linux)**
Deploy via `az webapp up` or GitHub Actions. No Docker required — App Service
can run a Python app directly. Simpler than containers but has a minimum cost
even when idle unless you use the free tier (which has CPU/memory limits and
cold starts). Familiar if you're already using App Service for .NET.

For this project, local-only is the practical choice unless there's a specific
reason to have a persistent public endpoint.

---

## License / corpus attribution

Project source code: MIT.

The ingested corpus is the public Semantic Kernel documentation
(`MicrosoftDocs/semantic-kernel-docs`), CC-BY-4.0 for content / MIT for code
samples. It is not committed to this repo.
