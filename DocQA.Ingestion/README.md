# DocQA — Documentation Q&A over Semantic Kernel docs (RAG)

> A retrieval-augmented generation system that answers questions about Microsoft
> Semantic Kernel, grounded in the official documentation, with citations and an
> automated evaluation harness.
>
> I built this project to learn AI engineering in a hands-on manner. I chose to
> work in C# since I am more familiar with it than python and I use it daily in
> my job.

## What it does

Ask a natural-language question about Semantic Kernel and get an answer drawn
only from the official docs, with source citations. If the docs don't cover the
question, it declines instead of guessing.

## Getting the Docs

This project ingests the public Microsoft Semantic Kernel documentation, which is
not included in this repo. To run it, clone MicrosoftDocs/semantic-kernel-docs and
copy the semantic-kernel/concepts/ folder (and optionally get-started/ and frameworks/)
into a local docs/ directory at the project root. The content is licensed CC-BY-4.0
(code samples MIT) and belongs to Microsoft, which is why it's gitignored rather than
committed here.

## Architecture

Two pipelines over a shared vector store.

**Ingestion** — Markdown docs → clean (strip docs markup) → chunk (~500 tokens,
~50 token overlap) → embed (Azure OpenAI `text-embedding-3-small`) → store in
Azure AI Search.

**Query** — question → embed → vector similarity search (top-k) → assemble a
grounded prompt with retrieved chunks → Azure OpenAI `gpt-4o-mini` → answer with
citations.

| Layer | Choice |
|---|---|
| Backend | .NET 8 console (ingestion + query) |
| AI orchestration | Semantic Kernel / Microsoft.Extensions.AI |
| LLM | Azure OpenAI `gpt-4o-mini` |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store | Azure AI Search (free tier) |

## Grounding and refusal

The system prompt instructs the model to answer only from retrieved sources and
to say it doesn't know otherwise. This refusal behavior is the difference between
a trustworthy tool and a confident fabricator — verified by including
deliberately unanswerable questions in the eval set, which the system correctly
declined 100% of the time.

## Evaluation (the part I'm most interested in)

I built an automated eval harness rather than judging answers by eye.

- **Eval set:** 10 questions across four categories — answerable, specific
  (exact technical detail), unanswerable (should refuse), and multi-part.
- **Scoring:** LLM-as-judge (gpt-4o-mini grades each answer 1–5) plus deterministic
  checks (expected-term presence, correct refusal).
- **Reporting:** per-category average, refusal accuracy, overall score.

### A measured improvement (before → after)



The first eval run scored **4.80/5 overall**, but the per-category breakdown
showed `specific` questions lagging at **4.00** while everything else scored 5.00.
In reviewing the retrieved chunks and my understanding of the source material, I 
considered whether the "zonepivot" markup and answers for multiple programming 
languages (C#, Java, and Python) were adding unnecessary noise that was degrading
the specificity metric. 

To address this, I removed the ingested documents and created a method to target
the noisy markup content before chunking. The cleaned documents were then reingested
and re-evaluated. On the next pass through, the eval scored a perfect 5/5 and the 
`specific` metric **increased from 4.8 to 5.0** without degrading other metrics.

### Limitations of this evaluation

- **Same-model judge:** answers and grades both come from gpt-4o-mini. In these
  situations of a model unknowingly grading an output produced by itself introduces
  grading bias. The grader has the same weights as the generator, so it will naturally
  favor its own outputs, potentially leading to higher scores than warranted.
- **Metric saturation:** after the fix, every category scored a perfect 5.00.
  While I'd like to call it a perfect system, there are still many factors that
  could be causing this high score. The sample size of 10 is very small and would
  need to be increased to accurately evaluate the system. The criteria are also
  limited and not strenuous enough to approve for production. 

## What I'd do next

- more strict evaluation criteria
- larger sample sizes for evaluation
- create as an API with an angular UI
- different judge model
- filter chunks to a single language to keep responses focused

## Running it

```bash
dotnet run                 # ingest docs + smoke test
dotnet run -- --query-only # query without re-ingesting
dotnet run -- --eval       # run the eval harness
dotnet run -- --reset      # delete + rebuild the index, then re-ingest
```

Secrets (Azure OpenAI + AI Search endpoints/keys) are stored via
`dotnet user-secrets`, not committed.

## Notes / licensing

The ingested corpus is the public Semantic Kernel documentation
(`MicrosoftDocs/semantic-kernel-docs`), CC-BY-4.0 for content / MIT for samples.
It is not committed to this repo; the ingestion step reads it from a local `docs/`
folder.

When starting this project, I came in with knowledge of C# and .NET as a backend
developer. I hit some snags while getting started around imports and syntax issues
relating to the unfamiliar libraries. However, after working through the initial setup
and making a few iterations on the various tasks of this program, I was able to create
a simple rag pipeline for ingesting, retrieving, and evaluating answers to questions
regarding MS semantic kernel.