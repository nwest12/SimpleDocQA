from app.core.azure_clients import get_openai_client
from app.core.config import Settings, get_settings
from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a documentation assistant for Microsoft Semantic Kernel. "
    "Answer the user's question using ONLY the provided context chunks. "
    "If the context does not contain enough information to answer, say so "
    "clearly rather than guessing. Cite sources by their [n] index when "
    "you use information from a chunk."
)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    if not chunks:
        return "(No relevant context was found.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = chunk.source or chunk.id
        parts.append(f"[{i}] (source: {source_label})\n{chunk.content}")

    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build a grounded prompt from the retrieved chunks and call Azure OpenAI
    chat completion to produce an answer.
    """
    settings: Settings = get_settings()
    client = get_openai_client()

    context_block = build_context_block(chunks)

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the context above, citing chunk numbers "
        "like [1], [2] where relevant."
    )

    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content or ""