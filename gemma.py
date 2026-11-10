#!/usr/bin/env python3
"""
gemma.py -- chunk text, embed the chunks locally with Google EmbeddingGemma 2,
and rank them against a query with cosine similarity.

Model : google/embeddinggemma-2 (740M params, text backbone only here)
Dims  : 768 (Matryoshka: 512 / 256 / 128 also supported)
Runs  : CUDA when available (bf16 on Ampere+, fp32 on Turing and CPU), or mps.
        AMD ROCm builds report as cuda.
        Never float16: the model overflows it and silently returns NaN/garbage.

Setup:
    uv venv .venv --python 3.12
    uv pip install torch torchvision                # PyPI wheels are CUDA builds
    uv pip install sentence-transformers

Run:
    .venv/bin/python gemma.py

If another process is using the GPU, force CPU:
    GEMMA_DEVICE=cpu .venv/bin/python gemma.py
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from device import pick_device

MODEL_ID = "google/embeddinggemma-2"
SENTENCES_PER_CHUNK = 3   # chunk = this many sentences, whatever their length
SENTENCE_OVERLAP = 1      # sentences repeated at the start of the next chunk
MAX_CHUNK_CHARS = 1200    # hard-split sentences longer than this (tables, dense runs)
TOP_K = 5

DEVICE = pick_device()
# float16 is unsafe here (EmbeddingGemma 2 overflows it and returns NaN or silent
# garbage). bfloat16 needs Ampere or newer; Turing (RTX 20xx) falls back to fp32.
if DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32
BATCH_SIZE = 32 if DEVICE == "cuda" else 8

SAMPLE_DOC = """Retrieval-Augmented Generation: A Practical Primer

Retrieval-Augmented Generation (RAG) pairs a search system with a language model. Instead of
relying only on what the model memorised during training, RAG fetches relevant passages from
an external corpus at query time and puts them in the prompt. The model then answers from that
evidence. This keeps answers grounded, makes citations possible, and lets you update knowledge
by re-indexing documents instead of retraining a model.

Chunking is the first design decision. Documents are split into passages small enough to embed
and retrieve precisely, but large enough to stay self-contained. A common starting point is a
few hundred tokens per chunk with 10-20 percent overlap, so a sentence that straddles a boundary
still appears intact in at least one chunk. Chunks that are too small lose the context needed to
interpret them; chunks that are too large blend several topics and dilute the embedding, making
similarity scores less discriminative. Splitting on structural boundaries -- headings, paragraphs,
sentences -- usually beats cutting at a fixed character count.

Embeddings turn each chunk into a dense vector. EmbeddingGemma 2 maps text into a 768-dimensional
space, and because it was trained with Matryoshka Representation Learning the leading 512, 256 or
128 dimensions can be kept and re-normalised with little loss in quality. Queries and documents
use different instruction prefixes: a search query is prefixed with a task description, while a
document is prefixed with its title (or "none"). Omitting the prefixes still works but lowers
retrieval precision.

Vector search ranks chunks by cosine similarity between the query vector and the stored chunk
vectors. With normalised vectors this is just a dot product, which is why production systems
normalise at index time. Exact search scans every vector; approximate indexes such as HNSW trade a
little recall for speed at millions of vectors. Metadata filters (date, source, language) are
usually applied alongside the vector search so that obvious mismatches never enter the candidate
set.

Reranking is an optional second stage. A cross-encoder reads the query and each candidate chunk
together and produces a relevance score, which is far more accurate than comparing two independent
vectors but too slow to run over the whole corpus. Typical pipelines retrieve 50-100 candidates by
vector similarity, rerank them, and pass the best 3-5 passages to the language model.

Evaluation should be built before tuning. Recall at k measures whether the right passage appears
in the candidate set at all; nDCG measures whether it is ranked near the top; answer faithfulness
measures whether the generated answer is supported by the retrieved text. A small hand-labelled
set of queries with known correct passages catches most regressions, and it makes chunk-size and
model changes comparable instead of anecdotal.

Known limitations matter. Fixed chunk boundaries can separate a definition from its explanation,
embeddings often miss exact identifiers such as part numbers or error codes, and an index goes
stale the moment documents change. Hybrid search that combines dense vectors with keyword matching
covers the identifier case, and a re-indexing schedule keeps the corpus current."""

SAMPLE_QUERIES = [
    "How should I choose chunk size and overlap?",
    "Why might an exact part number be missed by vector search?",
]


def _split_long_sentence(sentence: str) -> list[str]:
    """Hard-split a sentence longer than MAX_CHUNK_CHARS at word boundaries."""
    if len(sentence) <= MAX_CHUNK_CHARS:
        return [sentence]
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + len(word) + 1 > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_text(
    text: str,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
    overlap: int = SENTENCE_OVERLAP,
) -> list[str]:
    """Group whole sentences into chunks, so chunk length varies with the text.

    A chunk is `sentences_per_chunk` consecutive sentences; the next chunk starts
    `overlap` sentences back. Nothing is cut mid-sentence and there is no character
    budget, so a long sentence simply makes a long chunk. Sentences longer than
    MAX_CHUNK_CHARS (tables, dense runs) are hard-split at word boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(_split_long_sentence(sentence))
    sentences = expanded

    step = max(1, sentences_per_chunk - overlap)
    chunks: list[str] = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
        if i + sentences_per_chunk >= len(sentences):
            break
    return chunks


def load_model() -> SentenceTransformer:
    """Load the text-only variant on the GPU when one is available."""
    model = SentenceTransformer(
        MODEL_ID,
        device=DEVICE,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": DTYPE},
    )
    model.max_seq_length = 8192                  # the model's real context window
    return model


def embed_documents(model: SentenceTransformer, chunks: list[str], batch_size: int | None = None) -> np.ndarray:
    """Embed corpus chunks with the document prefix ('title: none | text: ...')."""
    return model.encode(
        chunks,
        prompt_name="Document",
        normalize_embeddings=True,
        batch_size=batch_size or BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Embed a search query with the query prefix ('task: search result | query: ...')."""
    return model.encode(
        [query],
        prompt_name="SearchQuery",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]


def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = TOP_K) -> list[tuple[int, float]]:
    """Cosine similarity == dot product because both sides are L2-normalised."""
    if doc_vecs.size == 0:
        return []
    scores = doc_vecs @ query_vec
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


def embed_and_search(
    text: str,
    queries: list[str],
    *,
    model: SentenceTransformer | None = None,
    k: int = TOP_K,
) -> dict:
    """Chunk `text`, embed it, and rank every query against the chunks.

    One-call entry point:

        from gemma import embed_and_search
        out = embed_and_search(document_text, ["how does chunking work?"])
        out["results"]["how does chunking work?"][0]["text"]

    Pass model=load_model() to reuse an already-loaded model across calls.

    Returns {"chunks", "embeddings", "device", "results"}; results maps each query
    to a list of {"chunk", "score", "text"} dicts.
    """
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": [], "embeddings": np.zeros((0, 768), dtype=np.float32),
                "device": str(model.device) if model else DEVICE,
                "results": {query: [] for query in queries}}
    model = model or load_model()
    doc_vecs = embed_documents(model, chunks)
    results = {}
    for query in queries:
        query_vec = embed_query(model, query)
        results[query] = [
            {"chunk": idx, "score": score, "text": chunks[idx]}
            for idx, score in top_k(query_vec, doc_vecs, k)
        ]
    return {"chunks": chunks, "embeddings": doc_vecs, "device": str(model.device), "results": results}


def main() -> None:
    print("Loading EmbeddingGemma 2 (text-only)...")
    t0 = time.time()
    model = load_model()
    weight_dtype = next(model.parameters()).dtype
    print(f"Loaded in {time.time() - t0:.1f}s | device={model.device} | dtype={weight_dtype} "
          f"| dim={model.get_embedding_dimension()}")
    if model.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")

    t0 = time.time()
    out = embed_and_search(SAMPLE_DOC, SAMPLE_QUERIES, model=model)
    chunks, doc_vecs = out["chunks"], out["embeddings"]
    print(f"Document: {len(SAMPLE_DOC)} chars -> {len(chunks)} chunks "
          f"(sentences_per_chunk={SENTENCES_PER_CHUNK}, overlap={SENTENCE_OVERLAP})")
    print(f"Embedded {len(chunks)} chunks in {time.time() - t0:.2f}s "
          f"| matrix={doc_vecs.shape} | row norm={np.linalg.norm(doc_vecs[0]):.4f}")
    if model.device.type == "cuda":
        print(f"GPU peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    for query, hits in out["results"].items():
        print(f"\nQuery: {query}")
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. score={hit['score']:.4f} chunk={hit['chunk']} | {hit['text'][:100]}...")


if __name__ == "__main__":
    main()
