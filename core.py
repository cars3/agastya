"""ToC-navigated RAG over one book, exposed as a small FastAPI endpoint.

Pipeline per question:
  1. Embed the question once with EmbeddingGemma 2.
  2. Walk the ToC with Jev (noul per candidate section, relative cutoff).
  3. Candidate leaves = walked leaves union top-k leaves by whole-leaf
     embedding similarity; if the walk kills every branch, fall back to the
     top-k embedding leaves so a question never returns zero leaves.
  4. Return the top-5 precomputed chunks per candidate leaf.

Leaf vectors and chunk vectors are embedded once at startup, so the request
path does exactly one embedding call (the query) plus one Jev call.

Run as a server:
    .venv/bin/python core.py --serve
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "What is TTS?"}'

Or ask once from the CLI:
    .venv/bin/python core.py "What is TTS?"
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typesafe_sdk import Noul

from gemma import chunk_text, embed_documents, embed_query, load_model, top_k
from jev import ask

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RAG_DATA") or BASE / "data" / "books" / "slp3")
THRESHOLD = 0.85
EMBED_TOP_K = 3          # embedding leaves added alongside the walk
EMBED_FALLBACK_K = 5     # embedding leaves used when the walk returns nothing
CHUNKS_PER_LEAF = 5
DOC_CHAR_LIMIT = 8000    # same cap the benchmark embeds

with open(DATA / "toc.json", encoding="utf-8") as file:
    document = json.load(file)

with open(DATA / "leaves.jsonl", encoding="utf-8") as file:
    leaves = [json.loads(line) for line in file]

model = load_model()     # once, reused by every request

# ---- Precompute: one vector per leaf (title + capped content) and per chunk.
leaf_entries = [leaf for leaf in leaves if leaf["content"].strip()]
leaf_texts = [f"{leaf['title']}\n{leaf['content'][:DOC_CHAR_LIMIT]}" for leaf in leaf_entries]

FULL_VECS = (
    embed_documents(model, leaf_texts, batch_size=2)
    if leaf_texts
    else np.zeros((0, 768), dtype=np.float32)
)

chunk_texts: list[str] = []
chunk_bounds: list[tuple[int, int]] = []
for leaf in leaf_entries:
    chunks = chunk_text(leaf["content"])
    chunk_bounds.append((len(chunk_texts), len(chunk_texts) + len(chunks)))
    chunk_texts.extend(chunks)
CHUNK_VECS = (
    embed_documents(model, chunk_texts, batch_size=8)
    if chunk_texts
    else np.zeros((0, 768), dtype=np.float32)
)

# path in leaves.jsonl holds only the ancestors, so add the leaf title to match
leaf_index = {
    tuple(leaf["path"] + [leaf["title"]]): i for i, leaf in enumerate(leaf_entries)
}
print(f"precomputed {len(leaf_entries)} leaf vectors and {len(chunk_texts)} chunk vectors from {DATA}",
      flush=True)


class Question(BaseModel):
    question: str


def _lineage_label(nodes: list[dict]) -> str:
    """Render up to 3 levels of ToC lineage: Section > Chapter > Subtopic."""
    parts = []
    for node in nodes[-3:]:
        level = node.get("level", 0)
        kind = {1: "Section", 2: "Chapter", 3: "Subtopic"}.get(level, f"Level {level}")
        number = f" {node['number']}" if node.get("number") else ""
        parts.append(f"{kind}{number}: {node['title']}")
    return " > ".join(parts)


def jev_scores(nodes, path_nodes, question):
    """One Jev call: a noul probability per candidate section.

    Each candidate is presented with up to 3 levels of lineage from the ToC
    index (section > chapter > subtopic) so Jev can judge it in context.
    """
    context = _lineage_label(path_nodes) if path_nodes else "book root"
    response = ask(
        state=(
            f"Question: {question}\n"
            f"Current section: {context}\n"
            "Candidate sections: "
            + "; ".join(_lineage_label(path_nodes[-2:] + [node]) for node in nodes)
        ),
        questions={
            f"s{i}": Noul(
                instructions=(
                    f"Does the section '{_lineage_label(path_nodes[-2:] + [node])}' "
                    "likely contain the answer to the question?"
                )
            )
            for i, node in enumerate(nodes)
        },
    )
    return [response.nouls[f"s{i}"].noul for i, node in enumerate(nodes)]


def kill(path, score):
    """A path Jev is not satisfied with dies here."""
    print(f"kill {score:.2f} | " + " > ".join(path[-2:]))


def walk(nodes, path, path_nodes, question, found):
    if not nodes:
        print("reached leaf:", " > ".join(path[-2:]))
        found.append(path)
        return

    scores = jev_scores(nodes, path_nodes, question)
    best = max(scores)

    for node, score in zip(nodes, scores):
        # noul scores are conservative here (the root tops out around 0.7), so 0.85
        # means "within 15% of the best branch at this level". Use `score` instead
        # of `score / best` if you want a literal 0.85 cutoff.
        relative = score / best if best else 0
        if relative >= THRESHOLD:
            walk(node["children"], path + [node["title"]], path_nodes + [node], question, found)
        else:
            kill(path + [node["title"]], score)


def answer(question: str) -> dict:
    """Walk the ToC with Jev, union with embedding top-k, return chunk hits."""
    query_vec = embed_query(model, question)
    sims = FULL_VECS @ query_vec
    order = np.argsort(-sims)

    found: list[list[str]] = []
    walk(document["tree"], [], [], question, found)

    candidates: dict[tuple, dict] = {}
    for path in found:
        key = tuple(path)
        idx = leaf_index.get(key)
        if idx is not None:
            candidates[key] = {"idx": idx, "source": "walk"}

    if candidates:
        for i in order[:EMBED_TOP_K]:
            key = tuple(leaf_entries[int(i)]["path"] + [leaf_entries[int(i)]["title"]])
            if key not in candidates:
                candidates[key] = {"idx": int(i), "source": "embed"}
    else:
        for i in order[:EMBED_FALLBACK_K]:
            key = tuple(leaf_entries[int(i)]["path"] + [leaf_entries[int(i)]["title"]])
            candidates[key] = {"idx": int(i), "source": "fallback"}

    results = []
    for key, cand in candidates.items():
        i = cand["idx"]
        leaf = leaf_entries[i]
        start, end = chunk_bounds[i]
        hits = [
            {"chunk": idx, "score": score, "text": chunk_texts[start + idx]}
            for idx, score in top_k(query_vec, CHUNK_VECS[start:end], CHUNKS_PER_LEAF)
        ]
        results.append({
            "path": list(key),
            "source": cand["source"],
            "similarity": float(sims[i]),
            "page_start": leaf["page_start"],
            "page_end": leaf["page_end"],
            "chunks": hits,
        })
    return {"question": question, "leaves": results}


app = FastAPI(title="ToC RAG")


@app.get("/health")
def health():
    return {"ok": True, "leaves": len(leaves)}


@app.post("/ask")
def ask_endpoint(req: Question):
    return answer(req.question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
        return

    question = args.question or input("question: ")
    result = answer(question)
    for leaf in result["leaves"]:
        print(f"\nleaf [{leaf['source']}] sim={leaf['similarity']:.3f}:", " > ".join(leaf["path"]))
        print("pages:", leaf["page_start"], "-", leaf["page_end"])
        for hit in leaf["chunks"]:
            print(f"  {hit['score']:.3f} | {hit['text'][:120]}")


if __name__ == "__main__":
    main()
