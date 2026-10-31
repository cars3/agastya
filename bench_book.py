#!/usr/bin/env python3
"""
bench_book.py -- benchmark a retriever on one book split into ToC leaves.

Pipeline:
  1. read leaves.jsonl (from split_pdf.py)
  2. generate one question per leaf with an OpenAI-compatible LLM (STAIR does the
     same thing with Mixtral; here it defaults to a local Ollama endpoint)
  3. embed every leaf with EmbeddingGemma 2 and every question, dot-product the
     normalised vectors, rank
  4. score R@1 / R@3 / R@5 / nDCG@3 / nDCG@5 / MRR, overall and per heading level

The gold label is the leaf the question was generated from -- exactly the
SearchTome setup (query -> section title), just with a book you own.

Usage:
    .venv/bin/python bench_book.py --leaves data/books/slp3/leaves.jsonl \
        --out data/books/slp3/bench --limit 60 --workers 6

    # reuse questions you already generated
    .venv/bin/python bench_book.py --leaves ... --queries data/books/slp3/bench/queries.jsonl

Generation endpoint (defaults to local Ollama):
    BOOK_LLM_BASE=http://localhost:11434/v1
    BOOK_LLM_KEY=ollama
    BOOK_LLM_MODEL=deepseek-v4.1-flash:cloud
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
DEFAULT_BASE = os.getenv("BOOK_LLM_BASE", "http://localhost:11434/v1")
DEFAULT_KEY = os.getenv("BOOK_LLM_KEY", "ollama")
DEFAULT_MODEL = os.getenv("BOOK_LLM_MODEL", "gemma4:31b-cloud")
MIN_CHARS = 200          # leaves shorter than this carry no usable text
DOC_CHAR_LIMIT = 8000   # keep prompts/embeddings well inside the 8K-token window

PROMPT = """You are writing a retrieval benchmark question for a textbook section.

Write ONE specific question that can only be answered by the section below, plus a short
answer taken from the text.

Rules:
- Do not mention the section title, chapter name, or section number in the question.
- The question must be self-contained and answerable from this text alone.
- Prefer "how" / "why" / "what is" questions about the actual content.
- Return strict JSON only: {{"question": "...", "answer": "..."}}

Section title: {title}

Section text:
{text}
"""

# Same prompt with the title line removed, for blinded question generation.
PROMPT_BLIND = PROMPT.replace("\nSection title: {title}\n", "\n")


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])


def generate_one(leaf: dict, base: str, key: str, model: str, retries: int = 2, blind: bool = False) -> dict | None:
    template = PROMPT_BLIND if blind else PROMPT
    prompt = template.format(title=leaf["title"], text=leaf["content"][:DOC_CHAR_LIMIT])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = parse_json(content)
            question = str(data["question"]).strip()
            answer = str(data.get("answer", "")).strip()
            if not question:
                raise ValueError("empty question")
            return {
                "query": question,
                "answer": answer,
                "gold_leaf_id": leaf.get("leaf_id", ""),
                "gold_section_id": leaf["section_id"],
                "gold_title": leaf["title"],
                "level": leaf["level"],
                "path": leaf["path"],
            }
        except Exception as exc:  # noqa: BLE001 - report and retry
            if attempt == retries:
                print(f"  ! failed for '{leaf['title'][:50]}': {type(exc).__name__}: {exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def generate_queries(leaves: list[dict], base: str, key: str, model: str, workers: int, blind: bool = False) -> list[dict]:
    queries: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, leaf, base, key, model, 2, blind): leaf for leaf in leaves}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row:
                queries.append(row)
            if done % 10 == 0 or done == len(futures):
                print(f"  generated {done}/{len(futures)} questions ({len(queries)} ok)")
    return queries


def load_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    model = SentenceTransformer(
        MODEL_ID,
        device=device,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": dtype},
    )
    model.max_seq_length = 8192
    return model


def metrics(ranks: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(np.mean(ranks < k))
    for k in (3, 5):
        out[f"nDCG@{k}"] = float(np.mean([1 / math.log2(r + 2) if r < k else 0.0 for r in ranks]))
    out["MRR"] = float(np.mean(1 / (ranks + 1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=None, help="reuse queries.jsonl instead of generating")
    ap.add_argument("--limit", type=int, default=0, help="use only N evenly spaced leaves (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--blind", action="store_true", help="withhold the section title from the question writer")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="generation model")
    ap.add_argument("--base", default=DEFAULT_BASE, help="OpenAI-compatible base URL")
    ap.add_argument("--key", default=DEFAULT_KEY)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    leaves = [json.loads(line) for line in args.leaves.open()]
    usable = [leaf for leaf in leaves if leaf["chars"] >= MIN_CHARS and leaf["content"].strip()]
    if args.limit and args.limit < len(usable):
        step = len(usable) / args.limit
        usable = [usable[int(i * step)] for i in range(args.limit)]
    print(f"leaves: {len(leaves)} total, {len(usable)} usable "
          f"(dropped {len(leaves) - len(usable)} short/empty)")

    if args.queries:
        queries = [json.loads(line) for line in args.queries.open()]
        print(f"queries: {len(queries)} loaded from {args.queries}")
    else:
        print(f"generating questions with {args.model} via {args.base} ...")
        t0 = time.time()
        queries = generate_queries(usable, args.base, args.key, args.model, args.workers, blind=args.blind)
        print(f"generated {len(queries)} questions in {time.time() - t0:.1f}s")
        (args.out / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
        print(f"wrote {args.out/'queries.jsonl'}")
    if not queries:
        raise SystemExit("no queries; nothing to score")

    print("loading EmbeddingGemma 2 (text-only) ...")
    model = load_embedder()
    t0 = time.time()
    doc_texts = [f"{leaf['title']}\n{leaf['content'][:DOC_CHAR_LIMIT]}" for leaf in usable]
    doc_vecs = model.encode(doc_texts, prompt_name="Document", normalize_embeddings=True,
                            batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    q_vecs = model.encode([q["query"] for q in queries], prompt_name="SearchQuery",
                          normalize_embeddings=True, batch_size=16, convert_to_numpy=True,
                          show_progress_bar=False)
    print(f"embedded {len(usable)} leaves and {len(queries)} queries in {time.time() - t0:.1f}s "
          f"on {model.device} | matrix={doc_vecs.shape}")
    if model.device.type == "cuda":
        print(f"GPU peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    # rank every query against every leaf; gold = the leaf the question came from
    scores = q_vecs @ doc_vecs.T
    order = np.argsort(-scores, axis=1)
    leaf_to_idx = {leaf["leaf_id"]: i for i, leaf in enumerate(usable)}
    section_to_idx = {leaf["section_id"]: i for i, leaf in enumerate(usable)}
    ranks: list[int] = []
    levels_used: list[int] = []
    per_query: list[dict] = []
    for qi, query in enumerate(queries):
        gold_idx = leaf_to_idx.get(query.get("gold_leaf_id", ""))
        if gold_idx is None:
            gold_idx = section_to_idx.get(query["gold_section_id"])
        if gold_idx is None:
            continue
        rank = int(np.where(order[qi] == gold_idx)[0][0])
        ranks.append(rank)
        levels_used.append(query["level"])
        per_query.append({
            "query": query["query"],
            "gold_title": query["gold_title"],
            "rank": rank,
            "top1": usable[int(order[qi][0])]["title"],
            "hit": rank < 3,
        })

    ranks_arr = np.asarray(ranks)
    overall = metrics(ranks_arr)
    print(f"\nscored {len(ranks_arr)} queries against {len(usable)} leaves")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()))

    by_level = {}
    for level in sorted(set(levels_used)):
        idx = [i for i, lv in enumerate(levels_used) if lv == level]
        by_level[f"level_{level}"] = metrics(ranks_arr[idx])
    for name, values in by_level.items():
        print(f"  {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in values.items()))

    results = {
        "embedding_model": MODEL_ID,
        "generation_model": args.model if not args.queries else "reused queries",
        "documents": len(usable),
        "queries": len(ranks_arr),
        "metrics": overall,
        "by_level": by_level,
        "per_query": per_query,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out/'results.json'}")
    print("\nmisses (rank >= 3):")
    for row in sorted(per_query, key=lambda r: -r["rank"])[:5]:
        print(f"  rank {row['rank']:>2} | Q: {row['query'][:70]}")
        print(f"           gold: {row['gold_title'][:70]}")
        print(f"           top1: {row['top1'][:70]}")


if __name__ == "__main__":
    main()
