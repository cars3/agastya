#!/usr/bin/env python3
"""Benchmark the core.py endpoint: does Jev navigate to the gold leaf?

Generates one question per sampled leaf (gold = that leaf), POSTs each question to
the running /ask endpoint, and checks whether the gold leaf is among the leaves the
system returned.

    .venv/bin/python bench_core.py --n 20 --url http://127.0.0.1:8000/ask
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from bench_book import generate_queries

BASE = Path(__file__).resolve().parent
LEAVES = BASE / "data" / "books" / "slp3" / "leaves.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--url", default="http://127.0.0.1:8000/ask")
    ap.add_argument("--out", type=Path, default=BASE / "data/books/slp3/bench_core.json")
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--base", default="http://localhost:11434/v1")
    ap.add_argument("--key", default="ollama")
    ap.add_argument(
        "--leaves",
        type=Path,
        default=LEAVES,
        help="leaves.jsonl for the book (defaults to the SLP3 fixture)",
    )
    ap.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="reuse queries from a prior bench_core.json (rows with gold_path) or queries.jsonl",
    )
    args = ap.parse_args()

    leaves = [json.loads(line) for line in args.leaves.open()]
    usable = [leaf for leaf in leaves if leaf["chars"] >= 200]
    step = len(usable) / args.n
    sample = [usable[int(i * step)] for i in range(args.n)]

    if args.queries:
        text = args.queries.read_text()
        try:
            raw = json.loads(text)
            rows = raw.get("rows") if isinstance(raw, dict) else raw
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        queries = []
        for row in rows:
            gold_path = list(
                row.get("gold_path")
                or (row.get("path", []) + [row.get("gold_title") or row.get("gold", "")])
            )
            queries.append(
                {"query": row["query"], "path": gold_path[:-1], "gold_title": gold_path[-1]}
            )
        print(f"reusing {len(queries)} queries from {args.queries}\n")
    else:
        print(f"generating {len(sample)} questions with {args.model} ...")
        queries = generate_queries(sample, args.base, args.key, args.model, workers=6)
        print(f"generated {len(queries)} questions\n")

    hits, leaf_counts, latencies, rows = 0, [], [], []
    for query in queries:
        t0 = time.perf_counter()
        response = requests.post(args.url, json={"question": query["query"]}, timeout=300)
        latencies.append(time.perf_counter() - t0)
        response.raise_for_status()

        paths = {tuple(leaf["path"]) for leaf in response.json()["leaves"]}
        gold = tuple(query["path"] + [query["gold_title"]])
        hit = gold in paths
        hits += hit
        leaf_counts.append(len(paths))
        rows.append({"query": query["query"], "gold": query["gold_title"], "hit": hit,
                     "returned": len(paths), "gold_path": list(gold)})
        print(("hit  " if hit else "miss ") + query["gold_title"][:60] + f" | {len(paths)} leaves")

    summary = {
        "queries": len(queries),
        "leaf_hit_rate": hits / len(queries) if queries else 0.0,
        "avg_leaves_returned": statistics.mean(leaf_counts) if leaf_counts else 0.0,
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "max_latency_s": max(latencies) if latencies else 0.0,
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n=== core.py benchmark ===")
    print(f"queries:        {summary['queries']}")
    print(f"leaf hit rate:  {summary['leaf_hit_rate']:.1%}")
    print(f"avg leaves:     {summary['avg_leaves_returned']:.2f}")
    print(f"p50 latency:    {summary['p50_latency_s']:.2f}s")
    print(f"max latency:    {summary['max_latency_s']:.2f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
