#!/usr/bin/env python3
"""bench_library.py -- library-scale test: all six SearchTome domains merged
into one 413-section corpus with a real 3-level tree (book > chapter > section).

Question: at library scale, does the Jev walk preserve leaf recall while only
examining a fraction of the corpus?

Gold queries are the dataset's own test queries (6,005), not generated.

Systems compared:
  - Gemma flat over all 413 sections: R@1 / R@3 / R@10
  - Jev walk over the 3-level tree: leaves kept, set-hit, queries with 0 leaves
  - Agastya (walk leaves + gemma top3 union, gemma ordering): R@1 / R@3 /
    set-hit, plus share of the library examined

Jev walk results are cached per (domain, query_id) in
data/searchtome/jevlibrary_<domain>.json so reruns are free.

    .venv/bin/python bench_library.py --workers 8
"""

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from typesafe_sdk import Noul

from gemma import embed_documents, embed_query, load_model
from jev import ask

DOMAINS = ["education", "finance", "law", "medicine", "natural_sciences", "social_sciences"]
CACHE_DIR = Path(__file__).resolve().parent / "data" / "searchtome"
DOC_CHAR_LIMIT = 8000
JEV_JOIN = 0.85
GEMMA_TOP_K = 3
EMBED_FALLBACK_K = 5

NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def chapter_key(title: str) -> str | None:
    m = NUMBER_RE.match(title)
    if not m:
        return None
    return m.group(1).split(".")[0]


def build_library():
    """Return (entries, tree_root). entries: list of dicts with domain/title/content."""
    entries = []
    books = []
    for domain in DOMAINS:
        corpus = load_dataset("meghanadh/SearchTome", f"{domain}_corpus", split="corpus")
        book_title = "Book: " + domain.replace("_", " ").title()
        chapters: dict[str, dict] = {}
        order: list[str] = []
        for row in corpus:
            title = row["title"]
            content = row["content"] or ""
            entries.append({"domain": domain, "title": title, "content": content})
            key = chapter_key(title)
            if key is None:
                key = "misc"
            if key not in chapters:
                chapters[key] = {
                    "title": f"Chapter {key}" if key != "misc" else "Unnumbered",
                    "children": [],
                }
                order.append(key)
            chapters[key]["children"].append({"title": title, "children": []})
        books.append({
            "title": book_title,
            "children": [chapters[k] for k in order],
        })
    return entries, books


def _label(nodes: list[dict]) -> str:
    parts = []
    for node in nodes[-3:]:
        kind = "Book" if node["title"].startswith("Book:") else (
            "Chapter" if node["title"].startswith("Chapter") else "Subtopic")
        parts.append(f"{kind}: {node['title']}" if kind != "Subtopic" else node["title"])
    return " > ".join(parts)


def jev_scores(question: str, nodes: list[dict], path_nodes: list[dict]) -> list[float]:
    response = ask(
        state=(
            f"Question: {question}\n"
            f"Current section: {_label(path_nodes) if path_nodes else 'library root'}\n"
            "Candidate sections: "
            + "; ".join(_label(path_nodes[-2:] + [n]) for n in nodes)
        ),
        questions={
            f"s{i}": Noul(
                instructions=(
                    f"Does the section '{_label(path_nodes[-2:] + [n])}' "
                    "likely contain the answer to the question?"
                )
            )
            for i, n in enumerate(nodes)
        },
    )
    return [response.nouls[f"s{i}"].noul for i in range(len(nodes))]


def walk(nodes: list[dict], path_nodes: list[dict], question: str) -> tuple[list[str], int]:
    """Return (leaf titles kept, jev calls made)."""
    if not nodes:
        return [], 0
    scores = jev_scores(question, nodes, path_nodes)
    calls = 1
    best = max(scores) if scores else 0.0
    leaves: list[str] = []
    for node, score in zip(nodes, scores):
        if best > 0 and score >= JEV_JOIN * best:
            if node["children"]:
                sub, sub_calls = walk(node["children"], path_nodes + [node], question)
                leaves.extend(sub)
                calls += sub_calls
            else:
                leaves.append(node["title"])
    return leaves, calls


def metrics_from_ranks(ranks: list[int]) -> dict:
    arr = np.asarray(ranks)
    return {
        "R@1": float(np.mean(arr < 1)),
        "R@3": float(np.mean(arr < 3)),
        "nDCG@3": float(np.mean([1 / math.log2(r + 2) if r < 3 else 0.0 for r in arr])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test queries")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("building library tree ...", flush=True)
    entries, tree = build_library()
    titles = [e["title"] for e in entries]
    title_to_idx = {t: i for i, t in enumerate(titles)}
    print(f"library: {len(entries)} sections, {len(tree)} books", flush=True)

    model = load_model()
    doc_texts = [f"{e['title']}\n{e['content'][:DOC_CHAR_LIMIT]}" for e in entries]
    doc_vecs = embed_documents(model, doc_texts, batch_size=4)

    queries = []
    for domain in DOMAINS:
        tests = load_dataset("meghanadh/SearchTome", f"{domain}_queries", split="test")
        for row in tests:
            if row["gold_doc_title"] in title_to_idx:
                queries.append({"domain": domain, "qid": row["query_id"],
                                "query": row["query"], "gold": row["gold_doc_title"]})
    if args.limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(queries), size=min(args.limit, len(queries)), replace=False)
        queries = [queries[i] for i in sorted(idx)]
    print(f"queries: {len(queries)}", flush=True)

    q_vecs = model.encode(
        [q["query"] for q in queries],
        prompt_name="SearchQuery",
        normalize_embeddings=True,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # ---- flat dense over the whole 413-section library
    flat_ranks = []
    for qi, q in enumerate(queries):
        order = np.argsort(-(doc_vecs @ q_vecs[qi]))
        gold_idx = title_to_idx[q["gold"]]
        flat_ranks.append(int(np.where(order == gold_idx)[0][0]))
    flat = metrics_from_ranks(flat_ranks)
    print(f"\nGemma flat over 413 sections: R@1={flat['R@1']:.3f} R@3={flat['R@3']:.3f} "
          f"nDCG@3={flat['nDCG@3']:.3f}", flush=True)

    # ---- Jev walk over the 3-level tree (cached)
    per_domain_cache: dict[str, dict] = {}
    missing = []
    for q in queries:
        cache_path = CACHE_DIR / f"jevlibrary_{q['domain']}.json"
        if q["domain"] not in per_domain_cache:
            per_domain_cache[q["domain"]] = (
                json.loads(cache_path.read_text()) if cache_path.exists() else {}
            )
        if str(q["qid"]) not in per_domain_cache[q["domain"]]:
            missing.append(q)
    print(f"jev walk: {len(missing)} uncached queries, {args.workers} workers", flush=True)
    if missing:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(walk, tree, [], q["query"]): q for q in missing}
            for done, fut in enumerate(as_completed(futs), 1):
                q = futs[fut]
                leaves, calls = fut.result()
                per_domain_cache[q["domain"]][str(q["qid"])] = {"leaves": leaves, "calls": calls}
                if done % 250 == 0 or done == len(futs):
                    print(f"  walk {done}/{len(futs)} ({time.time()-t0:.0f}s)", flush=True)
        for domain, cache in per_domain_cache.items():
            (CACHE_DIR / f"jevlibrary_{domain}.json").write_text(json.dumps(cache))

    # ---- combine walk + gemma; score the product
    walk_hits, prod_ranks, prod_sets, prod_hits, calls_total, empty = [], [], [], [], 0, 0
    for qi, q in enumerate(queries):
        c = per_domain_cache[q["domain"]][str(q["qid"])]
        walked, calls = c["leaves"], c["calls"]
        calls_total += calls
        if not walked:
            empty += 1
        walk_hits.append(q["gold"] in walked)

        sims = doc_vecs @ q_vecs[qi]
        order_idx = np.argsort(-sims)
        gemma_order = [titles[int(i)] for i in order_idx]

        product_set: list[str] = list(walked)
        if product_set:
            for t in gemma_order[:GEMMA_TOP_K]:
                if t not in product_set:
                    product_set.append(t)
        else:
            product_set = list(gemma_order[:EMBED_FALLBACK_K])

        sim_of = {titles[i]: float(sims[i]) for i in range(len(titles))}
        ordered = sorted(product_set, key=sim_of.get, reverse=True)
        try:
            rank = ordered.index(q["gold"])
        except ValueError:
            rank = len(ordered) + 1000
        prod_ranks.append(rank)
        prod_sets.append(len(product_set))
        prod_hits.append(q["gold"] in product_set)

    walk_m = {"set_hit": float(np.mean(walk_hits)), "zero_leaf_rate": empty / len(queries),
              "avg_calls": calls_total / len(queries)}
    prod = {**metrics_from_ranks(prod_ranks), "set_hit": float(np.mean(prod_hits)),
            "avg_set": float(np.mean(prod_sets)),
            "avg_library_fraction": float(np.mean(prod_sets)) / len(titles)}

    print(f"\nJev walk only: set_hit={walk_m['set_hit']:.3f} "
          f"zero_leaf={walk_m['zero_leaf_rate']:.3f} calls/query={walk_m['avg_calls']:.1f}", flush=True)
    print(f"Agastya (walk+union): R@1={prod['R@1']:.3f} R@3={prod['R@3']:.3f} "
          f"nDCG@3={prod['nDCG@3']:.3f} set_hit={prod['set_hit']:.3f} "
          f"avg_set={prod['avg_set']:.1f} ({prod['avg_library_fraction']*100:.1f}% of library)",
          flush=True)

    out = CACHE_DIR / "results_library.json"
    out.write_text(json.dumps({
        "library_sections": len(entries),
        "queries": len(queries),
        "flat_gemma": flat,
        "jev_walk": walk_m,
        "agastya": prod,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
