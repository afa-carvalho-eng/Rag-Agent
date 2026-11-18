"""Retrieval eval: does the right document come back for each question?

It measures the retrieval step only (chunking, embedding, search), not the
quality of the model's answers. By default it runs offline with the hashing
embedder. With RAG_BACKEND=aws (and the variables that backend requires) it
scores the Bedrock embedder instead, still against an in-memory index.

Read hit@1 and MRR. On the five sample documents hit@k is close to
guaranteed for k=4, so it is reported but not gated on.

Usage: python evals/retrieval_eval.py [--k 4] [--min-hit-at-1 0.7]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ragagent.config import Settings
from ragagent.embeddings import Embedder
from ragagent.factory import build_embedder
from ragagent.ingest import SUPPORTED_SUFFIXES, extract_text, ingest_document
from ragagent.vectorstore import LocalVectorStore

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalResult:
    total: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    misses: list[str]


def run_eval(embedder: Embedder, docs_dir: Path, dataset: Path, k: int) -> EvalResult:
    store = LocalVectorStore()
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            ingest_document(path.name, extract_text(path.name, path.read_bytes()), embedder, store)

    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    if not cases:
        raise ValueError(f"{dataset} contains no eval cases")

    first, within_k, reciprocal, misses = 0, 0, 0.0, []
    for case in cases:
        hits = store.search(embedder.embed([case["question"]])[0], k)
        sources = [hit.chunk.source for hit in hits]
        if case["expected_source"] in sources:
            rank = sources.index(case["expected_source"]) + 1
            within_k += 1
            first += rank == 1
            reciprocal += 1 / rank
        else:
            misses.append(case["question"])
    total = len(cases)
    return EvalResult(total, first / total, within_k / total, reciprocal / total, misses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-hit-at-1", type=float, default=0.0)
    parser.add_argument("--docs", type=Path, default=ROOT / "sample_docs")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals" / "retrieval_dataset.jsonl")
    args = parser.parse_args()

    embedder = build_embedder(Settings.from_env())
    result = run_eval(embedder, args.docs, args.dataset, args.k)
    print(f"embedder:  {type(embedder).__name__}")
    print(f"questions: {result.total}")
    print(f"hit@1:     {result.hit_at_1:.2f}")
    print(f"hit@{args.k}:     {result.hit_at_k:.2f}")
    print(f"MRR:       {result.mrr:.2f}")
    for question in result.misses:
        print(f"  miss: {question}")
    if result.hit_at_1 < args.min_hit_at_1:
        print(f"FAIL: hit@1 is below {args.min_hit_at_1:.2f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
