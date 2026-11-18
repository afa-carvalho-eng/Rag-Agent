"""Command line for local use: python -m ragagent.cli {ingest,search,ask}"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import format_hits
from .config import Settings
from .factory import build_agent, build_embedder, build_store
from .ingest import SUPPORTED_SUFFIXES, extract_text, ingest_document


def _files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        found.extend(
            p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragagent")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="index files or folders (.md, .txt, .pdf)")
    ingest.add_argument("paths", nargs="+")
    search = commands.add_parser("search", help="retrieval only, no model call")
    search.add_argument("query")
    ask = commands.add_parser("ask", help="ask the agent a question (calls the model)")
    ask.add_argument("question")
    ask.add_argument("--session", default="cli")
    args = parser.parse_args(argv)

    settings = Settings.from_env()

    if args.command == "ingest":
        files = _files(args.paths)
        if not files:
            print("No .md, .txt or .pdf files found.", file=sys.stderr)
            return 1
        embedder, store = build_embedder(settings), build_store(settings)
        for path in files:
            count = ingest_document(
                path.name, extract_text(path.name, path.read_bytes()), embedder, store
            )
            print(f"{path.name}: {count} chunks")
        return 0

    if args.command == "search":
        embedder, store = build_embedder(settings), build_store(settings)
        print(format_hits(store.search(embedder.embed([args.query])[0], settings.top_k)))
        return 0

    answer = build_agent(settings).answer(args.session, args.question)
    print(answer.text)
    if answer.sources:
        print(f"\nRetrieved from: {', '.join(answer.sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
