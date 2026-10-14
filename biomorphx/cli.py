"""Command-Line Interface for BioMorphX."""
import argparse
import json
from .core import __version__, list_snippets, get_snippet, run_snippet, self_test


def main():
    parser = argparse.ArgumentParser(
        prog="biomorphx",
        description="BioMorphX v1.0.0 — Cross-Scale Mechano-Genomic, Biomaterial, Nanofab & Clinical SDK",
    )
    parser.add_argument("--version", action="version", version=f"biomorphx {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list-snippets", help="List all 12 built-in one-liner biomedical snippets")
    p_show = sub.add_parser("show-snippet", help="Print Python code for a specific snippet")
    p_show.add_argument("name", help="Snippet ID (e.g. 01_protein_mechano_unfolding)")

    p_run = sub.add_parser("run-snippet", help="Execute a built-in snippet and print JSON output")
    p_run.add_argument("name", help="Snippet ID")

    sub.add_parser("self-test", help="Run deterministic self-test across all 12 pipelines")

    args = parser.parse_args()
    if args.cmd == "list-snippets":
        print(json.dumps(list_snippets(), indent=2))
    elif args.cmd == "show-snippet":
        print(get_snippet(args.name))
    elif args.cmd == "run-snippet":
        out = run_snippet(args.name)
        print(json.dumps(out, indent=2, default=str))
    else:
        print(json.dumps(self_test(), indent=2))


if __name__ == "__main__":
    main()
