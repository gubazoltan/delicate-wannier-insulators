#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List

import nbformat


def find_notebooks(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*.ipynb"):
        # Skip checkpoint files
        if any(seg == ".ipynb_checkpoints" for seg in p.parts):
            continue
        out.append(p)
    return sorted(out)


def filter_paths(paths: Iterable[Path], pattern: str | None, exclude: str | None) -> List[Path]:
    def match(p: Path, pat: str) -> bool:
        return re.search(pat, str(p).replace("\\", "/"), flags=re.IGNORECASE) is not None

    out: List[Path] = []
    for p in paths:
        if pattern and not match(p, pattern):
            continue
        if exclude and match(p, exclude):
            continue
        out.append(p)
    return out


def clear_outputs(nb_path: Path) -> bool:
    """
    Remove cell outputs and execution counts. Return True if any changes were made.
    """
    nb = nbformat.read(nb_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
            # Optionally clear transient execution metadata if present
            md = cell.get("metadata", {})
            # Common transient fields sometimes added by tools; ignore if missing
            for key in ("execution", "collapsed", "scrolled"):
                if key in md:
                    md.pop(key, None)
                    changed = True
    if changed:
        nbformat.write(nb, nb_path)
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clear outputs from all notebooks under a root directory.")
    parser.add_argument("--root", default="code", help="Root directory to search (default: code)")
    parser.add_argument("--pattern", default=None, help="Regex to include only matching paths")
    parser.add_argument("--exclude", default=None, help="Regex to exclude matching paths")
    parser.add_argument("--dry-run", action="store_true", help="List notebooks without modifying")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N notebooks (0 = no limit)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    notebooks = filter_paths(find_notebooks(root), args.pattern, args.exclude)

    print(f"Discovered {len(notebooks)} notebooks under {root.resolve()}")
    for i, p in enumerate(notebooks, 1):
        print(f"  [{i}] {p}")

    if args.dry_run:
        return 0

    changed_count = 0
    processed = 0
    for p in notebooks:
        print(f"\n=== Clearing outputs: {p} ===")
        if clear_outputs(p):
            print(f"UPDATED: {p}")
            changed_count += 1
        else:
            print(f"UNCHANGED: {p}")
        processed += 1
        if args.limit and processed >= args.limit:
            break

    print("\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Updated:   {changed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
