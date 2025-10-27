#!/usr/bin/env python3
import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

import nbformat
from nbclient import NotebookClient

# Attempt to load the single-notebook runner (execute_notebook.py) even when this
# script is run directly (i.e., utils is not a Python package). This enables
# --save-figures and friends by delegating to the shared implementation.
run_one = None
try:
    # First try a relative import (works if utils is a package)
    from .execute_notebook import run_notebook as _run_one  # type: ignore
    run_one = _run_one
except Exception:
    # Fallback: load by file path
    try:
        import importlib.util
        here = Path(__file__).resolve().parent
        mod_path = here / "execute_notebook.py"
        spec = importlib.util.spec_from_file_location("execute_notebook", str(mod_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            run_one = getattr(module, "run_notebook", None)
    except Exception:
        run_one = None


def run_notebook(
    path: str,
    timeout: int = 600,
    kernel: str = "python3",
    save_figures: bool = False,
    figdir: str | None = None,
    figfmt: str = "png",
    figdpi: int = 200,
) -> None:
    # Prefer the shared runner (supports autosave)
    if run_one is not None:
        return run_one(
            path,
            timeout=timeout,
            kernel=kernel,
            save_figures=save_figures,
            figdir=figdir,
            figfmt=figfmt,
            figdpi=figdpi,
        )

    # Fallback minimal runner
    os.environ.setdefault("MPLBACKEND", "Agg")
    nb_path = os.path.abspath(path)
    nb_dir = os.path.dirname(nb_path) or os.getcwd()
    cwd = os.getcwd()
    try:
        os.chdir(nb_dir)
        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=timeout, kernel_name=kernel)
        client.execute()
        nbformat.write(nb, nb_path)
    finally:
        os.chdir(cwd)


def find_notebooks(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*.ipynb"):
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run all notebooks in-place under a root directory.")
    parser.add_argument("--root", default="code", help="Root directory to search (default: code)")
    parser.add_argument("--pattern", default=None, help="Regex to include only matching paths")
    parser.add_argument("--exclude", default=None, help="Regex to exclude matching paths")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds (default: 600)")
    parser.add_argument("--kernel", default="python3", help="Kernel name (default: python3)")
    parser.add_argument("--dry-run", action="store_true", help="List notebooks without executing")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--limit", type=int, default=0, help="Execute at most N notebooks (0 = no limit)")
    # Figure autosave options (delegated to execute_notebook)
    parser.add_argument("--save-figures", action="store_true", help="Autosave figures on plt.show() into a figures folder")
    parser.add_argument("--figdir", type=str, default=None, help="Directory to save figures (default: ./figures next to each notebook)")
    parser.add_argument("--figfmt", type=str, default="png", help="Figure format (default: png)")
    parser.add_argument("--figdpi", type=int, default=200, help="Figure DPI (default: 200)")
    args = parser.parse_args(argv)

    # Windows asyncio policy to avoid ZMQ warnings and improve compatibility
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    root = Path(args.root)
    notebooks = find_notebooks(root)
    notebooks = filter_paths(notebooks, args.pattern, args.exclude)

    print(f"Discovered {len(notebooks)} notebooks under {root.resolve()}")
    for i, p in enumerate(notebooks, 1):
        print(f"  [{i}] {p}")

    if args.dry_run:
        return 0

    ran = 0
    failed: list[tuple[Path, str]] = []

    for p in notebooks:
        print(f"\n=== Executing: {p} ===")
        try:
            run_notebook(
                str(p),
                timeout=args.timeout,
                kernel=args.kernel,
                save_figures=args.save_figures,
                figdir=args.figdir,
                figfmt=args.figfmt,
                figdpi=args.figdpi,
            )
            print(f"SUCCESS: {p}")
        except Exception as e:
            failed.append((p, str(e)))
            print(f"FAILED: {p}\n{e}")
            if args.fail_fast:
                break
        ran += 1
        if args.limit and ran >= args.limit:
            break

    print("\nSummary:")
    print(f"  Executed: {ran}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for p, msg in failed:
            print(f"    - {p}: {msg.splitlines()[-1] if msg else 'Error'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
