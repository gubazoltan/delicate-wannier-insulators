#!/usr/bin/env python3
import argparse
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell


def _detect_repo_root(nb_dir: str) -> str:
    """Best-effort detection of repository root.

    Preference order:
    1) If path contains a 'code' folder, use the parent of that folder as repo root.
    2) Else, look upwards for a directory containing a README.md or LICENSE.
    3) Fallback to nb_dir's parent.
    """
    norm = os.path.normpath(nb_dir)
    parts = norm.split(os.sep)
    if "code" in parts:
        idx = parts.index("code")
        if idx > 0:
            return os.sep.join(parts[:idx]) or os.path.sep
    # Heuristic: look upwards for README.md or LICENSE
    cur = norm
    for _ in range(5):  # don't traverse indefinitely
        if os.path.exists(os.path.join(cur, "README.md")) or os.path.exists(os.path.join(cur, "LICENSE")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(norm) or norm


def run_notebook(
    path: str,
    timeout: int = 600,
    kernel: str = "python3",
    save_figures: bool = False,
    figdir: str | None = None,
    figfmt: str = "png",
    figdpi: int = 200,
) -> None:
    """
    Execute a Jupyter notebook in-place using nbclient.

    Parameters:
    - path: Path to the .ipynb file
    - timeout: Per-cell execution timeout (seconds)
    - kernel: Jupyter kernel name to use
    """
    # Use a non-interactive backend for headless environments
    os.environ.setdefault("MPLBACKEND", "Agg")

    nb_path = os.path.abspath(path)
    nb_dir = os.path.dirname(nb_path) or os.getcwd()
    cwd = os.getcwd()
    try:
        # Ensure relative paths inside the notebook resolve relative to the notebook's folder
        os.chdir(nb_dir)
        nb = nbformat.read(nb_path, as_version=4)

        # Optional autosave of figures by monkeypatching plt.show via a temporary prelude cell
        inserted_prelude = False
        if save_figures:
            # Prepare environment for the prelude
            os.environ["DWI_SAVEFIG"] = "1"
            os.environ["DWI_NOTEBOOK_STEM"] = os.path.splitext(os.path.basename(nb_path))[0]
            repo_root = _detect_repo_root(nb_dir)
            # Determine the group name (first directory under 'code')
            norm = os.path.normpath(nb_dir)
            parts = norm.split(os.sep)
            group = os.path.basename(nb_dir)
            if "code" in parts:
                idx = parts.index("code")
                if len(parts) > idx + 1:
                    group = parts[idx + 1]

            if figdir is None:
                # Default: <repo>/figures/<group>
                figdir_path = os.path.join(repo_root, "figures", group)
            else:
                # Absolute path used as-is; relative path resolved against repository root
                figdir_path = figdir if os.path.isabs(figdir) else os.path.join(repo_root, figdir)
            # Ensure absolute path for the prelude so it doesn't depend on CWD nuances
            figdir_path = os.path.abspath(figdir_path)
            os.environ["DWI_FIGDIR"] = figdir_path
            os.environ["DWI_FIGFMT"] = figfmt
            os.environ["DWI_FIGDPI"] = str(figdpi)

            prelude = (
                "# DWI_AUTOSAVE_PRELUDE\n"
                "import os, pathlib\n"
                "import matplotlib.pyplot as plt\n"
                "if os.environ.get('DWI_SAVEFIG','0') == '1':\n"
                "    figdir = os.environ.get('DWI_FIGDIR', 'figures')\n"
                "    fmt = os.environ.get('DWI_FIGFMT', 'png')\n"
                "    dpi = int(os.environ.get('DWI_FIGDPI', '200'))\n"
                "    nb_stem = os.environ.get('DWI_NOTEBOOK_STEM','figure')\n"
                "    pathlib.Path(figdir).mkdir(parents=True, exist_ok=True)\n"
                "    _dwi_fig_counter = {'n': 0}\n"
                "    _orig_show = plt.show\n"
                "    def _autosave_show(*args, **kwargs):\n"
                "        _dwi_fig_counter['n'] += 1\n"
                "        fname = f\"{nb_stem}_{_dwi_fig_counter['n']:02d}.{fmt}\"\n"
                "        path = str(pathlib.Path(figdir) / fname)\n"
                "        try:\n"
                "            plt.gcf().savefig(path, dpi=dpi, format=fmt, bbox_inches='tight')\n"
                "        except Exception as e:\n"
                "            print(f'[autosave] failed: {e}')\n"
                "        return _orig_show(*args, **kwargs)\n"
                "    plt.show = _autosave_show\n"
            )
            nb.cells.insert(0, new_code_cell(prelude))
            inserted_prelude = True

        client = NotebookClient(nb, timeout=timeout, kernel_name=kernel)
        client.execute()

        # Remove the temporary prelude cell before saving, if we inserted it
        if inserted_prelude and nb.cells and isinstance(nb.cells[0], dict):
            try:
                src = nb.cells[0].get("source", "")
                if isinstance(src, str) and "DWI_AUTOSAVE_PRELUDE" in src:
                    nb.cells.pop(0)
            except Exception:
                # Even if removal fails, proceed to write
                pass

        nbformat.write(nb, nb_path)
    finally:
        os.chdir(cwd)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute one or more notebooks in place.")
    parser.add_argument("paths", nargs="+", help="Notebook file paths (.ipynb)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds (default: 600)")
    parser.add_argument("--kernel", type=str, default="python3", help="Kernel name (default: python3)")
    parser.add_argument("--save-figures", action="store_true", help="Autosave figures on plt.show() into a figures folder")
    parser.add_argument("--figdir", type=str, default=None, help="Directory to save figures (default: ./figures next to the notebook)")
    parser.add_argument("--figfmt", type=str, default="png", help="Figure format (default: png)")
    parser.add_argument("--figdpi", type=int, default=200, help="Figure DPI (default: 200)")
    args = parser.parse_args(argv)

    for path in args.paths:
        run_notebook(
            path,
            timeout=args.timeout,
            kernel=args.kernel,
            save_figures=args.save_figures,
            figdir=args.figdir,
            figfmt=args.figfmt,
            figdpi=args.figdpi,
        )


if __name__ == "__main__":
    raise SystemExit(main())
