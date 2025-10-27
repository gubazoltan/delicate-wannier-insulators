# Delicate Wannier insulators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17456830.svg)](https://doi.org/10.5281/zenodo.17456830) 
[![arXiv](https://img.shields.io/badge/arXiv-2506.05179-b31b1b.svg)](https://arxiv.org/abs/2506.05179)

Code and data accompanying the paper:

- Delicate Wannier insulators — Zoltán Guba, Aris Alexandradinata, Tomáš Bzdušek
- arXiv: https://arxiv.org/abs/2506.05179 — DOI: https://doi.org/10.48550/arXiv.2506.05179

This repository contains Python code and notebooks to generate spectra, Wannier bands, and phase diagrams for the models in the paper, along with representative datasets used to plot the figures.

If you use this code or data, please cite the paper (see Citation below).

## Requirements

- Python 3.10+ (tested on Windows 10/11; should also work on Linux/macOS)
- Packages: numpy, scipy, sympy, matplotlib
- One of the following for notebooks:
	- Interactive: jupyterlab (or notebook)
	- Headless/terminal: nbclient and nbformat
	- Kernel (for either mode): ipykernel (usually present; install if missing)

Quick setup with conda (recommended):

```powershell
# Create and activate an environment (Windows PowerShell)
conda create -n dwi python=3.10 -y
conda activate dwi

# Interactive notebooks
pip install numpy scipy sympy matplotlib jupyterlab ipykernel

# For headless runs (optional)
pip install nbclient nbformat
```

Or with venv + pip:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Interactive notebooks
pip install numpy scipy sympy matplotlib jupyterlab ipykernel

# For headless runs (optional)
pip install nbclient nbformat
```

```bash
# Linux/macOS (bash)
python3 -m venv .venv
source .venv/bin/activate

# Interactive notebooks
pip install numpy scipy sympy matplotlib jupyterlab ipykernel

# For headless runs (optional)
pip install nbclient nbformat
```

## Data

Use the precomputed data included in the repository:

- 1D delicate chain phase diagrams: `code/delicate_chain/phasediagdata/`
- Layered RTP phase diagrams: `code/layered_rtp/phasediagdata/`

No data generation is required to run the notebooks.

## Notebooks

Launch the notebooks to reproduce spectra and figures:

```powershell
cd code
jupyter lab
```

```bash
cd code
jupyter lab
```

- `dartboard/`: `overview.ipynb`, `stacked_2d_spectra.ipynb`, `wannier_chern.ipynb`, `wire_geometry.ipynb`
- `delicate_chain/`: `overview.ipynb`, `stacked_1d_spectra.ipynb`, `wannier_winding.ipynb`, `phase_diagram_alphabeta.ipynb`, `phase_diagram_layered.ipynb`
- `layered_rtp/`: `energy_and_wannier.ipynb`, `wire_geometry.ipynb`, `phase_diagram_plotting.ipynb`
- `rtp/`: `wilson_spectrum.ipynb`, `projected_position_spectrum.ipynb`

Tip: Some cells can be computationally intensive (SymPy algebra and dense diagonalizations). Consider reducing grid sizes (`Nx, Ny, Nz`) for quick checks, then scale up.

### Run notebooks from the terminal (headless)

If you prefer running notebooks non-interactively (e.g., for validation before archiving), use the helpers in `code/utils/`:

Run a single notebook in place:

```powershell
# Use a non-interactive backend
$env:MPLBACKEND = 'Agg'

py -3 "code\utils\execute_notebook.py" "code\rtp\projected_position_spectrum.ipynb"
```

```bash
# Use a non-interactive backend
export MPLBACKEND=Agg

python3 code/utils/execute_notebook.py code/rtp/projected_position_spectrum.ipynb
```

Run all notebooks under `code/` with filters and safety flags:

```powershell
$env:MPLBACKEND = 'Agg'
py -3 "code\utils\run_all_notebooks.py" --root "code"            # run everything
py -3 "code\utils\run_all_notebooks.py" --root "code" --dry-run   # list only
py -3 "code\utils\run_all_notebooks.py" --root "code" --pattern "rtp"   # subset
py -3 "code\utils\run_all_notebooks.py" --root "code" --limit 2 --fail-fast # smoke test
```

```bash
export MPLBACKEND=Agg
python3 code/utils/run_all_notebooks.py --root code            # run everything
python3 code/utils/run_all_notebooks.py --root code --dry-run  # list only
python3 code/utils/run_all_notebooks.py --root code --pattern "rtp"   # subset
python3 code/utils/run_all_notebooks.py --root code --limit 2 --fail-fast # smoke test
```

These helpers avoid relying on a `jupyter` command being on PATH and use `nbclient`/`nbformat` under the hood. They execute notebooks from each notebook’s own directory (so relative paths like `./phasediagdata/...` work), update them in place, and print a success/failure summary.

### Reproducibility extras: export figures and clear outputs

Export all figures automatically (repo-level `figures/<group>/...`)

```powershell
$env:MPLBACKEND = 'Agg'
py -3 "code\utils\run_all_notebooks.py" --root "code" --save-figures
```

```bash
export MPLBACKEND=Agg
python3 code/utils/run_all_notebooks.py --root code --save-figures
```

- Files are saved to `figures/<group>/NotebookName_01.png`, where `<group>` is one of `dartboard`, `delicate_chain`, `layered_rtp`, `rtp`.
- Customize output:

```powershell
py -3 "code\utils\run_all_notebooks.py" --root "code" --save-figures --figfmt "pdf" --figdpi 300
# or save under a different base folder at the repo root
py -3 "code\utils\run_all_notebooks.py" --root "code" --save-figures --figdir "figures_pub"
```

```bash
python3 code/utils/run_all_notebooks.py --root code --save-figures --figfmt pdf --figdpi 300
# or save under a different base folder at the repo root
python3 code/utils/run_all_notebooks.py --root code --save-figures --figdir figures_pub
```

Export for a single notebook:

```powershell
$env:MPLBACKEND = 'Agg'
py -3 "code\utils\execute_notebook.py" "code\rtp\projected_position_spectrum.ipynb" --save-figures
```

```bash
export MPLBACKEND=Agg
python3 code/utils/execute_notebook.py code/rtp/projected_position_spectrum.ipynb --save-figures
```

Clear notebook outputs (reduce size before archiving)

```powershell
py -3 "code\utils\clear_notebook_outputs.py" --root "code"
```

```bash
python3 code/utils/clear_notebook_outputs.py --root code
```

Options:
- List without modifying: `--dry-run`
- Process a subset: `--pattern "rtp"` or exclude: `--exclude "layered_rtp"`

Note: If a notebook already calls `plt.savefig(...)`, you may see both the manual saves and autosaved images; keep one approach if you want to avoid duplicates.

### Ensuring Linux/macOS runs smoothly

- Use python3 and a virtual environment:
	- Create venv: `python3 -m venv .venv` then `source .venv/bin/activate`
	- Install packages listed above with `pip install ...`
- Set the non-interactive backend when running headless: `export MPLBACKEND=Agg`
- Prefer forward slashes in paths on bash (already used in the examples): `code/utils/...`
- Our helpers change into each notebook's folder before execution, so relative data paths like `./phasediagdata/...` work on all OSes.
- If you hit a permission error when invoking scripts directly, call them with `python3 path/to/script.py` as shown above.
- If you use a non-default kernel name, pass `--kernel your-kernel-name`.

WSL tips:
- Create your virtual environment on the Linux filesystem for speed (e.g., `~/.venvs/dwi`) and activate it while working in the repo under `/mnt/c/...`.
- On Ubuntu/WSL you may see a PEP 668 “externally-managed-environment” error if you try to install into the system Python. Use a venv (or conda), and prefer `python -m pip install ...` to ensure you’re using the venv’s pip.

One-shot reproducible run (headless):

```bash
export MPLBACKEND=Agg
python3 code/utils/run_all_notebooks.py --root code --save-figures && \
python3 code/utils/clear_notebook_outputs.py --root code
```

## How to cite

- Software: Please cite the archived software via the Zenodo concept DOI: https://doi.org/10.5281/zenodo.17456830 (click the DOI badge above for BibTeX/APA/other formats).

- Paper: If you use this repository for results or figures from the manuscript, please also cite the paper. See the arXiv record for citation formats (use arXiv’s BibTeX export to preserve accents): https://arxiv.org/abs/2506.05179

## License

Code and notebooks are licensed under the MIT License. See `LICENSE` for details.

