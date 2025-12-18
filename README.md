# Delicate Wannier insulators

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17456830-blue?logo=zenodo)](https://doi.org/10.5281/zenodo.17456830) [![arXiv](https://img.shields.io/badge/arXiv-2506.05179-b31b1b.svg)](https://arxiv.org/abs/2506.05179)

Code and data accompanying the paper:

- Delicate Wannier insulators — Zoltán Guba, Aris Alexandradinata, Tomáš Bzdušek
- arXiv: https://arxiv.org/abs/2506.05179 — DOI: https://doi.org/10.48550/arXiv.2506.05179

This repository contains Python code and notebooks to generate spectra, Wannier bands, and phase diagrams for the models in the paper, along with representative datasets used to plot the figures.

If you use this code or data, please cite the paper (see Citation below).

## Requirements

- Python 3.10+ (tested on Windows 10/11; should also work on Linux/macOS)
- Packages: numpy, scipy, sympy, matplotlib, jupyterlab, ipykernel
- Optional (for headless notebook execution): nbclient, nbformat

**Setup:**

```bash
# Create and activate environment (conda)
conda create -n dwi python=3.10 -y
conda activate dwi

# Or use venv (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Or use venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install numpy scipy sympy matplotlib jupyterlab ipykernel

# Optional: for headless runs
pip install nbclient nbformat
```

## Data

Use the precomputed data included in the repository:

- 1D delicate chain phase diagrams: `code/delicate_chain/phasediagdata/`
- Layered RTP phase diagrams: `code/layered_rtp/phasediagdata/`

No data generation is required to run the notebooks.

## Notebooks

Launch JupyterLab to reproduce spectra and figures:

```bash
cd code && jupyter lab
```

**Available notebooks:**
- `dartboard/`: overview, stacked_2d_spectra, wannier_chern, wire_geometry
- `delicate_chain/`: overview, stacked_1d_spectra, wannier_winding, phase_diagram_alphabeta, phase_diagram_layered
- `layered_rtp/`: energy_and_wannier, wire_geometry, phase_diagram_plotting
- `rtp/`: wilson_spectrum, projected_position_spectrum

**Tip:** Some cells can be computationally intensive. Consider reducing grid sizes (`Nx, Ny, Nz`) for quick checks.

### Headless execution

For non-interactive execution:

```bash
# Set non-interactive backend (Linux/macOS)
export MPLBACKEND=Agg

# Windows PowerShell: $env:MPLBACKEND = 'Agg'

# Run a single notebook
python3 code/utils/execute_notebook.py code/rtp/projected_position_spectrum.ipynb

# Run all notebooks
python3 code/utils/run_all_notebooks.py --root code

# Options: --dry-run, --pattern "rtp", --limit 2, --fail-fast
```

### Export figures

```bash
# Export all figures to figures/<group>/
python3 code/utils/run_all_notebooks.py --root code --save-figures

# Customize: --figfmt pdf --figdpi 300 --figdir figures_pub

# Export from single notebook
python3 code/utils/execute_notebook.py code/rtp/projected_position_spectrum.ipynb --save-figures
```

### Clear outputs

```bash
# Clear notebook outputs (reduce size before archiving)
python3 code/utils/clear_notebook_outputs.py --root code

# Options: --dry-run, --pattern "rtp", --exclude "layered_rtp"
```

## How to cite

- Software: Please cite the archived software via the Zenodo concept DOI: https://doi.org/10.5281/zenodo.17456830 (click the DOI badge above for BibTeX/APA/other formats).

- Paper: If you use this repository for results or figures from the manuscript, please also cite the paper. See the arXiv record for citation formats (use arXiv's BibTeX export to preserve accents): https://arxiv.org/abs/2506.05179

## License

Code and notebooks are licensed under the MIT License. See `LICENSE` for details.
