# SEM-Device-UQ

Implementation of **Uncertainty Propagation from SEM Boundaries to MOSFET Electrical Characteristics**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data

The bundled problem uses `analytic_mos`: geometry in nm and a square-law/subthreshold current model in amperes on the JSON voltage grid. Boundary draws, parameter bounds, chain budget and random seed are defined in that file. Custom adapter paths resolve relative to the input JSON.

## Run

```bash
sem-uncertainty execute --input examples/analytic_posterior.json --output runs/full
```

## Outputs

`runs/full/electrical.json` stores electrical summaries and chain diagnostics; `posterior.npz` contains parameter chains and I-V draws. Input and output paths are controlled by `--input` and `--output`.
