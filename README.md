# SNPE in Simulation-Based Bayesian Inference

Code and numerical experiments for the paper "Sequential Neural Posterior Estimation in Simulation-Based Bayesian Inference".

## Repository structure

Each example directory contains five self-contained scripts, one per method. Every script can be run independently (e.g., as separate SLURM jobs).

```
example1_linear_regression/     Example 1: Bayesian linear regression
example2_nonlinear_gaussian/    Example 2: nonlinear regression with Gaussian errors
example3_nonlinear_student_t/   Example 3: nonlinear regression with Student-t errors
```

Methods in each directory:

| Script | Method |
|---|---|
| `NPE.py` | Neural posterior estimation (non-sequential baseline) |
| `SNPE-B(plain).py` | SNPE-B with importance-weighted loss |
| `SNPE-C.py` | SNPE-C / APT with atomic proposals |
| `TSNPE-P.py` (Example 1: `TSNPE(ParameterSpace).py`) | Truncated SNPE, parameter-space truncation |
| `TSNPE-D.py` (Example 1: `TSNPE(DataSpace).py`) | Truncated SNPE, data-space truncation |

Examples 2 and 3 additionally contain the Stan models (`exp2term.stan`, `exp3term.stan`) used to obtain reference MCMC posteriors.

## Installation

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Examples 2 and 3 use [CmdStanPy](https://mc-stan.org/cmdstanpy/) to run MCMC as the reference posterior. Besides the Python package, this requires the CmdStan toolchain (tested with CmdStan 2.35.0), which can be installed once via:

```bash
python -m cmdstanpy.install_cmdstan
```

Example 1 uses a conjugate model and does not need Stan.

The Stan models are compiled automatically by CmdStanPy the first time a script is run (this takes a minute or two); the compiled executable is placed next to the `.stan` file and reused on subsequent runs.

## Running

Each script is standalone:

```bash
cd example2_nonlinear_gaussian
python NPE.py
```

Stan model files are resolved relative to the script location, so the scripts can be launched from any working directory. Outputs (per-round C2ST accuracies, KL divergences, mixture parameters, and plots) are written to the directory given by the `OUTDIR` environment variable (default: the current working directory).

## Author

Renjie Peng
