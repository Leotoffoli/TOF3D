# TOF3D

TOF3D is an open source Python code for 3D density based topology optimization in single precision on one consumer
GPU. Its GPU kernels use [Taichi](https://www.taichi-lang.org/). It solves four problem classes on structured grids of
eight node hexahedra:

- stress constrained compliance minimization (relaxed von Mises stress aggregated by the P-norm);
- robust compliance minimization against manufacturing errors (eroded, intermediate and dilated designs);
- compliant mechanism design (force inverter, min-max over the three projected designs);
- porous infill by local volume constraints.

The largest runs have 33.5 million elements (101.6 million degrees of freedom) on an NVIDIA GeForce RTX 3060 with 12 GB.

The preprint *An open source Python code for three-dimensional topology optimization in single precision on a consumer
GPU* (L. T. Pedrini, R. G. M. Andrade and R. M. Ferro), [arXiv:2611.12027](https://arxiv.org/abs/2611.12027),
describes the method and the results.

## This repository and the Zenodo record

The reference archive is the Zenodo record [10.5281/zenodo.24050327](https://doi.org/10.5281/zenodo.24050327). It holds
the code, the protocol fixed before the runs, the analysis and figure scripts, the logs and every result of the article.

This repository holds the code only. The tag `v0.1.0` is identical, file by file, to the archive `TOF3D-0.1-code.zip`
of that record. The README of the record, with the list of its archives and the source of each table and figure, is in
[`docs/ZENODO-README.md`](docs/ZENODO-README.md).

## What is in the code

| File | Content |
|---|---|
| `tof3d/tof3d_core.py` | Matrix-free finite element operator on the GPU, density filter, Heaviside projection, responses and adjoint sensitivities |
| `tof3d/tof3d_mg.py` | Geometric multigrid preconditioner for the conjugate gradient: Galerkin coarse operators computed by probing, damped nodal block Jacobi smoother, dense Cholesky factorization in float64 on the CPU at the coarsest level |
| `tof3d/tof3d_mma.py` | Method of moving asymptotes, written from Svanberg (1987, 2007), on the CPU |
| `tof3d/tof3d_mma_gpu.py` | The same method with the passes of the dual subproblem on the GPU |
| `tof3d/tof3d_filters.py` | Density filter by FFT |
| `tof3d/tof3d_examples.py` | The eight cases of the article (`L_comp`, `L_tens`, `C_std`, `C_rob`, `M_std`, `M_rob`, `I_inf`, `I_ref`) |
| `tof3d/tests/` | Five checks: gradients against finite differences, multigrid against the exact solution, MMA against SciPy, MMA on the GPU against the CPU version, FFT filter against the direct filter |

## Requirements

The optimizations of the article ran on Ubuntu 24.04 with Python 3.12.12, NumPy 2.3.5, SciPy 1.16.3 and Taichi 1.7.4,
on an NVIDIA RTX 3060 with 12 GB (driver 595.91.07, CUDA 13.2).

These package versions need Python 3.11, 3.12 or 3.13: NumPy 2.3.5 and SciPy 1.16.3 require Python 3.11 or newer, and
Taichi 1.7.4 has no packages for Python 3.14. The Python 3.9 that comes with macOS is too old, and with it `pip` stops
with "No matching distribution found for numpy==2.3.5". Install a newer Python first, for example
`brew install python@3.12` on macOS. A virtual environment keeps the packages apart from the rest of the system:

```bash
git clone https://github.com/Leotoffoli/TOF3D.git
cd TOF3D
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5 scipy==1.16.3 taichi==1.7.4
```

The code is written for an NVIDIA GPU with CUDA. `tof3d/tof3d_examples.py` (line 188) and
`tof3d/tests/check_mma_gpu.py` start Taichi with `C.init("cuda", mem_gb=11.0)`, which reserves about 11 GB of GPU
memory; on a card with less memory, lower `mem_gb` in that line of your copy. `pip` installs Taichi but not the GPU
driver. Without a CUDA GPU, Taichi prints the warning `Arch=[<Arch.cuda: 3>] is not supported, falling back to CPU`
and runs on the CPU: the quick start below took 49 s on an Apple M4 computer, but the full runs of the article are much slower on a CPU.

## Quick start

A short run (20 design iterations) of the reference cantilever on a 96 × 48 × 48 grid, written to
`tof3d/results/C_std_smoke/` without checkpoints:

```bash
cd tof3d
python tof3d_examples.py C_std --size 96 --iters 20 --smoke
```

`--size N` sets the largest side of the grid. The other sides and the lengths of each case follow from it; the
thickness of the L-bracket is rounded to an even number of elements near N/3 that the multigrid accepts. A full run of a
case (300 design iterations, with a checkpoint every 50 iterations):

```bash
python tof3d_examples.py CASE --size SIZE
```

Two cases read the final measures of another case from the same results folder, so run that case first, at the same
size and without `--smoke`: `L_comp` before `L_tens`, and `I_inf` before `I_ref`. On the grid of 512 × 256 × 256 elements, the force inverters `M_std` and
`M_rob` did not form a mechanism with the regularization of the released code (`RAA0 = 1e-5`); the article and
[`docs/ZENODO-README.md`](docs/ZENODO-README.md) give the scaled value of Amendment 11.

The queue of the article runs, with the size and multigrid policy of each, is in the Zenodo record
(`tof3d/runs/fila_e8.txt` in `scripts-and-records.zip`).

## Checks

From the root of the repository:

```bash
bash tof3d/tests/run_all.sh python
```

Four checks run on the CPU: `check_gradients.py cpu`, `check_mg.py exato cpu` (it needs `TOF3D_FP64=1`, which
`run_all.sh` sets), `check_mma.py` and `check_filters.py`.
Two need the CUDA GPU: `check_mg.py iter cuda` and `check_mma_gpu.py`. Each check writes a JSON file next to it and must
report that it passed (`"passou": true`), except `check_mg_iter.json`, which compares iteration counts and times and has
no pass or fail. The JSON files committed here come from the code used for the runs of the article; running the checks
overwrites them.

## Output files

The code writes its outputs under Portuguese names. Each run folder holds:

| File | Content |
|---|---|
| `desenho.npz` | Design: `x` (design variables), `xt` (filtered), `xp_i` (intermediate projected design, `float16`; the robust cases `C_rob` and `M_rob` also hold the eroded `xp_e` and the dilated `xp_d`), `act` (active elements), `solid` (elements fixed as solid), `beta` |
| `historico.json` | History, one record per design iteration: `it`, `beta`, `vol`, `g` (constraint values), `change`, `cg` (conjugate gradient iterations), `mma_newton`, `mma_passagens`, `t` (seconds), and the measures of the case: `c` and `f0`; `PN`, `ratio_max` and `ck` in `L_tens`; `local_pnorm` and `local_max` in `I_inf`; `c_e`, `f0` and `vd_star` in `C_rob`; `u_out`, `vd_star` and `u_scale` in `M_std` and `M_rob` |
| `metricas.json` | Final measures, times and solver statistics |
| `checkpoint.npz` | Design variables every 50 iterations (full runs only) |
| `sigma.npy` | Relaxed von Mises stress of the L-bracket |

Portuguese words in the names of outputs and checks: `alertas` warnings, `cinza` grey measure (mean of
4 xp_i (1 − xp_i) over the active elements), `desenho` design, `erro` error, `exato` exact, `historico` history,
`iteracoes` iterations, `metricas` measures, `montagem(ns)` multigrid setup(s), `multigrelha` multigrid, `nivel`/`niveis` level(s), `passagens` passes,
`passou` passed, `referencia` reference, `residuo` residual, `restricao` constraint, `solucoes` linear solves,
`tamanho` size, `tempo` time, `tensao` stress.

## Precision

The optimization runs keep fields in single precision (float32) on the GPU and accumulate sums in double precision
(float64). The element products use displacement differences inside each element, so rigid translations never pass
through the rounded single precision element matrix. In the eight cases, the MMA stores its vectors in float64 up to
2 million design variables and in float32 above. The gradient checks and the exact multigrid check run in float64
throughout (`TOF3D_FP64=1`).

## Status

TOF3D is research code written during a doctorate. Version 0.1 (tag `v0.1.0`) is the code of the submitted
manuscript and of the Zenodo record 10.5281/zenodo.24050327, and that tag does not move. Issues are not open for now.

## How to cite

Use "Cite this repository" on this page, which reads [`CITATION.cff`](CITATION.cff), or cite the Zenodo record:

> L. T. Pedrini, R. G. M. Andrade and R. M. Ferro. TOF3D, version 0.1. Zenodo, 2026.
> https://doi.org/10.5281/zenodo.24050327

For the method, cite the preprint [arXiv:2611.12027](https://arxiv.org/abs/2611.12027).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Documentation, data, results and figures: CC BY 4.0, as in the Zenodo record.
Third party tools: Taichi (Apache 2.0), NumPy and SciPy (BSD 3-Clause).
