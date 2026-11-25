# TOF3D 0.1: code and data

TOF3D is an open source Python code for three-dimensional density-based topology optimization in single precision on one
consumer GPU. Its GPU kernels are written in Taichi. The authors wrote it for the article "An open source Python code for
three-dimensional topology optimization in single precision on a consumer GPU" (L. T. Pedrini, R. G. M. Andrade and
R. M. Ferro, prepared for Structural and Multidisciplinary Optimization). It solves four classes of problems on
structured grids of eight-node hexahedra:

- stress constrained compliance minimization (relaxed von Mises stress aggregated by the P-norm);
- robust compliance minimization against manufacturing errors (eroded, intermediate and dilated designs);
- compliant mechanism design (force inverter, min-max over the three projected designs);
- porous infill by local volume constraints.

The largest runs have 33.5 million elements (101.6 million degrees of freedom) on an NVIDIA GeForce RTX 3060 with 12 GB.

Zenodo record: https://doi.org/10.5281/zenodo.24050327

## What is in this record

Unpack every archive in one folder. Together they form the folder `TOF3D-0.1/`.

| File | Content | Folders |
|---|---|---|
| `TOF3D-0.1-code.zip` | The six modules and the checks, with their outputs | `tof3d/*.py`, `tof3d/tests/` |
| `protocol.zip` | The protocol fixed before the runs, its dated amendments and the result notes, in Portuguese, with a summary in English | `protocol/` |
| `scripts-and-records.zip` | Analysis and figure scripts, queue of runs, logs, GPU memory record, solver bench, checks of Amendment 8, comparison with the code of Träff et al. | `analysis/` (except `F2-mapdl`), `figures/`, `tof3d/runs/`, `gpu-comparison/` |
| `results-96-256.zip` | Final designs, histories and metrics on the coarse and fine meshes, the secondary analyses and the double precision runs | `tof3d/results_e8/{96,256,emenda4}`, `tof3d/results_e8_fp64/` |
| `results-512.zip` | The same for the six cases on 512 × 256 × 256 elements | `tof3d/results_e8/512/` |
| `results-512-amendment11.zip` | Final designs, histories, metrics and logs of the two inverters on 512 × 256 × 256 elements with the regularization of the MMA scaled (Amendment 11) | `analysis/emenda11/r512/` |
| `ansys-crosscheck.zip` | Input files of the cross-check with Ansys Mechanical APDL, the results extracted from it and its scripts | `analysis/F2-mapdl/` |
| `SHA256SUMS.txt` | SHA-256 of every file of `TOF3D-0.1/` | |
| `PATH-CHANGES.tsv` | Files whose text differs from the working copies, with both checksums (see "Differences from the working copies") | |

### The code

| File | Content |
|---|---|
| `tof3d_core.py` | Matrix-free finite element operator on the GPU (Taichi), density filter, Heaviside projection, responses and adjoint sensitivities |
| `tof3d_mg.py` | Geometric multigrid preconditioner for the conjugate gradient: Galerkin coarse operators computed by probing, damped nodal block Jacobi smoother |
| `tof3d_mma.py` | Method of moving asymptotes, written from Svanberg (1987, 2007), on the CPU |
| `tof3d_mma_gpu.py` | The same method with the dual subproblem solved on the GPU |
| `tof3d_filters.py` | Density filter by FFT |
| `tof3d_examples.py` | The eight cases of the article (`L_comp`, `L_tens`, `C_std`, `C_rob`, `M_std`, `M_rob`, `I_inf`, `I_ref`) |
| `tests/` | Five checks: gradients against finite differences, multigrid against the exact solution, MMA against SciPy, MMA on the GPU against the CPU version, FFT filter against the direct filter |

## Requirements

The optimizations ran on Ubuntu 24.04 with Python 3.12.12, NumPy 2.3.5, SciPy 1.16.3 and Taichi 1.7.4, on an NVIDIA RTX 3060
(driver 595.91.07, CUDA 13.2). The checks marked "cpu" run without a GPU.

```bash
python -m pip install numpy==2.3.5 scipy==1.16.3 taichi==1.7.4
```

The analysis and figure scripts read saved results and need no GPU. They ran on macOS with Python 3.9.6, NumPy 2.0.2,
SciPy 1.13.1, Matplotlib 3.9.4, scikit-image 0.24.0 and PyVista 0.46.5; the figures use the Arial font.

Scripts and queues refer to the unpacked folder by the variable `TOF3D_ROOT`:

```bash
cd TOF3D-0.1
export TOF3D_ROOT=$PWD
```

## Quick start

A short run (20 design iterations) of the reference cantilever on a 96 × 48 × 48 grid, written to
`results/C_std_smoke/` without checkpoints:

```bash
cd tof3d
python tof3d_examples.py C_std --size 96 --iters 20 --smoke
```

## Checks

```bash
bash tof3d/tests/run_all.sh python
```

Each check writes a JSON file next to it and must report that it passed (`"passou": true`), except `check_mg_iter.json`,
which compares iteration counts and times and has no pass or fail. The JSON files in this record come from the
code used for the runs of the article.

## The runs of the article

`tof3d/runs/fila_e8.txt` is the queue of the runs in the order they ran, and `tof3d/runs/corre_fila_e8.sh` runs it, one job
at a time. A line with a case, a size and a multigrid policy means:

```bash
cd tof3d
python tof3d_examples.py CASE --size SIZE --mg-policy POLICY --out results_e8/SIZE
```

The multigrid policy "design" rebuilds the coarse operators after every change of design and "iteration" once per design
iteration. The two are equivalent in the cases with one design per iteration. The queue gives the policy of each run.
Double precision runs add `TOF3D_FP64=1` and `--store f64`. Lines that start with `CMD` give the solver bench, the
secondary analyses and the comparison with the code of Träff et al.

## Where each table and figure comes from

Commands start from `TOF3D-0.1/`. Figures are written to `figures/output/`.

| Item | Saved data | Command |
|---|---|---|
| Table 4 | `tof3d/tests/check_gradients.json` | `bash tof3d/tests/run_all.sh python` |
| Table 5, after the difference form of Eq. (4) | `gpu-comparison/precisao_e8/` | `gpu-comparison/scripts/confere_precisao.py`, then `avalia_B1_e8.py` (queue line `B1_e8`) |
| Table 5, before the difference form of Eq. (4) | `analysis/candidato_emenda8/b1_antigos/`, `protocol/RESULTADO-EMENDA-8-K.md` | See Amendment 8 |
| Table 6 | `analysis/R96_e8/analise.json`, `analysis/R256_e8/analise.json` | `cd analysis; TOF3D_RES=$PWD/../tof3d/results_e8/96 TOF3D_OUT=$PWD/R96_e8 python analisa_e_desenha.py; python analisa_R256_e8.py` |
| Figs. 1, 3, 4 and 6 | `tof3d/results_e8/256/` | `python figures/figuras_artigo.py pairs` |
| Fig. 2 | `tof3d/results_e8/256/L_comp`, `L_tens` | `python figures/figuras_artigo.py stress` |
| Fig. 5 | `analysis/R96_e8/erosao_96.json`, `analysis/R256_e8/erosao_256.json` | `python figures/figuras_artigo.py erosion`; data from `analysis/avalia_erosao_e8.py` |
| Table 7 | `analysis/F2-mapdl/tabela_F2.md`, `tabela_F2.json` | `analysis/F2-mapdl/tabela_F2.py` |
| Fig. 7 | `analysis/bench_e8/` | `python figures/figuras_artigo.py scaling`; data from `analysis/bench_solver.py` |
| Table 8 | `tof3d/results_e8/{256,512}/`, `tof3d/runs/memoria_e8.csv`, `analysis/emenda11/` (rows marked 1) | `python analysis/table_time.py` |
| Fig. 8 | `gpu-comparison/tempos/ANALISE_traff.json` | `python figures/figuras_artigo.py gpu_codes`; data from `gpu-comparison/scripts/avalia_traff.py` |
| Section 3.4, inverters of Amendment 11, and Table S4, last rows | `analysis/emenda11/resultado_512.json`, `analysis/emenda11/r512/` | `TOF3D_DIR=$PWD/tof3d PY=python bash analysis/emenda11/corre_512.sh`, then `python analysis/emenda11/analisa_512.py tof3d/results_e8` |
| Section 3.4, 512 × 256 × 256 | `analysis/R512_e8/analise.json` | `cd analysis; python analisa_fila_final.py 512 R512_e8` (writes `analise_512.json`) |
| Section 3.1, double precision | `analysis/B2_e8/` | `cd analysis; python analisa_fila_final.py b2 B2_e8` |
| Fig. S1 | `tof3d/results_e8/256/I_inf`, `I_ref` | `python figures/figuras_artigo.py infill` |
| Table S1 | `protocol/AMENDMENTS-summary.md` | |
| Table S2 | `tof3d/results_e8/emenda4/*/resumo.json`, `analysis/B2_e8/analise_96_fp64.json` | `analysis/variantes_L_e8.py` (queue line `S2_e8`) |
| Fig. S2 | `tof3d/results_e8/512/`, `analysis/emenda11/r512/` (inverters) | `python figures/figuras_artigo.py 512` |
| Table S3 and the scaled radii in Sects. 3.2 and 3.4 | `analysis/emenda10/abertura_escalada.json` | `python analysis/emenda10/abertura_escalada.py tof3d/results_e8` |
| Table S4 | `analysis/emenda10/resultado_raa0.json`, `analysis/emenda10/raa0/` | `TOF3D_DIR=$PWD/tof3d PY=python bash analysis/emenda10/corre_raa0.sh`, then `python analysis/emenda10/analisa_raa0.py tof3d/results_e8` |

Tables 1, 2, 3 and 9 describe the code, the cases and the published literature. They hold no computed results.

The regularization of the MMA is fixed at `RAA0 = 1e-5` in `tof3d/tof3d_mma.py`. On 512 × 256 × 256 elements the force inverters form a mechanism only with this value divided by 151.70 (Amendment 11). `analysis/emenda10/corre_raa0.py` sets the factor at run time, so the released files do not change. Only this value was tried.

## Saved files and their names

The code writes its outputs under Portuguese names. Each run folder `CASE/` holds:

| File | Content |
|---|---|
| `desenho.npz` | Design: `x` (design variables), `xt` (filtered), `xp_i` (intermediate projected design, `float16`), `act` (active elements), `solid` (elements fixed as solid), `beta` |
| `historico.json` | History: one record per design iteration (`it`, `beta`, `vol`, `c`, `f0`, `g`, `change`, `cg`, `mma_newton`, `mma_passagens`, `t` in seconds) |
| `metricas.json` | Final measures, times and solver statistics |
| `sigma.npy` | Relaxed von Mises stress of the L-bracket |

Other Portuguese words in names and comments: `analise` analysis, `avalia` evaluate, `caso(s)` case(s), `compara` compare, `confere` check, `corre` run, `emenda` amendment, `adenda` addendum, `erosao` erosion, `estado` status, `feitos` done, `fila` queue, `memoria` memory, `modelo` model, `precisao` precision, `protocolo` protocol, `regista` record, `resultado(s)` result(s), `tabela` table, `tempos` timings, `terceiros` third party.

## Precision

The optimization runs keep fields in single precision (float32) on the GPU and accumulate sums in double precision
(float64). The MMA stores its vectors in float64 up to 2 million design variables and in float32 above. The gradient checks
run in float64 throughout (`TOF3D_FP64=1`).

## Cross-check with Ansys Mechanical APDL

Ansys Mechanical APDL is commercial software and is not part of this record. `analysis/F2-mapdl/casos/` holds the input
files written for MAPDL 2025 R1 (`modelo.inp`, compressed on the fine mesh) and the arrays extracted from each solution
(`mapdl.npz`), without the binary result files. The scripts used PyMAPDL 0.74.1.

## Comparison with the code of Träff et al. (2023)

This record does not redistribute that code. `gpu-comparison/terceiros/mudancas-build-opencl.diff` is the patch applied to
version 0.9.1, which is available at https://doi.org/10.5281/zenodo.7791871. It selects the OpenCL backend and writes
output files less often. `gpu-comparison/scripts/corre_traff_pc.sh` needs that code unpacked and patched in
`gpu-comparison/terceiros/build-opencl`, Futhark 0.21.11 in the folder `FUTHARK_BIN`, gcc and the OpenCL headers.

## Differences from the working copies

The code deposited here is the code of the runs, apart from Amendment 9 (`protocol/EMENDA-9-so-nomes.md`), which changes
names only and no calculation. The scripts, queues and logs differ from the working copies in folder names and paths, which
became relative to `TOF3D_ROOT`. `PATH-CHANGES.tsv` lists each of these files with its original and deposited SHA-256.
The checksum files inside `analysis/F2-mapdl` are those of the runs, so they do not match the files listed there.
The files in `protocol/` are copies of the frozen originals with old working names replaced; `protocol/NOTE-on-names.md`
gives the details and the SHA-256 of each original and each copy. Amendments 10 and 11 and their result notes needed no
replacement: they are identical to the originals, so the note does not list them.

## License

Code: MIT (see `LICENSE`). Data, results, figures and documentation: CC BY 4.0. Third-party tools: Taichi (Apache 2.0),
NumPy and SciPy (BSD 3-Clause).

## How to cite

See `CITATION.cff`. Zenodo: https://doi.org/10.5281/zenodo.24050327. The reference of the article will be added when it exists.
