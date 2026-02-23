# GMP Cross-Domain Workflow (HEP + non-HEP point clouds)

Date: 2026-02-23

## 1) What GMP is in this repo

GMP in `efficient_particle_transformer_src/networks/parT.py` is a residual local message pass over quantized coordinate bins:

- Input token features `x` are mapped to 2D bins with `(c1, c2) -> (grid_eta, grid_phi)`.
- Features are scattered onto a dense grid (`scatter_add`), optionally mean-reduced.
- A depthwise 2D convolution is applied on the grid.
- Grid features are gathered back to tokens.
- A pointwise linear + layernorm + residual is applied.

Current coordinate modes:

- `gmp_coords="raw"`: uses `(eta, phi)` from 4-vectors.
- `gmp_coords="pt"`: uses `(pt*eta, pt*phi_centered)`.

Files:

- `efficient_particle_transformer_src/networks/parT.py`
- `efficient_particle_transformer_src/networks/example_ParticleTransformerGMP.py`
- `efficient_particle_transformer_src/networks/example_ParticleTransformer_ptGMP.py`

## 2) Recommended non-HEP target dataset

Primary target: **LagrangeBench** (Lagrangian fluid particle simulations).

Why this is a good first non-HEP target:

- Particle-cloud format is close to HEP tokenized particles.
- Public benchmark package includes reproducible baselines and pretrained checkpoints.
- It is specifically built for particle dynamics, so GMP locality bias is a natural fit.

SOTA snapshot for planning:

- Official LagrangeBench package reports strong baseline checkpoints for GNS and SEGNN across scenarios (Taylor-Green vortex, reverse Poiseuille, lid-driven cavity, dam break).
- NeuralSPH (ICML 2024) reports substantial improvements on LagrangeBench-style rollouts and should be treated as a strong reference baseline for modern comparisons.

Note: there is no single always-updated centralized leaderboard that resolves all scenarios into one "global #1" model. For fair comparison, use per-scenario metrics and compare against both the official LagrangeBench baselines and NeuralSPH numbers reported for matching tasks.

## 3) Experiment design (minimal viable study)

Goal: verify whether adding GMP improves accuracy vs the same backbone without GMP.

Matrix:

- HEP control: ParT vs ParT+GMP on JetClass (same training setup, only GMP toggled).
- Non-HEP: GNS/SEGNN baseline vs GNS/SEGNN+GMP on one LagrangeBench scenario first (start with TGV2D), then extend.

Ablations:

- `gmp_grid`: `0.02`, `0.05`, `0.1`
- `gmp_kernel`: `3`, `5`
- `gmp_reduce`: `sum`, `mean`

Metrics to track:

- Rollout MSE (same horizon used in benchmark protocol)
- Sinkhorn / Earth-mover style particle distribution metric (if provided by benchmark task)
- Inference throughput and memory

Success criterion:

- GMP is a win only if it improves rollout metric with <=10% wall-time overhead at matched parameter count.

## 4) Practical workflow

1. Reproduce HEP baseline in this repo.
2. Reproduce HEP + GMP in this repo.
3. In LagrangeBench codebase, insert GMP block after node feature encoding and before the first message-passing update.
4. Run baseline and GMP variants with identical seeds and training budget.
5. Aggregate mean/std over at least 3 seeds.
6. Compare HEP deltas vs non-HEP deltas.

## 5) Immediate next step

Start with one paired run to de-risk setup before full sweep:

- HEP: ParT vs ParTGMP on a reduced JetClass budget.
- Non-HEP: SEGNN baseline vs SEGNN+GMP on TGV2D only.

If both show positive direction, expand to full benchmark grid.
