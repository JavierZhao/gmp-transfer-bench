# GMP Experiments on LagrangeBench (Sequential)

Date: 2026-02-23

This repo now supports an optional GMP block for both `segnn` and `gns` models.

## 1) Setup

```bash
cd /Users/zhaozihan/Desktop/GMP_point_cloud/lagrangebench_src
poetry install --only main
```

(Optional) download dataset first:

```bash
bash download_data.sh tgv_2d datasets/
```

## 2) SEGNN baseline -> SEGNN+GMP (recommended first)

Run baseline:

```bash
python main.py config=configs/tgv_2d/segnn.yaml mode=all
```

Run GMP variant:

```bash
python main.py config=configs/tgv_2d/segnn_gmp.yaml mode=all
```

## 3) GNS baseline -> GNS+GMP

Run baseline:

```bash
python main.py config=configs/tgv_2d/gns.yaml mode=all
```

Run GMP variant:

```bash
python main.py config=configs/tgv_2d/gns_gmp.yaml mode=all
```

## 4) Hyperparameter sweep (GMP only)

```bash
python main.py config=configs/tgv_2d/segnn_gmp.yaml model.gmp_grid_bins=32 mode=all
python main.py config=configs/tgv_2d/segnn_gmp.yaml model.gmp_grid_bins=64 mode=all
python main.py config=configs/tgv_2d/segnn_gmp.yaml model.gmp_grid_bins=96 mode=all
python main.py config=configs/tgv_2d/segnn_gmp.yaml model.gmp_reduce=mean mode=all
```

## 5) Notes

- GMP uses fixed-size quantization grids for JAX JIT stability (`model.gmp_grid_bins`).
- In SEGNN, GMP is injected after embedding and before the first message-passing layer.
- In GNS, GMP is injected after encoder and before processor/message-passing.
