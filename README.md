# YieldLineage

Implementation of **YieldLineage: Reliability-Gated Wafer Genealogy Reasoning for Early Yield-Excursion Warning in Semiconductor Manufacturing**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate timestamped packet graphs at cutoffs 2/4/6, four wafers per lot and lot-isolated groups. Excursions have event time 8; controls use infinity.

```bash
yieldlineage demo --output data/generated.npz --count 192 --config config.json --seed 13
yieldlineage prepare --input data/generated.npz --output data/prepared
```

`x[N,V,D]` is a graph of source/process/wafer packets, `adjacency[N,V,V]` indicates allowed source-to-destination edges, `time[N,V]` acquisition times, `cutoff[N]` decision times, `available[N,V]`, `uncertainty[N,V]`, and `quality[N,V,Q]` describe reliability. `y[N]` and `lot_y[N]` are wafer/lot excursion labels. Each group is a complete lot; construct packets from actual process/metrology genealogy with timestamped availability.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

Real warning streams also require numeric `wafer`, increasing `cutoff`, and one `event_time` per wafer; the configured threshold and persistence are fixed before test.

## Training and inference

```bash
yieldlineage train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
yieldlineage evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_reliability`, `no_time_guard`. Reuse the prepared data and training seed:

```bash
yieldlineage train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
yieldlineage evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
yieldlineage plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
