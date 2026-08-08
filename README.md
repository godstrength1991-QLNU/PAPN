# PAPN — Persistence-Anchored Probabilistic Network

Reference implementation of **PAPN**, a lightweight probabilistic forecasting
model for agricultural-reservoir storage rates. PAPN anchors its forecasts on
the persistence baseline and predicts a horizon-adaptive residual distribution,
producing calibrated prediction intervals with a small parameter budget.

This repository contains the **model and training code only**. The observational
dataset and trained weights are **not** included (see *Data* below).

## Method in brief

- **Persistence anchor.** The forecast is centered on the last observed storage
  value; the network learns a bounded residual on top of it, so the model
  degrades gracefully to persistence when no signal is present.
- **Horizon-adaptive uncertainty.** The residual bound and predictive spread
  scale with the forecast horizon, reflecting that near-term storage is
  strongly inertia-dominated while longer horizons are more uncertain.
- **Quantile output + conformal calibration.** The model is trained with a
  pinball (quantile) loss and calibrated post-hoc with split conformal
  prediction to achieve nominal interval coverage.

## Repository layout

```
models/
  papn.py            PAPN model (self-contained, depends only on torch)
  decomposition.py   trend/seasonal decomposition utilities
  patching.py        patch-embedding utilities
scripts/
  conformal.py       split conformal calibration (per-horizon)
  prob_metrics.py    CRPS, interval coverage/width, Winkler, RMSE
  quantile_loss.py   multi-quantile (pinball) loss
  metrics.py         point-forecast metrics
data/
  dataset.py         windowing / normalization / DataLoader construction
train_papn.py        training + evaluation entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

The experiments in the paper use daily storage-rate records for agricultural
reservoirs obtained from Korea's **Rural Agricultural Water Resource Information
System (RAWRIS)**. This dataset is **not redistributed here** and must be
obtained from the original provider. `data/dataset.py` documents the expected
input format (per-reservoir daily storage series plus static attributes); once
a compatible dataset object is supplied, training runs unchanged.

## Usage

```bash
# Train PAPN (expects a prepared dataset object; see data/dataset.py)
python train_papn.py --epochs 150 --seed 42
```

Evaluation reports CRPS, 90% prediction-interval coverage and mean width,
Winkler score, and point RMSE, both before and after conformal calibration.

## Citation

If you use this code, please cite the associated paper (details to be added
upon publication).

## License

Code is released under the MIT License (see `LICENSE`). The MIT License covers
the source code in this repository only; it does **not** grant any rights to the
underlying reservoir dataset, which remains subject to the terms of its original
provider.
