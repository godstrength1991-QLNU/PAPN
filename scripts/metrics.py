"""
Evaluation metrics: RMSE / NSE / R2 / WBR
"""
import torch
import numpy as np


def rmse_per_reservoir(y_pred, y_true):
    """Per-reservoir RMSE.
    y_pred, y_true: (..., N, H)
    return: (..., N)
    """
    return ((y_pred - y_true) ** 2).mean(dim=-1).sqrt()


def nse_per_reservoir(y_pred, y_true, min_var=1e-3):
    """Per-reservoir Nash-Sutcliffe Efficiency.
    NSE = 1 - SS_res / SS_tot
    Returns NaN when SS_tot is tiny (near-constant truth) to avoid unstable extremes.
    """
    y_mean = y_true.mean(dim=-1, keepdim=True)  # (..., N, 1)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=-1)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=-1)
    # When variation is too small, NSE is meaningless -> NaN
    nse = 1.0 - ss_res / (ss_tot + 1e-8)
    # Mark reservoirs with tiny ss_tot as NaN (filtered when aggregating)
    nse = torch.where(ss_tot > min_var, nse, torch.full_like(nse, float('nan')))
    return nse


def r2_per_reservoir(y_pred, y_true):
    """R2 coefficient of determination.
    Same formula as NSE."""
    return nse_per_reservoir(y_pred, y_true)


def water_balance_residual(y_pred, meteo_fut, capacity, Q_out_est=None):
    """
    WBR — Water Balance Residual
    
    Mean absolute water-balance residual per reservoir.
    
    y_pred: (B, N, H)
    meteo_fut: (B, N, H, 4)
    capacity: (N,)
    """
    B, N, H = y_pred.shape
    y_padded = torch.cat([y_pred[..., :1], y_pred], dim=-1)
    delta_S = y_padded[..., 1:] - y_padded[..., :-1]  # (B, N, H)

    P = meteo_fut[..., 0]
    ET = meteo_fut[..., 2]
    if Q_out_est is None:
        Q_out_est = torch.full_like(P, 1.5)

    cap = capacity.view(1, N, 1).to(y_pred.device)
    physical = (P - ET - Q_out_est) / (cap + 1e-6) / 50
    residual = (delta_S - physical).abs().mean(dim=-1)  # (B, N)
    return residual


def aggregate_metrics(y_pred, y_true, meteo_fut=None, capacity=None):
    """Aggregate all metrics over a batch.
    Returns a dict: rmse, nse, r2, (wbr).
    NSE/R2 skip reservoirs whose SS_tot is too small (weak signal variation).
    """
    rmse = rmse_per_reservoir(y_pred, y_true).mean().item()
    nse_arr = nse_per_reservoir(y_pred, y_true)
    r2_arr  = r2_per_reservoir(y_pred, y_true)
    # nanmean skips reservoirs with tiny SS_tot
    nse = torch.nanmean(nse_arr).item() if not torch.isnan(nse_arr).all() else float('nan')
    r2  = torch.nanmean(r2_arr).item() if not torch.isnan(r2_arr).all() else float('nan')
    out = {'rmse': rmse, 'nse': nse, 'r2': r2}
    if meteo_fut is not None and capacity is not None:
        wbr = water_balance_residual(y_pred, meteo_fut, capacity).mean().item()
        out['wbr'] = wbr
    return out


# Test
if __name__ == "__main__":
    B, N, H = 2, 60, 30
    y_pred = torch.rand(B, N, H)
    y_true = torch.rand(B, N, H)
    meteo = torch.randn(B, N, H, 4).abs() + 0.1
    cap = torch.rand(N) * 1e5 + 1e4

    metrics = aggregate_metrics(y_pred, y_true, meteo, cap)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
