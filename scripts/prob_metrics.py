"""
Evaluation metrics for probabilistic forecasts.

Includes:
  - CRPS (Continuous Ranked Probability Score)
  - PI Coverage (Prediction Interval Coverage Probability)
  - PI Width (Mean Prediction Interval Width)
  - Winkler Score (penalizes both coverage and width)
  - Point RMSE (using q50)

Input convention:
  y_pred: (..., 3)  corresponds to [q05, q50, q95]
  y_true: (...)
"""
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def crps_quantile_approx(y_pred, y_true, quantiles=(0.05, 0.5, 0.95)):
    """
    Approximate CRPS as a sum of pinball losses over quantiles.
    
    CRPS(F, y) ~ 2 * sum_q pinball(y_pred_q, y_true, q) * dq
    Here we use 3 quantiles (q05, q50, q95).
    """
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    
    # Approximate CRPS via the trapezoidal rule
    losses = []
    for i, q in enumerate(quantiles):
        diff = y_true - y_pred[..., i]
        l = np.maximum(q * diff, (q - 1) * diff)
        losses.append(l)
    losses = np.stack(losses, axis=0)  # (n_q, ...)
    return float(losses.mean() * 2)


def pi_coverage(y_pred, y_true, alpha=0.10):
    """
    Prediction Interval Coverage Probability (PICP)
    
    y_pred: (..., 3) with [q_low, q_mid, q_high]
    y_true: (...)
    alpha: 1 - confidence level (e.g., 0.10 for 90% PI)
    
    Returns: float in [0, 1]
    """
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    
    q_low = y_pred[..., 0]   # q05
    q_high = y_pred[..., 2]  # q95
    
    inside = (y_true >= q_low) & (y_true <= q_high)
    return float(inside.mean())


def pi_width(y_pred):
    """Prediction Interval Width: mean(q95 - q05)"""
    y_pred = to_numpy(y_pred)
    return float((y_pred[..., 2] - y_pred[..., 0]).mean())


def winkler_score(y_pred, y_true, alpha=0.10):
    """
    Winkler Score (accounts for both coverage and width).
    
    Formula:
      W = (q_high - q_low) + (2/α) * max(0, q_low - y) + (2/α) * max(0, y - q_high)
    
    Narrow intervals are rewarded, but misses are penalized heavily.
    """
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    
    q_low = y_pred[..., 0]
    q_high = y_pred[..., 2]
    
    width = q_high - q_low
    below = np.maximum(0, q_low - y_true)
    above = np.maximum(0, y_true - q_high)
    
    score = width + (2 / alpha) * below + (2 / alpha) * above
    return float(score.mean())


def point_rmse(y_pred, y_true):
    """Point RMSE using q50."""
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    return float(np.sqrt(((y_pred[..., 1] - y_true) ** 2).mean()))


def all_prob_metrics(y_pred, y_true):
    """Compute all probabilistic-forecast metrics at once."""
    return {
        'crps':           crps_quantile_approx(y_pred, y_true),
        'pi_coverage_90': pi_coverage(y_pred, y_true, alpha=0.10),
        'pi_width_mean':  pi_width(y_pred),
        'winkler_90':     winkler_score(y_pred, y_true, alpha=0.10),
        'point_rmse':     point_rmse(y_pred, y_true),
    }


if __name__ == "__main__":
    np.random.seed(0)
    y_true = np.random.rand(100, 30) * 0.6 + 0.2  # ground truth in [0.2, 0.8]
    
    # Simulate a reasonable prediction: q50 near truth, sensible interval
    q50 = y_true + np.random.randn(100, 30) * 0.05
    q05 = q50 - np.abs(np.random.randn(100, 30) * 0.05) - 0.02
    q95 = q50 + np.abs(np.random.randn(100, 30) * 0.05) + 0.02
    y_pred = np.stack([q05, q50, q95], axis=-1)
    
    metrics = all_prob_metrics(y_pred, y_true)
    print("probabilistic metrics (synthetic data):")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    
    print("\nexpected behavior:")
    print("  pi_coverage_90 should be close to 0.90")
    print("  pi_width_mean: smaller is better (sharper)")
    print("  winkler_90: smaller is better")
    print("  point_rmse: smaller is better")
