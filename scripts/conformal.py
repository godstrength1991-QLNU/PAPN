"""
Split Conformal Prediction - calibrate prediction intervals to guarantee coverage.

Method (Conformalized Quantile Regression, Romano et al. NeurIPS 2019):
  1. Compute conformity scores on a calibration set:
     score_i = max(q_low_i - y_i, y_i - q_high_i)
     (how far the truth falls outside the interval; negative = inside)
  2. Take the (1-alpha) quantile of the scores = Q_hat
  3. At test time, widen both ends of the interval by Q_hat:
     [q_low - Q_hat, q_high + Q_hat]
  
Guarantee: test coverage >= 1-alpha (finite-sample).

This upgrades an approximate empirical coverage to a formal coverage guarantee.
"""
import numpy as np


def compute_conformity_scores(y_true, q_low, q_high):
    """
    y_true: (...,)  ground truth
    q_low:  (...,)  lower bound (q05)
    q_high: (...,)  upper bound (q95)
    Returns: conformity scores (...,) - negative means inside the interval
    """
    return np.maximum(q_low - y_true, y_true - q_high)


def calibrate(y_cal, q_low_cal, q_high_cal, alpha=0.10):
    """
    Compute the conformal adjustment Q_hat on the calibration set.
    
    Args:
      y_cal:      (n_cal,) calibration ground truth
      q_low_cal:  (n_cal,) calibration lower bounds
      q_high_cal: (n_cal,) calibration upper bounds
      alpha:      significance level (0.10 -> 90% coverage)
    
    Returns: Q_hat (float) - interval widening amount
    """
    scores = compute_conformity_scores(y_cal, q_low_cal, q_high_cal)
    n = len(scores)
    # Finite-sample correction: use the ceil((n+1)(1-alpha))/n quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    Q_hat = np.quantile(scores, q_level, method='higher')
    return float(Q_hat)


def apply_conformal(q_low, q_high, Q_hat):
    """Apply the calibration amount to the test intervals."""
    return q_low - Q_hat, q_high + Q_hat


def conformal_calibrate_and_apply(
    y_cal, q_pred_cal,        # calibration set
    q_pred_test,              # test predictions (..., 3)
    alpha=0.10
):
    """
    Full pipeline: compute Q_hat on the calibration set, apply to the test set.
    
    Args:
      y_cal:       (n_cal,) or higher-dim, will be flattened
      q_pred_cal:  (..., 3) calibration [q05, q50, q95]
      q_pred_test: (..., 3) test [q05, q50, q95]
      alpha:       0.10 → 90%
    
    Returns:
      q_test_calibrated: (..., 3) calibrated test quantiles
      Q_hat: float
    """
    y_cal = np.asarray(y_cal).flatten()
    q_low_cal = np.asarray(q_pred_cal[..., 0]).flatten()
    q_high_cal = np.asarray(q_pred_cal[..., 2]).flatten()
    
    Q_hat = calibrate(y_cal, q_low_cal, q_high_cal, alpha)
    
    # Apply to the test set
    q_test_cal = q_pred_test.copy()
    q_test_cal[..., 0] = q_pred_test[..., 0] - Q_hat  # widen lower bound down
    q_test_cal[..., 2] = q_pred_test[..., 2] + Q_hat  # widen upper bound up
    # q50 unchanged
    # clip to [0, 1]
    q_test_cal = np.clip(q_test_cal, 0.0, 1.0)
    
    return q_test_cal, Q_hat


def per_horizon_conformal(y_cal, q_pred_cal, q_pred_test, alpha=0.10):
    """
    Calibrate each horizon separately (longer horizons are more uncertain).
    
    y_cal:       (n_cal, H)
    q_pred_cal:  (n_cal, H, 3)
    q_pred_test: (n_test, H, 3)
    
    Returns: (q_test_calibrated (n_test,H,3), Q_hats (H,))
    """
    y_cal = np.asarray(y_cal)
    q_pred_cal = np.asarray(q_pred_cal)
    q_pred_test = np.asarray(q_pred_test).copy()
    
    H = y_cal.shape[-1]
    Q_hats = np.zeros(H)
    
    for h in range(H):
        Q_hat = calibrate(
            y_cal[..., h],
            q_pred_cal[..., h, 0],
            q_pred_cal[..., h, 2],
            alpha
        )
        Q_hats[h] = Q_hat
        q_pred_test[..., h, 0] -= Q_hat
        q_pred_test[..., h, 2] += Q_hat
    
    q_pred_test = np.clip(q_pred_test, 0.0, 1.0)
    return q_pred_test, Q_hats


if __name__ == "__main__":
    np.random.seed(0)
    # Simulate: ground truth + an under-confident prediction (intervals too narrow)
    n_cal, n_test, H = 500, 500, 30
    y_cal = np.random.rand(n_cal, H) * 0.5 + 0.25
    y_test = np.random.rand(n_test, H) * 0.5 + 0.25
    
    def fake_pred(y):
        q50 = y + np.random.randn(*y.shape) * 0.03
        q05 = q50 - 0.04  # deliberately too narrow
        q95 = q50 + 0.04
        return np.stack([q05, q50, q95], axis=-1)
    
    q_cal = fake_pred(y_cal)
    q_test = fake_pred(y_test)
    
    # Coverage before calibration
    cov_before = ((y_test >= q_test[..., 0]) & (y_test <= q_test[..., 2])).mean()
    print(f"coverage before (90% PI): {cov_before:.3f} (too narrow, below 0.90)")
    
    # Global conformal
    q_test_cal, Q_hat = conformal_calibrate_and_apply(y_cal, q_cal, q_test, alpha=0.10)
    cov_after = ((y_test >= q_test_cal[..., 0]) & (y_test <= q_test_cal[..., 2])).mean()
    print(f"coverage after (90% PI): {cov_after:.3f} (Q_hat={Q_hat:.4f})")
    print("  OK: coverage close to 0.90" if cov_after >= 0.88 else "  WARN: needs adjustment")
    
    # Per-horizon conformal
    q_test_ph, Q_hats = per_horizon_conformal(y_cal, q_cal, q_test, alpha=0.10)
    cov_ph = ((y_test >= q_test_ph[..., 0]) & (y_test <= q_test_ph[..., 2])).mean()
    print(f"per-horizon coverage after: {cov_ph:.3f}")
    print(f"  Q_hats range: [{Q_hats.min():.4f}, {Q_hats.max():.4f}]")
