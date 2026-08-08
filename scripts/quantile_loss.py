"""
Quantile (Pinball) Loss

Formulation:
  Loss_q(y_true, y_pred) = max(q * (y_true - y_pred), (q-1) * (y_true - y_pred))
  
Intuition:
  - If y_true > y_pred (under-prediction), loss = q * diff
  - If y_true < y_pred (over-prediction), loss = (1-q) * diff
  - q=0.5 reduces to the symmetric absolute loss (MAE)
  - q=0.05 penalizes over-prediction more (encourages lower predictions)
  - q=0.95 penalizes under-prediction more (encourages higher predictions)
"""
import torch


def pinball_loss(y_pred, y_true, quantile):
    """
    y_pred: (...,)
    y_true: (...,) 
    quantile: float in (0, 1)
    Returns: scalar loss
    """
    diff = y_true - y_pred
    loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
    return loss.mean()


def multi_quantile_loss(y_pred_quantiles, y_true, quantiles=(0.05, 0.5, 0.95)):
    """
    y_pred_quantiles: (..., N_Q)   the N_Q quantile predictions
    y_true: (...,)                 single ground truth, broadcast
    quantiles: tuple of N_Q floats
    Returns: average loss
    """
    losses = []
    for i, q in enumerate(quantiles):
        loss_q = pinball_loss(y_pred_quantiles[..., i], y_true, q)
        losses.append(loss_q)
    return sum(losses) / len(losses)


if __name__ == "__main__":
    y_true = torch.rand(10, 30)  # ground truth
    y_pred = torch.rand(10, 30, 3)  # predict 3 quantiles
    # Sort to enforce monotonicity
    y_pred = torch.sort(y_pred, dim=-1)[0]
    
    loss = multi_quantile_loss(y_pred, y_true)
    print(f"Pinball loss: {loss.item():.4f}")
    print("Pinball loss OK")
