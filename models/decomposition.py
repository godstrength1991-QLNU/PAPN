"""
HydroPatch Module 1: Seasonal-Trend Decomposition (STD)

Inspired by DLinear (AAAI 2023).
Idea: a time series = trend (low frequency) + seasonal (high frequency).
Well suited to reservoir storage, whose seasonal pattern is strong.

Formulation:
  trend_t = MovingAvg(x, kernel_size)
  seasonal_t = x_t - trend_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """Moving average filter implemented via 1D pooling."""
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        # Padding keeps the output length equal to the input length
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        """
        x: (B, T, D) where D is the feature dim (1 for storage rate)
        Returns: (B, T, D) trend
        """
        # Pad both ends with edge values to avoid boundary artifacts
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # AvgPool expects (B, D, T), so permute
        x_t = x_padded.permute(0, 2, 1)
        trend = self.avg(x_t).permute(0, 2, 1)
        return trend


class SeasonalTrendDecomp(nn.Module):
    """
    Series -> (trend, seasonal).
    kernel_size options:
      - 25: standard (DLinear default)
      - 30: monthly average
      - 365: yearly average (annual trend)
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # must be odd
        self.moving_avg = MovingAvg(kernel_size)
    
    def forward(self, x):
        """
        x: (B, T, D)
        Returns:
          trend: (B, T, D) - low frequency
          seasonal: (B, T, D) - high-frequency residual
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


# Test
if __name__ == "__main__":
    import numpy as np
    
    # Build a test series: trend + seasonal
    T = 60
    t = np.arange(T)
    trend_true = 0.5 + 0.01 * t  # slow upward trend
    seasonal_true = 0.1 * np.sin(2 * np.pi * t / 14)  # biweekly cycle
    noise = 0.02 * np.random.randn(T)
    signal = trend_true + seasonal_true + noise
    
    x = torch.tensor(signal, dtype=torch.float32).view(1, T, 1)
    
    decomp = SeasonalTrendDecomp(kernel_size=15)
    trend, seasonal = decomp(x)
    print(f"input: {x.shape}")
    print(f"trend: {trend.shape}, mean {trend.mean():.3f}")
    print(f"seasonal: {seasonal.shape}, mean {seasonal.mean():.3f}")
    print(f"reconstruction error: {(x - trend - seasonal).abs().mean():.6f} (should be ~0)")
