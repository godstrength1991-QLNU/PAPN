"""
PAPN - Persistence-Anchored Probabilistic Network

Design motivation:
  - Residuals exhibit strong horizon structure (std grows with lead time).
  - Residual median is ~0 (inertia-dominated) with a heavy tail (drought/flood).
  - A fixed residual bound truncates large deviations.

Design choices:
  - The residual bound is horizon-adaptive (smaller near-term, larger long-term).
  - The residual head is initialized near zero, so the model defaults to persistence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_Q = 3
Z_90 = 1.645


class ResidualEncoder(nn.Module):
    def __init__(self, T_in=60, hidden=64, n_layers=4, kernel=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = 1 if i == 0 else hidden
            layers += [
                nn.Conv1d(in_ch, hidden, kernel,
                          padding=(kernel-1)*dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(dropout),
            ]
        self.tcn = nn.Sequential(*layers)
        self.T_in = T_in

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))[:, :, :self.T_in]
        return h[:, :, -1]


class PAPN(nn.Module):
    def __init__(self, T_in=60, H=30, hidden=64, n_layers=4,
                 max_residual=0.6, dropout=0.1):
        super().__init__()
        self.T_in, self.H = T_in, H

        self.encoder = ResidualEncoder(T_in, hidden, n_layers, dropout=dropout)

        self.residual_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, H),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, H),
        )

        # Horizon-adaptive residual bound
        # Near-term horizons get a tighter bound, long-term horizons a looser one
        # Shape ~ sqrt(h/H), scaled by the base max_residual
        horizon_scale = torch.sqrt(torch.arange(1, H + 1).float() / H)
        self.register_buffer('max_res_per_h', horizon_scale * max_residual)

        # Horizon prior for sigma
        self.register_buffer('horizon_prior', torch.sqrt(torch.arange(1, H+1).float() / H))

        # Initialize the residual head near zero (defaults to persistence)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, storage_hist, **kwargs):
        B, N, T_in = storage_hist.shape
        x = storage_hist.reshape(B * N, T_in, 1)

        anchor = storage_hist[:, :, -1].reshape(B * N, 1).expand(B * N, self.H)

        feat = self.encoder(x)

        # Per-horizon residual bound
        delta = torch.tanh(self.residual_head(feat)) * self.max_res_per_h.unsqueeze(0)
        q50 = anchor + delta

        log_sigma = self.uncertainty_head(feat)
        sigma = F.softplus(log_sigma) * self.horizon_prior.unsqueeze(0) + 1e-4

        q05 = q50 - Z_90 * sigma
        q95 = q50 + Z_90 * sigma

        q = torch.stack([q05, q50, q95], dim=-1)
        q = torch.clamp(q, 0.0, 1.0)
        q = torch.sort(q, dim=-1)[0]
        return q.view(B, N, self.H, N_Q)


if __name__ == "__main__":
    model = PAPN(T_in=60, H=30)
    print(f"PAPN v2 params: {sum(p.numel() for p in model.parameters()):,}")
    B, N = 2, 50
    storage = torch.rand(B, N, 60) * 0.4 + 0.4
    q = model(storage)
    print(f"output: {q.shape}")

    # At init, delta should be ~0, so q50 ~ anchor (i.e. persistence)
    anchor = storage[:, :, -1:].expand(-1, -1, 30)
    q50 = q[..., 1]
    diff = (q50 - anchor).abs().mean()
    print(f"at init |q50 - anchor| = {diff:.5f} (should be ~0, i.e. persistence)")
    print(f"  {'OK: defaults to persistence' if diff < 0.01 else 'WARN: check init'}")

    # Check the per-horizon residual bound
    print(f"max_residual per-horizon: H=1 {model.max_res_per_h[0]:.4f}, H=30 {model.max_res_per_h[29]:.4f}")
    print("✓ PAPN v2 OK")
