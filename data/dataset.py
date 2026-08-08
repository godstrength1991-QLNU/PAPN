"""
Data loader: turns (N, T, ...) reservoir data into training samples.
Each sample: (N reservoirs, T_in days history) -> (N reservoirs, H days forecast).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict


class ReservoirDataset(Dataset):
    """
    Input: a dict of already split-by-time arrays.
    Each returned sample:
        storage_hist:  (N, T_in)     historical storage rate
        meteo_hist:    (N, T_in, 4)  historical meteorology
        climate_hist:  (T_in, 3)     historical climate indices
        storage_fut:   (N, H)        future storage rate (labels)
        meteo_fut:     (N, H, 4)     future meteorology (optional)
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        T_in: int = 60,
        H: int = 30,
        stride: int = 1,
        normalize_stats: dict = None,
    ):
        self.T_in = T_in
        self.H = H
        self.storage  = data['storage_rate']  # (N, T)
        self.meteo    = data['meteo']         # (N, T, 4)
        self.climate  = data['climate']       # (T, 3)
        self.group_ids = data['group_ids']    # (N,)
        self.locations = data['locations']    # (N, 2)
        self.capacity  = data['capacity']     # (N,)

        N, T = self.storage.shape
        self.N = N
        self.T = T

        # Sliding-window start positions
        self.starts = list(range(0, T - T_in - H + 1, stride))

        # Normalization stats (fit on train only, then passed to val/test)
        if normalize_stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = normalize_stats

        # Apply normalization
        self.meteo_norm = (self.meteo - self.stats['meteo_mean']) / (self.stats['meteo_std'] + 1e-6)
        self.climate_norm = (self.climate - self.stats['climate_mean']) / (self.stats['climate_std'] + 1e-6)
        # storage_rate is already in [0, 1], no normalization needed

    def _compute_stats(self):
        return {
            'meteo_mean':   self.meteo.mean(axis=(0, 1), keepdims=False),    # (4,)
            'meteo_std':    self.meteo.std(axis=(0, 1), keepdims=False),     # (4,)
            'climate_mean': self.climate.mean(axis=0),                       # (3,)
            'climate_std':  self.climate.std(axis=0),                        # (3,)
        }

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e_hist = s + self.T_in
        e_fut  = e_hist + self.H

        return {
            'storage_hist':  torch.from_numpy(self.storage[:, s:e_hist]).float(),       # (N, T_in)
            'meteo_hist':    torch.from_numpy(self.meteo_norm[:, s:e_hist]).float(),    # (N, T_in, 4)
            'climate_hist':  torch.from_numpy(self.climate_norm[s:e_hist]).float(),     # (T_in, 3)
            'storage_fut':   torch.from_numpy(self.storage[:, e_hist:e_fut]).float(),   # (N, H)
            'meteo_fut':     torch.from_numpy(self.meteo[:, e_hist:e_fut]).float(),     # (N, H, 4) raw values (optional)
        }


def make_loader(data, T_in, H, batch_size=4, shuffle=True, stats=None):
    """Convenience function to build a DataLoader."""
    ds = ReservoirDataset(data, T_in=T_in, H=H, normalize_stats=stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, ds.stats


if __name__ == "__main__":
    # Minimal smoke test with random arrays in the expected format.
    # Replace this with your own dataset (see README for the required fields).
    import numpy as np
    N, T = 60, 730
    data = {
        'storage_rate': np.random.rand(N, T).astype('float32'),
        'meteo':        np.random.rand(N, T, 4).astype('float32'),
        'climate':      np.random.rand(T, 3).astype('float32'),
        'group_ids':    np.zeros(N, dtype='int64'),
        'locations':    np.random.rand(N, 2).astype('float32'),
        'capacity':     (np.random.rand(N) * 1e4).astype('float32'),
    }
    loader, stats = make_loader(data, T_in=60, H=30, batch_size=4, shuffle=True)
    print(f"Batches: {len(loader)}")
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
