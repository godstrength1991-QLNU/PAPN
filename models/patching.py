"""
HydroPatch Module 2: Patching

Inspired by PatchTST (ICLR 2023).
Idea: split a 1D time series into patches (analogous to ViT patches).

Input: (B, T, D), e.g. T=60 days.
patching:
  - patch_size=7, stride=7 -> 8 patches of 7 days (1 week) each
  - each patch is flattened into one token

Output: (B, num_patches, patch_size * D)
"""
import torch
import torch.nn as nn


class Patching(nn.Module):
    """
    Split a time series into patches.
    
    Design:
      - patch_size = 7 (one week, matching the irrigation cycle)
      - stride = 7 (non-overlapping)
    """
    def __init__(self, patch_size=7, stride=7):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, x):
        """
        x: (B, T, D)
        Returns: 
          patches: (B, num_patches, patch_size * D)
        """
        B, T, D = x.shape
        
        # Use unfold for efficient splitting
        # (B, T, D) -> (B, D, T) -> unfold -> (B, D, num_patches, patch_size)
        x_t = x.transpose(1, 2)  # (B, D, T)
        patches = x_t.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # (B, D, num_patches, patch_size)
        
        # Rearrange to (B, num_patches, patch_size * D)
        B_, D_, NP, PS = patches.shape
        patches = patches.permute(0, 2, 3, 1)  # (B, num_patches, patch_size, D)
        patches = patches.reshape(B_, NP, PS * D_)
        
        return patches


class PatchEmbedding(nn.Module):
    """
    Project patches to d_model dimensions.
    Standard PatchTST approach.
    """
    def __init__(self, patch_size, in_channels, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_size * in_channels, d_model)
        # Positional encoding (learnable, as in PatchTST)
        self.pos_emb = None  # initialized in forward (num_patches depends on input)
    
    def forward(self, patches):
        """
        patches: (B, num_patches, patch_size * D)
        Returns: (B, num_patches, d_model)
        """
        x = self.proj(patches)  # (B, num_patches, d_model)
        
        # Add positional encoding (learnable, sin-cos init)
        if self.pos_emb is None or self.pos_emb.shape[0] != x.shape[1]:
            self.pos_emb = nn.Parameter(
                self._get_sincos_pos_emb(x.shape[1], x.shape[2]).to(x.device),
                requires_grad=True
            )
        x = x + self.pos_emb.unsqueeze(0)
        return x
    
    @staticmethod
    def _get_sincos_pos_emb(num_patches, d_model):
        pe = torch.zeros(num_patches, d_model)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# Test
if __name__ == "__main__":
    B, T, D = 4, 60, 1
    x = torch.randn(B, T, D)
    
    patcher = Patching(patch_size=7, stride=7)
    patches = patcher(x)
    print(f"input: {x.shape}")
    print(f"patches: {patches.shape}")  # expected (4, 8, 7)
    
    embed = PatchEmbedding(patch_size=7, in_channels=1, d_model=64)
    embedded = embed(patches)
    print(f"embedded: {embedded.shape}")  # expected (4, 8, 64)
