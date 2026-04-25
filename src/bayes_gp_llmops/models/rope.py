from __future__ import annotations

import torch


def build_rope_cache(
    sequence_length: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rotary embedding cosine/sine caches for one sequence length."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("head_dim must be even and at least 2 for RoPE.")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def apply_rope(
    tensor: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to a tensor shaped [batch, heads, seq, head_dim]."""

    if tensor.ndim != 4:
        raise ValueError("RoPE expects tensor with shape [batch, heads, seq, head_dim].")
    batch_size, num_heads, seq_len, head_dim = tensor.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even.")
    if cos.shape != (seq_len, head_dim // 2):
        raise ValueError("cos cache shape does not match tensor shape.")
    if sin.shape != (seq_len, head_dim // 2):
        raise ValueError("sin cache shape does not match tensor shape.")

    cos_view = cos.unsqueeze(0).unsqueeze(0)
    sin_view = sin.unsqueeze(0).unsqueeze(0)
    even = tensor[..., ::2]
    odd = tensor[..., 1::2]
    rotated_even = (even * cos_view) - (odd * sin_view)
    rotated_odd = (even * sin_view) + (odd * cos_view)
    interleaved = torch.stack((rotated_even, rotated_odd), dim=-1)
    return interleaved.reshape(batch_size, num_heads, seq_len, head_dim)


def apply_rope_to_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""

    return apply_rope(query, cos=cos, sin=sin), apply_rope(key, cos=cos, sin=sin)
