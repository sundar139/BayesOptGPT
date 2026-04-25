from __future__ import annotations

import torch

from bayes_gp_llmops.models.attention import MultiHeadSelfAttention
from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.models.rmsnorm import RMSNorm
from bayes_gp_llmops.models.rope import apply_rope, build_rope_cache
from bayes_gp_llmops.models.swiglu import SwiGLU


def test_rmsnorm_preserves_shape() -> None:
    layer = RMSNorm(hidden_size=32)
    inputs = torch.randn(2, 8, 32)
    outputs = layer(inputs)
    assert outputs.shape == inputs.shape


def test_rope_cache_and_application_shapes() -> None:
    tensor = torch.randn(2, 4, 16, 8)
    cos, sin = build_rope_cache(
        sequence_length=16,
        head_dim=8,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    rotated = apply_rope(tensor, cos=cos, sin=sin)
    assert cos.shape == (16, 4)
    assert sin.shape == (16, 4)
    assert rotated.shape == tensor.shape


def test_swiglu_preserves_hidden_dimension() -> None:
    layer = SwiGLU(hidden_size=64, multiplier=4.0, dropout=0.0)
    inputs = torch.randn(3, 10, 64)
    outputs = layer(inputs)
    assert outputs.shape == inputs.shape


def test_attention_forward_shape() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=64,
        num_attention_heads=8,
        dropout=0.0,
        rope_base=10000.0,
    )
    hidden_states = torch.randn(2, 12, 64)
    attention_mask = torch.ones(2, 12, dtype=torch.long)
    outputs = attention(hidden_states, attention_mask=attention_mask)
    assert outputs.shape == hidden_states.shape


def test_classifier_forward_shape() -> None:
    model = TinyLlamaForSequenceClassification(
        ModelConfig(
            vocab_size=256,
            max_sequence_length=32,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=8,
            feedforward_multiplier=4.0,
            dropout=0.1,
            num_classes=4,
            rope_base=10000.0,
            pooling="masked_mean",
        )
    )
    input_ids = torch.randint(0, 256, (4, 16), dtype=torch.long)
    attention_mask = torch.ones(4, 16, dtype=torch.long)
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 4)
