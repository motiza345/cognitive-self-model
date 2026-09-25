"""Hook isolation tests. The tensor tests do not load Qwen."""

import pytest
import torch

from src.cognitive_self_model.m22_1.intervention import apply_last_token_additive, make_resid_hook


def test_alpha_zero_does_not_copy_or_change_the_residual():
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    direction = torch.ones(8)
    output = apply_last_token_additive(residual, 0.0, direction)
    assert output is residual
    assert torch.equal(output, residual)


def test_nonzero_alpha_changes_only_the_last_position():
    residual = torch.zeros(2, 4, 8)
    direction = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    output = apply_last_token_additive(residual, 2.0, direction)
    assert torch.equal(output[:, :-1, :], residual[:, :-1, :])
    assert torch.equal(output[:, -1, :], residual[:, -1, :] + 2.0 * direction)
    assert not torch.equal(output[:, -1, :], residual[:, -1, :])


def test_hook_recorder_distinguishes_zero_and_nonzero_alpha():
    residual = torch.zeros(1, 3, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])

    class _Hook:
        name = "blocks.2.hook_resid_post"

    zero = {}
    nonzero = {}
    zero_out = make_resid_hook(0.0, direction, zero)(residual, _Hook())
    nonzero_out = make_resid_hook(1.5, direction, nonzero)(residual.clone(), _Hook())
    assert zero["fired"] == 1
    assert zero["last_modified"] is False
    assert zero["other_unchanged"] is True
    assert torch.equal(zero_out, residual)
    assert nonzero["fired"] == 1
    assert nonzero["hook_name"] == "blocks.2.hook_resid_post"
    assert nonzero["last_modified"] is True
    assert nonzero["other_unchanged"] is True
    assert nonzero["max_abs_delta_other"] == 0.0
    assert torch.equal(nonzero_out[:, -1, :], residual[:, -1, :] + 1.5 * direction)


def test_direction_shape_mismatch_is_rejected():
    residual = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError):
        apply_last_token_additive(residual, 1.0, torch.ones(3))


def test_repeated_application_is_deterministic():
    residual = torch.randn(1, 2, 4)
    direction = torch.randn(4)
    first = apply_last_token_additive(residual, -0.5, direction)
    second = apply_last_token_additive(residual, -0.5, direction)
    assert torch.equal(first, second)


def test_tiny_hooked_transformer_registers_only_the_named_layer():
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    config = HookedTransformerConfig(
        n_layers=2,
        d_model=16,
        n_ctx=8,
        d_head=8,
        n_heads=2,
        d_vocab=32,
        act_fn="relu",
        attn_only=False,
    )
    model = HookedTransformer(config)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    direction = torch.zeros(config.d_model)
    direction[0] = 1.0
    seen = []

    def hook_fn(residual, hook):
        seen.append(hook.name)
        updated = residual.clone()
        updated[:, -1, :] = updated[:, -1, :] + direction.to(updated)
        return updated

    with torch.no_grad():
        baseline = model(tokens)
        hooked = model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_resid_post", hook_fn)])
    assert seen == ["blocks.0.hook_resid_post"]
    assert baseline.shape == hooked.shape
    assert not torch.allclose(baseline, hooked)
