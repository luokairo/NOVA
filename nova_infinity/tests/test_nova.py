"""CPU behavior checks for NOVA's entropy, selection, and block wiring."""
import copy
import importlib.util
import math
from functools import partial
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_file('nova_utils_under_test', 'infinity/models/nova_utils.py')


def reference_entropy(x):
    energy = x.float().square().clamp_min(1e-6)
    p = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return -(p * (p + 1e-6).log()).sum(-1, keepdim=True) / math.log(x.shape[-1] + 1e-12)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['random', 'zero', 'noncontiguous'])
def test_entropy_matches_original_formula(dtype, kind):
    x = torch.randn(2, 15, 16, generator=torch.Generator().manual_seed(5)).to(dtype)
    if kind == 'zero':
        x.zero_()
    elif kind == 'noncontiguous':
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    original = x.clone()
    expected = reference_entropy(x)
    actual = utils.compute_layer_entropy(x)
    torch.testing.assert_close(actual['ent'], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual['ent_mean'], expected.mean(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize('requested', [0, 7, 15])
def test_selection_and_cache_restoration(requested):
    x = torch.randn(2, 15, 16, generator=torch.Generator().manual_seed(8))
    prior_mean = torch.tensor([[0.75], [0.8]])
    entropy = reference_entropy(x)
    mean = entropy.mean(dim=1)
    multiplier = ((mean - prior_mean) / (mean + prior_mean)).mean().item() + 1
    count = max(0, min(math.ceil(requested * multiplier), 15))
    indices = entropy.squeeze(-1).topk(count, dim=1, largest=True, sorted=False).indices.unsqueeze(-1)
    merge, unmerge, get_indices, updated = utils.masked_previous_scale_cache_dynamic(
        x, requested, (1, 3, 5), prior_mean,
    )
    torch.testing.assert_close(get_indices(), indices, rtol=0, atol=0)
    torch.testing.assert_close(updated, 0.5 * (mean + prior_mean), rtol=0, atol=0)
    selected = x.gather(1, indices.expand(-1, -1, 16))
    torch.testing.assert_close(merge(x), selected, rtol=0, atol=0)
    cache = torch.randn(2, 4, 16)
    original_cache = cache.clone()
    expected = F.interpolate(cache.view(2, 2, 2, 16).permute(0, 3, 1, 2), (3, 5), mode='area')
    expected = expected.permute(0, 2, 3, 1).reshape(2, 15, 16).clone()
    expected.scatter_(1, indices.expand(-1, -1, 16), selected * 2)
    torch.testing.assert_close(unmerge(selected * 2, cache, (2, 2)), expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)


def test_ineligible_submodule_keeps_input():
    x = torch.randn(2, 9, 8)
    merge, unmerge, indices, mean = utils.compute_merge(x, x_shape=(1, 3, 3))
    assert merge(x) is x
    assert unmerge(x, None, None) is x
    assert indices is None and mean == 0.0


@pytest.fixture
def blocks(monkeypatch):
    # These tests do not invoke CUDA kernels. Supply import placeholders, then
    # replace submodules with deterministic updates to test actual block wiring.
    flash = types.ModuleType('flash_attn')
    flash.flash_attn_func = None
    flash.flash_attn_varlen_kvpacked_func = None
    layers = types.ModuleType('timm.models.layers')
    layers.DropPath = torch.nn.Identity
    layers.drop_path = None
    for name, module in {
        'flash_attn': flash,
        'timm': types.ModuleType('timm'),
        'timm.models': types.ModuleType('timm.models'),
        'timm.models.layers': layers,
        'infinity.models': types.ModuleType('infinity.models'),
        'infinity.models.nova_utils': utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    basic = load_file('infinity.models.basic', 'infinity/models/basic.py')
    monkeypatch.setitem(sys.modules, 'infinity.models.basic', basic)
    return load_file('nova_blocks_under_test', 'infinity/models/nova_basic.py')


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_rotary_selection_matches_dense_positions(blocks, batch):
    schedule = [(1, 1, 1), (1, 3, 5)]
    angles = torch.randn(16, 4)
    grid = torch.stack([angles.cos(), angles.sin()]).reshape(2, 1, 1, 1, 16, 4)
    q, k = torch.randn(batch, 2, 15, 8), torch.randn(batch, 2, 15, 8)
    idx = torch.stack([torch.randperm(15)[:7] for _ in range(batch)]).unsqueeze(-1)
    kwargs = dict(scale_schedule=schedule, rope2d_freqs_grid={str(tuple(schedule)): grid},
                  pad_to_multiplier=1, rope2d_normalized_by_hw=2, scale_ind=1, seq_len=15)
    dense_q, dense_k = blocks.apply_rotary_emb_selected(q, k, **kwargs)
    gather_idx = idx[:, None].expand(-1, 2, -1, 8)
    selected_q, selected_k = blocks.apply_rotary_emb_selected(
        q.gather(2, gather_idx), k.gather(2, gather_idx), rope_idx_fn=lambda: idx, **kwargs,
    )
    torch.testing.assert_close(selected_q, dense_q.gather(2, gather_idx), rtol=0, atol=0)
    torch.testing.assert_close(selected_k, dense_k.gather(2, gather_idx), rtol=0, atol=0)


class RecordedUpdate(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor, self.lengths = factor, []

    def forward(self, x, *args, **kwargs):
        self.lengths.append(x.shape[1])
        return x * self.factor


@torch.no_grad()
def test_bf16_kv_cache_matches_full_attention(blocks, monkeypatch):
    def attention(q, k, v, dropout_p, softmax_scale):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, scale=softmax_scale,
        ).transpose(1, 2).contiguous()

    monkeypatch.setattr(blocks, 'flash_attn_func', attention)
    module = blocks.NOVASelfAttention(
        embed_dim=32, num_heads=4, cos_attn=True, customized_flash_attn=False,
    ).bfloat16().eval()
    full = copy.deepcopy(module)
    schedule = [(1, 1, 1), (1, 2, 2)]
    angles = torch.randn(5, 4)
    grid = torch.stack([angles.cos(), angles.sin()]).reshape(2, 1, 1, 1, 5, 4)
    kwargs = dict(attn_bias_or_two_vector=None, scale_schedule=schedule,
                  rope2d_freqs_grid={str(tuple(schedule)): grid})
    x = torch.randn(2, 5, 32).bfloat16()
    expected = full(x, scale_ind=0, ori_len=5, **kwargs)[:, 1:]
    module.kv_caching(True)
    module(x[:, :1], scale_ind=0, ori_len=1, **kwargs)
    actual = module(x[:, 1:], scale_ind=1, ori_len=4, **kwargs)
    assert module.cached_k.dtype == module.cached_v.dtype == torch.bfloat16
    assert module.cached_k.shape[2] == 5
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def simple_norm(C, eps, x, scale, shift):
    return F.layer_norm(x, (C,), eps=eps) * (scale + 1) + shift


@torch.no_grad()
@pytest.mark.parametrize('nova_enabled,threshold,starts', [
    (True, 1.2, True), (True, 0.1, False), (False, 1.2, False),
])
@pytest.mark.parametrize('last_width', [24, 64])
def test_shared_scale_entropy_preserves_block_decisions(blocks, nova_enabled, threshold, starts, last_width):
    block = blocks.NOVACrossAttnBlock(
        embed_dim=64, kv_dim=64, cross_attn_layer_scale=-1, cond_dim=64,
        act=False, shared_aln=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=4, fused_norm_func=simple_norm,
    ).eval()
    block.nova_enabled = nova_enabled
    block.nova_entropy_threshold = threshold
    block.sa, block.ca, block.ffn = RecordedUpdate(0.2), RecordedUpdate(0.3), RecordedUpdate(0.4)
    schedule = [(1, width, width) for width in [1, 2, 4, 6, 8, 12, 16, 20, last_width]]
    block._reset(schedule)
    shared = copy.deepcopy(block)
    cond = torch.zeros(2, 1, 6, 64)
    for si in [7, 8]:
        history = [torch.full((2, 9), 0.1 * index) for index in range(si)]
        means = [entry.float().mean() for entry in history]
        delta = (means[-1] - means[-2]).abs() + (means[-2] - means[-3]).abs()
        width = schedule[si][1]
        x = torch.randn(2, width * width, 64)
        args = dict(x=x, cond_BD=cond, ca_kv=None, attn_bias_or_two_vector=None,
                    scale_schedule=schedule, scale_ind=si, layer_idx=4, x_shape=schedule[si],
                    all_entropy=history, previous_entropy_mean_attn=1.0,
                    previous_entropy_mean_ca=1.0, previous_entropy_mean_fnn=1.0)
        expected = block(**args)
        actual = shared(**args, scale_entropy=delta.cpu())
        for reference, result in zip(expected[:4], actual[:4]):
            if isinstance(reference, torch.Tensor):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            else:
                assert result == reference
        assert shared.start_ind == block.start_ind == (7 if starts else 100)
        if si == 8 and starts:
            assert all(module.lengths[-1] < width * width for module in [shared.sa, shared.ca, shared.ffn])
            torch.testing.assert_close(actual[4](), expected[4](), rtol=0, atol=0)
            assert all(cache.shape[1] == 20 * 20 for cache in [
                shared.previous_scale_cache_self_attn, shared.previous_scale_cache_cross_attn,
                shared.previous_scale_cache_ffn,
            ])
        if not starts:
            assert all(module.lengths[-1] == width * width for module in [shared.sa, shared.ca, shared.ffn])
            assert actual[4] is None


def test_pruning_ramp_respects_lag():
    delayed = utils.compute_relative_pruning_ratio(10, 7, 0.2, transition_lag=3.0)
    earlier = utils.compute_relative_pruning_ratio(10, 7, 0.2, transition_lag=1.0)
    assert earlier > delayed
