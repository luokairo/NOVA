"""NOVA entropy scoring, pruning schedule, selection, and residual restoration."""

import math
from typing import Callable, Tuple

import torch
from torch.nn import functional as F


def compute_relative_pruning_ratio(scale_ind, start_ind, entropy_diff, transition_lag=1.0, temperature=0.8, entropy_sensitivity=0.1):
    """Return the entropy-damped scale pruning ratio, clipped to [0, .95]."""
    if start_ind == 100 or scale_ind < start_ind:
        return 0.0
    distance = scale_ind - start_ind
    base_ratio = 1.0 / (1.0 + math.exp(-(distance - transition_lag) / temperature))
    entropy_diff = float(entropy_diff)
    damping = entropy_sensitivity * math.tanh(max(0, entropy_diff))
    return max(0.0, min(0.95, base_ratio - damping))


def compute_layer_entropy(cur_x: torch.Tensor, eps: float = 1e-6):
    """Return normalized channel-energy entropy per token and its spatial mean.

    This is a feature-based proxy, not entropy over predicted token classes.
    Keep the original scoring formula, including its epsilon placement.
    """
    energy = cur_x.float().square().clamp_min_(eps)
    probability = energy.div_(energy.sum(dim=-1, keepdim=True).clamp_min_(eps))
    entropy = -(probability * (probability + eps).log()).sum(dim=-1, keepdim=True)
    entropy = entropy / math.log(cur_x.shape[-1] + 1e-12)
    return {"ent": entropy, "ent_mean": entropy.mean(dim=1)}


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def masked_previous_scale_cache_dynamic(cur_x, num_remain, cur_shape, pre_mean):
    """Select high-entropy tokens and fill skipped residuals from the prior scale."""
    batch, length, channels = cur_x.shape
    statistics = compute_layer_entropy(cur_x)
    entropy_mean = statistics["ent_mean"]
    relative_change = (entropy_mean - pre_mean) / (entropy_mean + pre_mean)
    # A Python token count is required by topk; retain this one scalar sync.
    multiplier = relative_change.mean().item() + 1
    num_remain = max(0, min(math.ceil(num_remain * multiplier), length))
    updated_mean = 0.5 * (entropy_mean + pre_mean)

    indices = torch.topk(
        statistics["ent"].squeeze(-1), num_remain, dim=1,
        largest=True, sorted=False,
    ).indices.unsqueeze(-1)
    gather_indices = indices.expand(-1, -1, channels)

    def merge(x):
        return torch.gather(x, dim=1, index=gather_indices)

    def unmerge(x_small, x_cache, cached_hw=None):
        cached_height, cached_width = cached_hw
        residual = x_cache.view(batch, cached_height, cached_width, channels).permute(0, 3, 1, 2)
        residual = F.interpolate(residual, size=cur_shape[1:], mode="area")
        residual = residual.permute(0, 2, 3, 1).reshape(batch, length, channels)
        # Interpolation creates fresh storage; a second full-sized clone is unnecessary.
        residual.scatter_(dim=1, index=gather_indices, src=x_small)
        return residual

    def get_src_tgt_idx():
        return indices

    return merge, unmerge, get_src_tgt_idx, updated_mean


def compute_merge(
    x: torch.Tensor,
    prune_scale_list=(),
    is_later_layer=False,
    x_shape=None,
    puring_ratio=0.0,
    previous_e_mean=0.0,
) -> Tuple[Callable, ...]:
    """Build token selection/restoration functions for an eligible submodule."""
    if x_shape[2] in prune_scale_list and is_later_layer:
        num_remain = x.shape[1] - int(x.shape[1] * puring_ratio)
        return masked_previous_scale_cache_dynamic(x, num_remain, x_shape, previous_e_mean)
    return do_nothing, do_nothing, None, 0.0
