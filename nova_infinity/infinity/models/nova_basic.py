"""NOVA attention and transformer blocks with entropy-guided token selection."""

import math
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from flash_attn import flash_attn_func

from infinity.models.basic import CrossAttention, FFN, FFNSwiGLU, get_dropout_layer
from infinity.models.nova_utils import compute_merge, compute_relative_pruning_ratio


def apply_rotary_emb_selected(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind, rope_idx_fn=None,seq_len=0):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = sum(item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        if rope_idx_fn is not None and rope_idx_fn.__name__ != 'do_nothing':
            rope_idx = rope_idx_fn()
            batch, kept = rope_idx.shape[:2]
            rope_cache = rope_cache.expand(-1, -1, batch, -1, -1, -1)
            gather_idx = rope_idx.reshape(1, 1, batch, 1, kept, 1)
            gather_idx = gather_idx.expand(2, 1, batch, 1, kept, rope_cache.shape[-1])
            rope_cache = torch.gather(rope_cache, dim=4, index=gather_idx)
        qk = qk.reshape(*qk.shape[:-1], -1, 2) # (2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)
    return q, k


class NOVASelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0,rope_idx=None,ori_len=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb_selected(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind, rope_idx_fn=rope_idx, seq_len=ori_len) #, freqs_cis=freqs_cis)
        if not self.use_flex_attn:
            # FlashAttention consumes K in V's dtype; keep the cache in that dtype too.
            k = k.to(v.dtype)
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim) # 10,521
        
        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                # flashattn
                q, k, v = q.transpose(1,2), k.transpose(1,2),v.transpose(1,2)
                oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale).reshape(B, L, C)

            # oup: bf16
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


class NOVACrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
    ):
        super(NOVACrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = NOVASelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only

        self.nova_enabled = True
        self.nova_entropy_threshold = None
        self._reset(())

    def _reset(self, scale_schedule):
        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_cross_attn = None
        self.previous_scale_cache_ffn = None
        self.label_cache = [shape[2] for shape in scale_schedule]
        self.cached_size = []
        self.purne_scale = []
        self.now_cached_size = [16, 16]
        self.entropy_threshold = -1
        self.start_ind = 100

    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, layer_idx=-1, x_shape=None, all_entropy=None, 
        previous_entropy_mean_attn=0.0,
        previous_entropy_mean_ca=0.0,
        previous_entropy_mean_fnn=0.0,
        scale_entropy=None,
    ):
        if scale_ind == 0:
            self._reset(scale_schedule)
        puring_ratio=0.0
        # Scale statistics are shared by all layers. The inference loop computes
        # them once and transfers only the scalar needed by the Python schedule.
        if scale_entropy is None:
            if scale_ind > 6:
                recent = [value.float().mean() for value in all_entropy[-3:]]
                scale_entropy = ((recent[2] - recent[1]).abs() + (recent[1] - recent[0]).abs()).cpu()
            else:
                scale_entropy = 3.0
        if self.nova_enabled and self.entropy_threshold == -1 and scale_ind > 6:
            if scale_entropy < self.nova_entropy_threshold:
                self.cached_size = self.label_cache[scale_ind:-1]
                self.purne_scale = self.label_cache[scale_ind + 1:]
                self.start_ind = scale_ind
                cache_size = self.label_cache[scale_ind]
                self.now_cached_size = [cache_size, cache_size]
                self.entropy_threshold = scale_entropy
        if scale_ind > self.start_ind:
            puring_ratio = compute_relative_pruning_ratio(
                scale_ind, self.start_ind, scale_entropy,
                transition_lag=None,
                temperature=0.8,
                entropy_sensitivity=0.1,
            )

        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        is_later_layer = 3 <= layer_idx < 28
        cache_residual = is_later_layer and x.shape[1] in [item*item for item in self.cached_size]
        merge_fn, unmerge_fn, idx_fn, new_e_sa = compute_merge(x, prune_scale_list=self.purne_scale, is_later_layer=is_later_layer,x_shape=x_shape, puring_ratio=puring_ratio, previous_e_mean=previous_entropy_mean_attn)
        shortcut = x

        re_idx_fn = idx_fn

        x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merge_fn(x), scale=scale1, shift=shift1)
        x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind,rope_idx=idx_fn,ori_len=shortcut.shape[1]).mul_(gamma1)

        x_sa = unmerge_fn(x_sa,self.previous_scale_cache_self_attn,self.now_cached_size)
        if cache_residual:
            self.previous_scale_cache_self_attn = x_sa
        x = shortcut + self.drop_path(x_sa)

        merge_fn, unmerge_fn, idx_fn, new_e_ca = compute_merge(x, prune_scale_list=self.purne_scale, is_later_layer=is_later_layer,x_shape=x_shape, puring_ratio=puring_ratio, previous_e_mean=previous_entropy_mean_ca)
        x_ca = unmerge_fn(self.ca(self.ca_norm(merge_fn(x)), ca_kv).float().mul_(self.ca_gamma),self.previous_scale_cache_cross_attn, self.now_cached_size)
        if cache_residual:
            self.previous_scale_cache_cross_attn = x_ca
        x = x + x_ca

        merge_fn, unmerge_fn, idx_fn, new_e_fnn = compute_merge(x, prune_scale_list=self.purne_scale, is_later_layer=is_later_layer,x_shape=x_shape, puring_ratio=puring_ratio, previous_e_mean=previous_entropy_mean_fnn)
        x_ffn = unmerge_fn(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merge_fn(x), scale=scale2, shift=shift2)).mul(gamma2),self.previous_scale_cache_ffn,self.now_cached_size)
        if cache_residual:
            self.previous_scale_cache_ffn = x_ffn
            s = math.isqrt(x.shape[1])
            self.now_cached_size = [s, s]


        x = x + self.drop_path(x_ffn) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP

        return x, new_e_sa, new_e_ca, new_e_fnn, re_idx_fn


    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'
