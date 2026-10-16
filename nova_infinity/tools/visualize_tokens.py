"""Optional token-pruning overlays, produced after inference."""
import os
import os.path as osp

import cv2
import numpy as np
import torch

def _to_keep_idx(idx_fn, b_select=0):
    """
    idx_fn: 可能是 tensor / callable / None / do_nothing
    返回: keep_idx: shape (M,) 的 numpy int64（对应该尺度 flatten 后的位置）
    """
    if idx_fn is None:
        return None

    # 有些实现会返回 callable（你的 masked_previous_scale_cache_dynamic 里 id_fn 是 get_src_tgt_idx）
    if callable(idx_fn):
        idx = idx_fn()
    else:
        idx = idx_fn

    if idx is None:
        return None

    # idx 可能是 torch.Tensor
    try:
        import torch
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
    except Exception:
        pass

    idx = np.asarray(idx)

    # 常见形状：(B, M, 1) / (B, M) / (M, 1) / (M,)
    if idx.ndim == 3:
        idx = idx[b_select, :, 0]
    elif idx.ndim == 2:
        # (B, M) -> 取第 b_select 个
        if idx.shape[0] > 1:
            idx = idx[b_select, :]
        else:
            idx = idx[0, :]
    elif idx.ndim == 1:
        pass
    else:
        return None

    idx = idx.astype(np.int64)
    return idx


def build_drop_mask_hw(keep_idx, h, w):
    """
    keep_idx:
      - None        → 不启用 pruning，不 mask
      - empty array → 启用了 pruning，但一个都没保留 → 全 mask
      - (M,)        → 正常情况
    返回:
      drop_mask_hw: (h, w) bool, True 表示被 mask（丢弃）
    """
    L = h * w

    # 🚩 情况 1：没有 pruning（None）→ 不 mask
    if keep_idx is None:
        return np.zeros((h, w), dtype=bool)

    # 🚩 情况 2：prune 但没留下 → 全 mask
    if keep_idx.size == 0:
        return np.ones((h, w), dtype=bool)

    keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < L)]
    keep_mask = np.zeros(L, dtype=bool)
    keep_mask[keep_idx] = True

    drop_mask = ~keep_mask
    return drop_mask.reshape(h, w)


def overlay_drop_mask_on_image(img_uint8, drop_mask_hw, alpha=0.6):
    """
    img_uint8: (H,W,3) uint8
    drop_mask_hw: (h,w) bool
    输出: overlay_img uint8
    这里用“变暗”方式标出被丢弃 token 覆盖到最终图的位置（不依赖 colormap）
    """
    H, W = img_uint8.shape[:2]
    mask_up = cv2.resize(drop_mask_hw.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    out = img_uint8.copy()
    # 变暗：out = out * (1-alpha) 只在 mask 区域
    out[mask_up] = (out[mask_up].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return out, mask_up


def visualize_dropped_tokens_per_scale(generated_image, remain_ind, scale_schedule, save_dir,
                                      b_select=0, alpha=0.6, prefix="dropviz"):
    """
    generated_image: torch 或 numpy, (H,W,3) uint8 或 float[0,255]
    remain_ind: list，长度≈尺度数（你现在是在 layer_idx==12 append）
    scale_schedule: list of pn, pn=(1,h,w)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 转成 uint8 HWC
    try:
        import torch
        if isinstance(generated_image, torch.Tensor):
            img = generated_image.detach().cpu().numpy()
        else:
            img = np.asarray(generated_image)
    except Exception:
        img = np.asarray(generated_image)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]
    overlays = []

    n = min(len(scale_schedule), len(remain_ind))
    for si in range(n):
        pn = scale_schedule[si]
        h, w = int(pn[1]), int(pn[2])

        keep_idx = _to_keep_idx(remain_ind[si], b_select=b_select)
        drop_mask_hw = build_drop_mask_hw(keep_idx, h, w)

        overlay, mask_up = overlay_drop_mask_on_image(img, drop_mask_hw, alpha=alpha)

        # 可选：在图上标一下尺度信息
        # text = f"si={si}, hw={h}x{w}, drop={(mask_up.sum())/(H*W):.2%}"
        # cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        out_path = osp.join(save_dir, f"{prefix}_si{si:02d}_{h}x{w}.png")
        cv2.imwrite(out_path, overlay)
        overlays.append(overlay)

    return overlays

