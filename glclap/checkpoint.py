"""Load GLCLAP adapters from either a full training checkpoint or safetensors."""
# =============================================================================
# 中文说明（checkpoint 加载）
#
# 只关心 adapter 相关张量：audio_adapter.* / text_adapter.* / logit_scale，共 13 个张量。
# 支持两种来源：
#   .safetensors —— 发布版权重（20 MiB）
#   .pt          —— 旧版完整训练 checkpoint（4.5 GiB，用 mmap 只读需要的张量）
# =============================================================================
from __future__ import annotations

from pathlib import Path

import torch


ADAPTER_PREFIXES = ("audio_adapter.", "text_adapter.")


def is_adapter_key(key: str) -> bool:
    return key == "logit_scale" or key.startswith(ADAPTER_PREFIXES)


# 读取 adapter 状态字典，并统一去掉可能存在的 "module." 前缀（DDP 保存时会有）。
# 用 mmap=True 避免把旧版 4.5GB checkpoint 整体读进内存。
def load_adapter_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        # mmap prevents optimizer and frozen tower tensors from being read when
        # loading a legacy 4.5GB GLCLAP-Hotword training checkpoint.
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        state = checkpoint.get("model", checkpoint)
    keep = {key.removeprefix("module."): value for key, value in state.items()
            if is_adapter_key(key.removeprefix("module."))}
    if not keep:
        raise ValueError(f"no GLCLAP adapter tensors found in {path}")
    return keep


# 把 adapter 权重注入模型。严格校验：adapter 缺失或出现意外键都直接报错，
# 防止“其实没加载成功”却被静默当成加载成功。
def load_adapters(model: torch.nn.Module, path: str | Path) -> dict[str, torch.Tensor]:
    state = load_adapter_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if is_adapter_key(key)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"invalid adapter checkpoint {path}: missing={bad_missing}, unexpected={unexpected}"
        )
    return state
