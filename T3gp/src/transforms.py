from __future__ import annotations

import torch
import torch.nn.functional as F

# Helper function to check the functions from config

def is_enabled(transforms: dict, key: str) -> bool:
    """
    Check if a transform is enabled in the config.
    Enabled-by-presence:
      transforms:
        Logx: {}        -> enabled
      transforms: {}     -> disabled
      transforms missing -> disabled

    Optional override:
      Logx: {enabled: false} -> disabled
    """

    if not isinstance(transforms, dict):
        return False
    if key not in transforms:
        return False

    cfg = transforms.get(key, {})
    if isinstance(cfg, dict) and "enabled" in cfg:
        return bool(cfg["enabled"])
    return True

def log_trafo(
    x: torch.Tensor,
    transforms: dict | None,
) -> torch.Tensor:
    """
    Apply input transforms to x according to the config.
    """
    x_clip = 1e-12

    if is_enabled(transforms, "Logx"):
        x = x.clamp(x_clip, 1.0 - x_clip)
        x = torch.log(x)

    return x

def softplus_trafo(
    f: torch.Tensor,
    transforms: dict | None,
) -> torch.Tensor:
    """
    Apply output transforms to y according to the config.
    """

    if is_enabled(transforms, "Softplus"):
        f = F.softplus(f)
    
    return f