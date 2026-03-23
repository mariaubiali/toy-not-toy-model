from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms import log_trafo, softplus_trafo, is_enabled


class BaseFModel(nn.Module):
    """
    Base interface for models that predict f(xgrid) on the grid.
    Return a dict to stay extensible (e.g. BNN can add kl, logvar, etc.).
    """

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class MLPFModel(nn.Module):
    """
    If out_dim=2:
      out[:,0] = f_mean(x)
      out[:,1] = log_var_f(x)
    """

    def __init__(
        self,
        hidden,
        activation="tanh",
        dropout=0.0,
        out_dim=1.0,
        scaling=True,
        init_alpha=1.0,
        init_beta=1.0,
        transforms: dict = {},
    ):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "leakyrelu": nn.LeakyReLU}.get(
            activation.lower()
        )
        dropout = float(dropout)
        if act is None:
            raise ValueError(f"Unknown activation: {activation}")
        if scaling is True:
            self.scaling_mode = "scaling"
        elif scaling:
            self.scaling_mode = str(scaling).lower()
        else:
            self.scaling_mode = None
        self.logalpha = nn.Parameter(torch.log(torch.tensor(float(init_alpha))))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(float(init_beta))))
        self.transforms = transforms or {}

        # print("scaling mode: ", self.scaling_mode)

        layers: List[nn.Module] = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, int(h)), act(), nn.Dropout(dropout)]
            in_dim = int(h)

        layers += [nn.Linear(in_dim, int(out_dim))]
        self.net = nn.Sequential(*layers)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ab_min = 1e-6
        x_clip = 1e-12

        # Keep physical x for endpoint pre-factor
        x_phys = x.squeeze(1).clamp(x_clip, 1.0 - x_clip)  # (N,)

        # Apply transforms to input (for the MLP only)
        x_in = log_trafo(x, self.transforms)  # still (N,1)

        raw = self.net(x_in)  # (N, out_dim)
        s = raw[:, 0].reshape(-1)  # latent mean (N,)

        # Endpoint factor (MUST use physical x)
        if (self.scaling_mode is not None) and self.scaling_mode != "none":
            alpha = torch.exp(self.logalpha).clamp_min(ab_min)

            if self.scaling_mode == "xalpha":
                pre = x_phys.pow(alpha)
                # print("scaling xalpha")
            else:  # "scaling"
                beta = torch.exp(self.logbeta).clamp_min(ab_min)
                pre = x_phys.pow(alpha) * (1.0 - x_phys).pow(beta)
                # print("nn scaling xalpha 1-xbeta")
        else:
            pre = None
            # print("no scaling")

        # Mean transform (SoftplusOut if enabled)
        g = softplus_trafo(s, self.transforms)  # (N,)

        f = pre * g if pre is not None else g
        out: Dict[str, torch.Tensor] = {"f_grid": f}

        # Heteroscedastic head
        if raw.shape[1] >= 2:
            logvar_s = raw[:, 1].reshape(-1)  # (N,)

            # If Softplus is enabled, map variance from latent -> g-space
            # Var(g) ≈ (sigmoid(s))^2 Var(s)
            if is_enabled(self.transforms, "Softplus"):
                dg_ds = torch.sigmoid(s).clamp_min(1e-12)
                logvar_g = logvar_s + 2.0 * torch.log(dg_ds)
            else:
                logvar_g = logvar_s

            # Now map variance through endpoint factor: f = pre * g
            if pre is not None:
                out["logvar_f_grid"] = logvar_g + 2.0 * torch.log(pre.clamp_min(1e-12))
            else:
                out["logvar_f_grid"] = logvar_g

        return out
