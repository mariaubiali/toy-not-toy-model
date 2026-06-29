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
            print("In no scaling!")
        # alpha is stored directly so fixed or trained scans can include
        # negative values, e.g. alpha=-0.5. Do not use log(alpha) here.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
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
            alpha = self.alpha
            if self.scaling_mode == "xalpha":
                pre = x_phys.pow(alpha)
                # print("scaling xalpha")
            else:  # "scaling"
                beta = self.beta
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

class StackedLinear(nn.Module):
    """
    Linear layer for K ensemble members trained in parallel.

    Input:
        x with shape (K, B, in_features)

    Output:
        y with shape (K, B, out_features)
    """

    def __init__(self, in_features, out_features, channels, init="kaiming"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.channels = int(channels)
        self.init = str(init).lower()

        self.weight = nn.Parameter(
            torch.empty(self.channels, self.out_features, self.in_features)
        )
        self.bias = nn.Parameter(torch.empty(self.channels, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        import math

        if self.init == "same":
            torch.nn.init.kaiming_uniform_(self.weight[0], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            torch.nn.init.uniform_(self.bias[0], -bound, bound)

            for i in range(1, self.channels):
                with torch.no_grad():
                    self.weight[i].copy_(self.weight[0])
                    self.bias[i].copy_(self.bias[0])

        elif self.init == "kaiming":
            for i in range(self.channels):
                torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                torch.nn.init.uniform_(self.bias[i], -bound, bound)
        else:
            raise ValueError(f"Unknown StackedLinear init={self.init!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"StackedLinear expects x with shape (K,B,in_features), got {tuple(x.shape)}"
            )
        return torch.baddbmm(
            self.bias[:, None, :],
            x,
            self.weight.transpose(1, 2),
        )


class RepulsiveMLPFModel(nn.Module):
    """
    T3 MLP ensemble trained in parallel with stacked weights.

    The returned dictionary mirrors MLPFModel, but the leading axis is the
    ensemble-member axis:

        f_grid:         (K, Ngrid)
        logvar_f_grid:  (K, Ngrid), only if out_dim >= 2

    This lets the training code map every member through the same FK table W.
    """

    def __init__(
        self,
        hidden,
        channels,
        activation="tanh",
        dropout=0.0,
        out_dim=1,
        scaling=True,
        init_alpha=1.0,
        init_beta=1.0,
        transforms: dict = {},
        init="kaiming",
    ):
        super().__init__()

        act = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
        }.get(activation.lower())
        if act is None:
            raise ValueError(f"Unknown activation: {activation}")

        self.channels = int(channels)
        self.out_dim = int(out_dim)
        self.transforms = transforms or {}

        if scaling is True:
            self.scaling_mode = "scaling"
        elif scaling:
            self.scaling_mode = str(scaling).lower()
        else:
            self.scaling_mode = None

        # Shared endpoint exponents. This keeps the same physics-inspired
        # prefactor convention as MLPFModel.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        # beta is kept positive through a logarithmic parameterisation.
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

        layers: List[nn.Module] = []
        in_dim = 1
        for h in hidden:
            layers.append(StackedLinear(in_dim, int(h), self.channels, init=init))
            layers.append(act())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)

        layers.append(StackedLinear(in_dim, self.out_dim, self.channels, init=init))
        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ab_min = 1e-6
        x_clip = 1e-12

        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"Expected x with shape (Ngrid,1), got {tuple(x.shape)}")

        x_phys = x.squeeze(1).clamp(x_clip, 1.0 - x_clip)  # (Ngrid,)
        x_in = log_trafo(x, self.transforms)               # (Ngrid,1)
        x_in = x_in[None, :, :].expand(self.channels, x_in.shape[0], x_in.shape[1])

        raw = x_in
        for layer in self.net:
            raw = layer(raw)

        # raw: (K, Ngrid, out_dim)
        s = raw[..., 0]  # (K, Ngrid)

        if (self.scaling_mode is not None) and self.scaling_mode != "none":
            alpha = self.alpha

            if self.scaling_mode == "xalpha":
                pre = x_phys.pow(alpha)  # (Ngrid,)
            else:
                beta = self.beta
                pre = x_phys.pow(alpha) * (1.0 - x_phys).pow(beta)
        else:
            pre = None

        g = softplus_trafo(s, self.transforms)  # (K, Ngrid)
        f = pre[None, :] * g if pre is not None else g

        out: Dict[str, torch.Tensor] = {"f_grid": f}

        if raw.shape[-1] >= 2:
            logvar_s = raw[..., 1]  # (K, Ngrid)

            if is_enabled(self.transforms, "Softplus"):
                dg_ds = torch.sigmoid(s).clamp_min(1e-12)
                logvar_g = logvar_s + 2.0 * torch.log(dg_ds)
            else:
                logvar_g = logvar_s

            if pre is not None:
                out["logvar_f_grid"] = logvar_g + 2.0 * torch.log(
                    pre[None, :].clamp_min(1e-12)
                )
            else:
                out["logvar_f_grid"] = logvar_g

        return out