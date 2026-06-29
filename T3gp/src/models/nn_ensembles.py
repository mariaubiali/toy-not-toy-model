from __future__ import annotations
from typing import Any, Dict
import copy
import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import trange

from losses import (
    pointwise_loss_members,
    _cholesky_C,
    _apply_Cinv,
)
from models.nn_models import RepulsiveMLPFModel
from models.nn_train import select_training_target, train_nn_forward


def _repulsive_kernel(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    kind: str = "rbf",
    sigma=None,
    xgrid: torch.Tensor | None = None,
    gibbs_l_min: float = 0.1,
    gibbs_l_max: float = 10.0,
    gibbs_x0: float = 0.01,
    gibbs_power: float = 1.0,
) -> torch.Tensor:
    """
    Repulsive kernel between ensemble members.

    x, y:
        Tensors with shape (K, Nfeatures).

    kind:
        "rbf" or "gibbs".

    For kind="rbf":
        K_ij = exp(-||x_i-y_j||^2 / (2 sigma))

    For kind="gibbs":
        K_ij = exp(-0.5 * sum_m (x_im-y_jm)^2 / ell(x_m)^2)

    The Gibbs version is a weighted, non-stationary RBF with an x-dependent
    length scale ell(x).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (K,Nfeatures), got {tuple(x.shape)}")

    if y is None:
        y = x.detach()

    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (K,Nfeatures), got {tuple(y.shape)}")

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}"
        )

    kind = str(kind).lower()
    channels = x.shape[0]

    diff2 = (
        x.reshape(channels, 1, -1)
        - y.reshape(1, channels, -1)
    ).square()

    if kind == "rbf":
        dnorm2 = diff2.sum(dim=2)

        if sigma is None:
            sigma_t = torch.quantile(dnorm2.detach(), 0.5)
            sigma_t = sigma_t / (2.0 * math.log(channels + 1.0))
            sigma_t = sigma_t.clamp_min(1.0e-12)
        else:
            sigma_t = torch.as_tensor(
                float(sigma),
                dtype=x.dtype,
                device=x.device,
            ).clamp_min(1.0e-12)

        return torch.exp(-dnorm2 / (2.0 * sigma_t))

    if kind == "gibbs":
        if xgrid is None:
            raise ValueError("Gibbs kernel requires xgrid when kind='gibbs'.")

        xgrid = xgrid.reshape(-1).to(dtype=x.dtype, device=x.device)

        if xgrid.shape[0] != x.shape[1]:
            raise ValueError(
                f"For Gibbs kernel, xgrid length must match feature dimension. "
                f"Got xgrid={xgrid.shape[0]} and features={x.shape[1]}."
            )

        # Use log-x coordinate because your grid spans many orders of magnitude.
        logx = torch.log(xgrid.clamp_min(1.0e-12))

        logx0 = math.log(float(gibbs_x0))
        logxmin = torch.min(logx)
        logxmax = torch.max(logx)

        # Smooth coordinate in [0, 1].
        t = (logx - logxmin) / (logxmax - logxmin).clamp_min(1.0e-12)

        # Optional pivot around gibbs_x0.
        t0 = (torch.as_tensor(logx0, dtype=x.dtype, device=x.device) - logxmin) / (
            logxmax - logxmin
        ).clamp_min(1.0e-12)

        # Length scale profile. This makes ell vary smoothly across x.
        # power > 1 makes the transition sharper.
        profile = torch.abs(t - t0).pow(float(gibbs_power))
        profile = profile / profile.max().clamp_min(1.0e-12)

        ell = float(gibbs_l_max) - (float(gibbs_l_max) - float(gibbs_l_min)) * profile

        ell2 = ell.square().clamp_min(1.0e-12)

        dweighted = (diff2 / ell2.reshape(1, 1, -1)).sum(dim=2)

        return torch.exp(-0.5 * dweighted)

    raise ValueError(f"Unknown repulsive kernel kind: {kind!r}")


def _repulsive_weight_norm(model: torch.nn.Module) -> torch.Tensor:
    """
    Squared L2 norm of all trainable parameters in the repulsive ensemble model.

    This is used for the optional Gaussian-prior/weight-decay-like term

        ||theta||^2 / (2 N prior_width).
    """
    norm = None

    for param in model.parameters():
        if not param.requires_grad:
            continue

        term = param.square().sum()
        norm = term if norm is None else norm + term

    if norm is None:
        first_param = next(model.parameters())
        norm = torch.zeros((), dtype=first_param.dtype, device=first_param.device)

    return norm


def train_repulsive_nn_forward(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a repulsive NN ensemble for the T3 FK-table setup.

    This is different from train_nn_ensemble_forward: all K members are trained
    simultaneously as channels of one StackedLinear network.
    """
    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32

    nncfg = cfg.get("nn", {})
    ens_cfg = nncfg.get("ensemble", {})
    rep_cfg = nncfg.get("repulsive", {})

    hidden = nncfg.get("hidden", [64, 64])
    activation = str(nncfg.get("activation", "tanh"))
    lr = float(nncfg.get("lr", 1e-3))
    epochs = int(nncfg.get("epochs", 3000))
    weight_decay = float(nncfg.get("weight_decay", 0.0))
    seed = int(cfg.get("seed", 0))
    dropout = float(nncfg.get("dropout", 0.0))
    scaling = nncfg.get("scaling", True)
    init_alpha = float(nncfg.get("init_alpha", 1.0))
    init_beta = float(nncfg.get("init_beta", 3.0))
    transforms = cfg.get("transforms", {})
    out_dim = int(nncfg.get("out_dim", 1))
    kernel_kind = str(rep_cfg.get("kernel", rep_cfg.get("kernel_kind", "rbf"))).lower()
    gibbs_l_min = float(rep_cfg.get("gibbs_l_min", 0.1))
    gibbs_l_max = float(rep_cfg.get("gibbs_l_max", 10.0))
    gibbs_x0 = float(rep_cfg.get("gibbs_x0", 0.01))
    gibbs_power = float(rep_cfg.get("gibbs_power", 1.0))
    if out_dim != 1:
        raise NotImplementedError(
            "The first T3 repulsive implementation is intentionally restricted "
            "to out_dim=1. Add heteroscedastic propagation only after the scalar "
            "case is validated."
        )

    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    jitter = float(cfg.get("kernel", {}).get("jitter", cfg.get("loss", {}).get("jitter", 1e-10)))

    channels = int(
        rep_cfg.get(
            "channels",
            ens_cfg.get("channels", ens_cfg.get("n_members", ens_cfg.get("members", 20))),
        )
    )
    beta = float(rep_cfg.get("beta", 1.0))
    prior_width = float(rep_cfg.get("prior_width", 1.0))
    h = rep_cfg.get("h", None)
    kernel_space = str(rep_cfg.get("kernel_space", "xt3")).lower()
    if kernel_space not in {"xt3", "model", "f", "function", "loss", "data", "y"}:
        raise ValueError(
            "nn.repulsive.kernel_space must be one of "
            "'xt3'/'model'/'f'/'function' or 'loss'/'data'/'y', "
            f"got {kernel_space!r}"
        )
    init = str(rep_cfg.get("init", "kaiming"))

    patience = int(nncfg.get("patience", 500))
    min_delta = float(nncfg.get("min_delta", 0.0))
    lambda_sr = float(cfg.get("loss", {}).get("lambda_sr", 0.0))

    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = ds.get("meta", {})
    xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()
    xgrid_ext = (
        meta.get("xgrid_ext").astype(np.float64)
        if "xgrid_ext" in meta
        else np.array([], dtype=np.float64)
    )

    xgrid = ds["xgrid"].astype(np.float32)
    W = ds["W"].astype(np.float32)
    C = ds["C"].astype(np.float32)
    y = select_training_target(ds, cfg).astype(np.float32)

    n_data = W.shape[0]
    n_grid = xgrid.shape[0]
    if xt3_true.size != n_grid:
        raise ValueError(
            f"xt3_true must be defined on xgrid: got {xt3_true.size} vs Ngrid={n_grid}"
        )

    t3_ref_int = float(np.trapz(xt3_true / xgrid, xgrid))

    x_torch = torch.tensor(xgrid, dtype=dtype, device=device).unsqueeze(1)
    xgrid_1d = x_torch.squeeze(1)
    W_torch = torch.tensor(W, dtype=dtype, device=device)
    C_torch = torch.tensor(C, dtype=dtype, device=device)
    y_torch = torch.tensor(y, dtype=dtype, device=device)

    xext_torch = torch.tensor(xgrid_ext, dtype=dtype, device=device).unsqueeze(1)
    x_pred_torch = xext_torch if xgrid_ext.size > 0 else x_torch
    x_pred_np = xgrid_ext.astype(np.float64) if xgrid_ext.size > 0 else xgrid.astype(np.float64)

    idx_all = np.arange(n_data)
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=seed,
    )

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    W_tr = W_torch[train_idx_t, :]
    W_val = W_torch[val_idx_t, :]
    y_tr = y_torch[train_idx_t]
    y_val = y_torch[val_idx_t]
    C_tr = C_torch[train_idx_t][:, train_idx_t]
    C_val = C_torch[val_idx_t][:, val_idx_t]

    L_tr = _cholesky_C(C_tr, jitter)
    L_val = _cholesky_C(C_val, jitter)

    model = RepulsiveMLPFModel(
        hidden=hidden,
        channels=channels,
        activation=activation,
        dropout=dropout,
        out_dim=out_dim,
        scaling=scaling,
        init_alpha=init_alpha,
        init_beta=init_beta,
        transforms=transforms,
        init=init,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_hist = []
    val_chi2_hist = []
    log_every = max(1, epochs // 20)

    best_val_chi2 = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = epochs - 1

    for ep in trange(epochs, desc="Training repulsive ensemble"):
        model.train()
        opt.zero_grad()

        out = model(x_torch)
        f_members = out["f_grid"]  # (K, Ngrid)

        y_pred_members = f_members @ W_tr.T  # (K, Ntrain)
        per_point_loss = pointwise_loss_members(
            y_pred_members,
            y_tr,
            loss_name=loss_name,
            C=C_tr,
            L=L_tr,
            jitter=jitter,
        )

        if loss_name == "chi2":
            data_loss = per_point_loss.sum(dim=1)
        else:
            data_loss = per_point_loss.mean(dim=1)

        # if lambda_sr > 0.0:
        #     I_members = torch.trapz(f_members / xgrid_1d[None, :], xgrid_1d, dim=1)
        #     sr_loss = lambda_sr * (I_members - t3_ref_int).square()
        #     data_loss = data_loss + sr_loss

        if kernel_space in {"xt3", "model", "f", "function"}:
            kernel_input = f_members
        else:
            kernel_input = per_point_loss

        kernel = _repulsive_kernel(
            kernel_input,
            kernel_input.detach(),
            kind=kernel_kind,
            sigma=h,
            xgrid=xgrid_1d if kernel_space in {"xt3", "model", "f", "function"} else None,
            gibbs_l_min=gibbs_l_min,
            gibbs_l_max=gibbs_l_max,
            gibbs_x0=gibbs_x0,
            gibbs_power=gibbs_power,
        )

        if loss_name == "chi2":
            repulsive_prefactor = beta
        else:
            repulsive_prefactor = beta / float(n_data)

        if ep == 0:
            print(f"Repulsive prefactor = {repulsive_prefactor:g} for loss={loss_name!r}")

        repulsive_term = repulsive_prefactor * (
            kernel.sum(dim=1)
            / kernel.detach().sum(dim=1).clamp_min(1e-12)
            - 1.0
        )

        loss = torch.sum(data_loss + repulsive_term)

        if prior_width > 0.0:
            loss = loss + (
                1.0
                / (2.0 * float(n_data) * prior_width)
                * _repulsive_weight_norm(model)
            )

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_v = model(x_torch)
            f_mean_v = out_v["f_grid"].mean(dim=0)
            y_pred_val = W_val @ f_mean_v
            r_val = (y_pred_val - y_val).reshape(-1)
            chi2_val = float(r_val @ _apply_Cinv(L_val, r_val))
            chi2_val_pt = chi2_val / float(len(val_idx))

        val_chi2_hist.append(float(chi2_val_pt))

        if chi2_val_pt < best_val_chi2 - min_delta:
            best_val_chi2 = float(chi2_val_pt)
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # if patience > 0 and epochs_without_improvement >= patience:
        #     stopped_early = True
        #     stopped_epoch = ep
        #     break

        if (ep + 1) % log_every == 0:
            loss_hist.append(float(loss.detach().cpu().item()))

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_full = model(x_pred_torch)
        replicas = out_full["f_grid"].detach().cpu().numpy().astype(np.float64)

    res = _summarise_replicas(
        xgrid=x_pred_np,
        replicas=replicas,
        vars_replicas=None,
        member_losses=[np.array(loss_hist, dtype=np.float64)],
        save_member_preds=True,
        member_means=[replicas[k] for k in range(replicas.shape[0])],
    )

    res.update(
        {
            "ensemble_kind": "repulsive",
            "repulsive": {
                "channels": int(channels),
                "beta": float(beta),
                "prior_width": float(prior_width),
                "h": None if h is None else float(h),
                "kernel_space": kernel_space,
                "init": init,
            },
            "f_grid_mean": res["mean_curve"],
            "loss_history": np.array(loss_hist, dtype=np.float64),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "cv": {
                "enabled": True,
                "val_fraction": 0.2,
                "split_seed": int(seed),
                "patience": int(patience),
                "min_delta": float(min_delta),
                "best_epoch": int(best_epoch),
                "best_val_chi2_per_point": float(best_val_chi2)
                if np.isfinite(best_val_chi2)
                else None,
                "stopped_early": bool(stopped_early),
                "stopped_epoch": int(stopped_epoch),
                "val_chi2_history": np.array(val_chi2_hist, dtype=np.float64),
            },
        }
    )

    return res


def _as_uniform_range(value, *, name: str):
    """
    Return (low, high) for a two-value config range, or None if disabled.

    Expected form:
        [low, high]
    """
    if value is None:
        return None

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list/tuple [low, high], got {value!r}")

    low, high = float(value[0]), float(value[1])

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{name} values must be finite, got {value!r}")

    if low > high:
        raise ValueError(f"{name} must satisfy low <= high, got {value!r}")

    return low, high


def _sample_l2_scaling_init(cfg: Dict[str, Any], *, fit_seed: int):
    """
    Optionally sample L2 chi2 endpoint-scaling initial values.

    Config:
        nn:
          l2_scaling_init:
            enabled: true
            alpha_range: [0.0, 1.1]
            beta_range: [1.0, 5.0]

    Missing ranges leave the corresponding nn.init_alpha/init_beta unchanged.

    The sampled values are only used as initial values. Alpha and beta remain
    trainable unless the model/freezing logic elsewhere disables gradients.
    """
    nncfg = cfg.get("nn", {})
    init_cfg = nncfg.get("l2_scaling_init", {})

    if not bool(init_cfg.get("enabled", False)):
        return None, None

    loss_name = str(cfg.get("loss", {}).get("name", "")).lower()
    if loss_name != "chi2":
        return None, None

    rng = np.random.default_rng(int(fit_seed))

    alpha_range = _as_uniform_range(
        init_cfg.get("alpha_range", init_cfg.get("alpha", None)),
        name="nn.l2_scaling_init.alpha_range",
    )
    beta_range = _as_uniform_range(
        init_cfg.get("beta_range", init_cfg.get("beta", None)),
        name="nn.l2_scaling_init.beta_range",
    )

    alpha = None if alpha_range is None else float(rng.uniform(alpha_range[0], alpha_range[1]))
    beta = None if beta_range is None else float(rng.uniform(beta_range[0], beta_range[1]))

    return alpha, beta


def _summarise_replicas(
    *,
    xgrid,
    replicas,
    vars_replicas=None,
    member_losses=None,
    save_member_preds=False,
    member_means=None,
    post_ntk_replicas=None,
    post_ntk_vars=None,
):
    res = {
        "xgrid": xgrid,
        "replicas": replicas,
        "vars_replicas": vars_replicas,
        "mean_curve": replicas.mean(axis=0),
        "lo68": np.percentile(replicas, 16.0, axis=0),
        "hi68": np.percentile(replicas, 84.0, axis=0),
        "lo95": np.percentile(replicas, 2.5, axis=0),
        "hi95": np.percentile(replicas, 97.5, axis=0),
        "member_losses": member_losses if member_losses is not None else [],
        "post_ntk_replicas": post_ntk_replicas,
        "post_ntk_vars_replicas": post_ntk_vars,
    }

    if vars_replicas is not None:
        res["var_mean_curve"] = vars_replicas.mean(axis=0)
        res["var_lo68"] = np.percentile(vars_replicas, 16.0, axis=0)
        res["var_hi68"] = np.percentile(vars_replicas, 84.0, axis=0)

    if save_member_preds:
        res["member_means"] = member_means
        if vars_replicas is not None:
            res["member_vars"] = vars_replicas

    return res


def train_nn_ensemble_forward(
    ds: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    L1/network ensemble wrapper.

    This should be used for ordinary L1 data where ds["y"] has shape (Ndata,).
    L2 data replicas are handled separately by train_l2_replicas_forward.
    """
    y_loaded = np.asarray(ds["y"])
    if y_loaded.ndim != 1:
        raise ValueError(
            "train_nn_ensemble_forward is intended for L1 data only, with "
            "ds['y'] shape (Ndata,). Use train_l2_replicas_forward for "
            "L2 data with shape (Ndata, N_l2_replicas)."
        )

    ens = cfg.get("nn", {}).get("ensemble", {})
    if not bool(ens.get("enabled", True)):
        return train_nn_forward(ds, cfg)

    ensemble_kind = str(ens.get("kind", "standard")).lower()
    if ensemble_kind == "repulsive":
        print("in train repulsive")
        return train_repulsive_nn_forward(ds, cfg)

    n_members = int(ens.get("n_members", 20))
    seed_offset = int(ens.get("seed_offset", 1000))
    save_member_preds = bool(ens.get("save_member_preds", False))
    subsample_members = ens.get("subsample_members", None)
    subsample_members = (
        int(subsample_members) if subsample_members is not None else None
    )

    base_seed = int(cfg.get("seed", 0))
    use_post_ntk = str(cfg.get("ntk", {}).get("when", "none")).lower() == "post"

    member_means = []
    member_vars = []
    member_losses = []

    member_post_ntk_means = []
    member_post_ntk_vars = []

    out_last = None

    for i in trange(n_members, desc="Training ensemble members"):
        cfg_i = dict(cfg)

        # Make both model init and split different per member.
        cfg_i["seed"] = base_seed + seed_offset + i
        cfg_i["replica"] = i

        out_i = train_nn_forward(ds, cfg_i)
        out_last = out_i

        member_means.append(out_i["f_grid_mean"])
        member_vars.append(out_i.get("f_grid_var", None))
        member_losses.append(out_i["loss_history"])

        member_post_ntk_means.append(out_i.get("post_ntk_f_mean", None))
        member_post_ntk_vars.append(out_i.get("post_ntk_f_var", None))

    raw_replicas = np.stack(member_means, axis=0)

    have_all_vars = all(v is not None for v in member_vars)
    raw_vars = np.stack(member_vars, axis=0) if have_all_vars else None

    have_all_post = (
        all(v is not None for v in member_post_ntk_means)
        and all(v is not None for v in member_post_ntk_vars)
    )

    post_ntk_replicas = (
        np.stack(member_post_ntk_means, axis=0) if have_all_post else None
    )
    post_ntk_vars = (
        np.stack(member_post_ntk_vars, axis=0) if have_all_post else None
    )

    # If requested and available, summarize the empirical NTK-GP curves rather
    # than the raw trained NN curves.
    if use_post_ntk and have_all_post:
        replicas = post_ntk_replicas
        vars_replicas = post_ntk_vars
    else:
        replicas = raw_replicas
        vars_replicas = raw_vars

    if subsample_members is not None and replicas.shape[0] > subsample_members:
        rng = np.random.default_rng(base_seed + seed_offset)
        keep = rng.choice(replicas.shape[0], subsample_members, replace=False)

        replicas = replicas[keep]
        member_losses = [member_losses[k] for k in keep.tolist()]

        if vars_replicas is not None:
            vars_replicas = vars_replicas[keep]

        if save_member_preds:
            member_means = [member_means[k] for k in keep.tolist()]

        # Keep diagnostic post-NTK arrays aligned with the public ensemble.
        if post_ntk_replicas is not None:
            post_ntk_replicas = post_ntk_replicas[keep]
        if post_ntk_vars is not None:
            post_ntk_vars = post_ntk_vars[keep]

    res = _summarise_replicas(
        xgrid=out_last["xgrid"],
        replicas=replicas,
        vars_replicas=vars_replicas,
        member_losses=member_losses,
        save_member_preds=save_member_preds,
        member_means=member_means,
        post_ntk_replicas=post_ntk_replicas,
        post_ntk_vars=post_ntk_vars,
    )

    if use_post_ntk and have_all_post:
        post_mean = replicas.mean(axis=0)

        if vars_replicas is not None:
            post_var = vars_replicas.mean(axis=0)
        else:
            post_var = np.zeros_like(post_mean)

        post_std = np.sqrt(np.maximum(post_var, 0.0))

        res["post_ntk_xgrid"] = out_last["xgrid"]
        res["post_ntk_f_mean"] = post_mean
        res["post_ntk_f_var"] = post_var
        res["post_ntk_f_std"] = post_std
        res["post_ntk_f_lo68"] = post_mean - post_std
        res["post_ntk_f_hi68"] = post_mean + post_std
        res["post_ntk_f_lo95"] = post_mean - 1.96 * post_std
        res["post_ntk_f_hi95"] = post_mean + 1.96 * post_std

        if post_ntk_vars is not None:
            res["post_ntk_f_cov"] = np.diag(post_var)

    return res


def train_l2_replicas_forward(
    ds: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    L2 data-replica wrapper.

    Requires ds["y"] with shape (Ndata, N_l2_replicas).

    If nn.ensemble.enabled is true, this trains n_members networks for each
    L2 data replica. If false, this trains one network per L2 data replica.

    Optional random initialization of trainable endpoint-scaling parameters:

        nn:
          l2_scaling_init:
            enabled: true
            alpha_range: [0.0, 1.1]
            beta_range: [1.0, 5.0]

    This only applies for loss.name == "chi2".
    """
    y_loaded = np.asarray(ds["y"])
    if y_loaded.ndim != 2:
        raise ValueError(
            "train_l2_replicas_forward requires ds['y'] with shape "
            "(Ndata, N_l2_replicas)."
        )

    ens = cfg.get("nn", {}).get("ensemble", {})
    ensemble_enabled = bool(ens.get("enabled", True))

    n_members = int(ens.get("n_members", 20)) if ensemble_enabled else 1
    seed_offset = int(ens.get("seed_offset", 1000))
    save_member_preds = bool(ens.get("save_member_preds", False))
    subsample_members = ens.get("subsample_members", None)
    subsample_members = (
        int(subsample_members) if subsample_members is not None else None
    )

    base_seed = int(cfg.get("seed", 0))
    n_l2_replicas = y_loaded.shape[1]
    use_post_ntk = str(cfg.get("ntk", {}).get("when", "none")).lower() == "post"

    member_means = []
    member_vars = []
    member_losses = []

    member_replica_ids = []
    member_l2_ids = []
    member_init_alpha = []
    member_init_beta = []

    member_post_ntk_means = []
    member_post_ntk_vars = []

    out_last = None

    for replica_l2 in range(n_l2_replicas):
        for i in trange(n_members, desc=f"Training L2 replica {replica_l2}"):
            cfg_i = dict(cfg)

            fit_id = replica_l2 * n_members + i
            fit_seed = base_seed + seed_offset + fit_id

            cfg_i["seed"] = fit_seed
            cfg_i["replica"] = i
            cfg_i["replica_l2"] = replica_l2

            sampled_alpha, sampled_beta = _sample_l2_scaling_init(cfg_i, fit_seed=fit_seed)
            if sampled_alpha is not None or sampled_beta is not None:
                cfg_i["nn"] = dict(cfg_i.get("nn", {}))
                if sampled_alpha is not None:
                    cfg_i["nn"]["init_alpha"] = sampled_alpha
                if sampled_beta is not None:
                    cfg_i["nn"]["init_beta"] = sampled_beta
                print(f"[L2 scaling init] fit_id={fit_id}, replica={i}, replica_l2={replica_l2}, init_alpha={cfg_i['nn'].get('init_alpha')}, init_beta={cfg_i['nn'].get('init_beta')}")

            out_i = train_nn_forward(ds, cfg_i)
            out_last = out_i

            member_means.append(out_i["f_grid_mean"])
            member_vars.append(out_i.get("f_grid_var", None))
            member_losses.append(out_i["loss_history"])

            member_replica_ids.append(i)
            member_l2_ids.append(replica_l2)
            member_init_alpha.append(
                float(cfg_i.get("nn", {}).get("init_alpha", cfg.get("nn", {}).get("init_alpha", 1.0)))
            )
            member_init_beta.append(
                float(cfg_i.get("nn", {}).get("init_beta", cfg.get("nn", {}).get("init_beta", 3.0)))
            )

            member_post_ntk_means.append(out_i.get("post_ntk_f_mean", None))
            member_post_ntk_vars.append(out_i.get("post_ntk_f_var", None))

    raw_replicas = np.stack(member_means, axis=0)

    have_all_vars = all(v is not None for v in member_vars)
    raw_vars = np.stack(member_vars, axis=0) if have_all_vars else None

    have_all_post = (
        all(v is not None for v in member_post_ntk_means)
        and all(v is not None for v in member_post_ntk_vars)
    )

    post_ntk_replicas = (
        np.stack(member_post_ntk_means, axis=0) if have_all_post else None
    )
    post_ntk_vars = (
        np.stack(member_post_ntk_vars, axis=0) if have_all_post else None
    )

    if use_post_ntk and have_all_post:
        replicas = post_ntk_replicas
        vars_replicas = post_ntk_vars
    else:
        replicas = raw_replicas
        vars_replicas = raw_vars

    # Before arbitrary subsampling, the structure is rectangular:
    # (N_l2_replicas, N_members, Ngrid).
    l2_member_replicas = replicas.reshape(
        n_l2_replicas,
        n_members,
        replicas.shape[-1],
    )

    if vars_replicas is not None:
        l2_member_vars = vars_replicas.reshape(
            n_l2_replicas,
            n_members,
            vars_replicas.shape[-1],
        )
    else:
        l2_member_vars = None

    if subsample_members is not None and replicas.shape[0] > subsample_members:
        rng = np.random.default_rng(base_seed + seed_offset)
        keep = rng.choice(replicas.shape[0], subsample_members, replace=False)

        replicas = replicas[keep]
        member_losses = [member_losses[k] for k in keep.tolist()]
        member_replica_ids = [member_replica_ids[k] for k in keep.tolist()]
        member_l2_ids = [member_l2_ids[k] for k in keep.tolist()]
        member_init_alpha = [member_init_alpha[k] for k in keep.tolist()]
        member_init_beta = [member_init_beta[k] for k in keep.tolist()]

        if vars_replicas is not None:
            vars_replicas = vars_replicas[keep]

        if save_member_preds:
            member_means = [member_means[k] for k in keep.tolist()]

        # Keep diagnostic post-NTK arrays aligned with the public ensemble.
        if post_ntk_replicas is not None:
            post_ntk_replicas = post_ntk_replicas[keep]
        if post_ntk_vars is not None:
            post_ntk_vars = post_ntk_vars[keep]

        # Arbitrary flat subsampling destroys the rectangular L2/member layout.
        l2_member_replicas = None
        l2_member_vars = None

    res = _summarise_replicas(
        xgrid=out_last["xgrid"],
        replicas=replicas,
        vars_replicas=vars_replicas,
        member_losses=member_losses,
        save_member_preds=save_member_preds,
        member_means=member_means,
        post_ntk_replicas=post_ntk_replicas,
        post_ntk_vars=post_ntk_vars,
    )

    res.update({
        "l2_member_replicas": l2_member_replicas,
        "l2_member_vars": l2_member_vars,
        "replica_ids": np.asarray(member_replica_ids, dtype=int),
        "replica_l2_ids": np.asarray(member_l2_ids, dtype=int),
        "init_alpha_members": np.asarray(member_init_alpha, dtype=np.float64),
        "init_beta_members": np.asarray(member_init_beta, dtype=np.float64),
        "n_members": int(n_members),
        "n_l2_replicas": int(n_l2_replicas),
    })

    if l2_member_replicas is not None:
        res["mean_per_l2_replica"] = l2_member_replicas.mean(axis=1)
        res["std_per_l2_replica"] = l2_member_replicas.std(axis=1)
        res["lo68_per_l2_replica"] = np.percentile(
            l2_member_replicas, 16.0, axis=1
        )
        res["hi68_per_l2_replica"] = np.percentile(
            l2_member_replicas, 84.0, axis=1
        )

    return res