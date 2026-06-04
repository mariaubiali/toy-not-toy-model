from __future__ import annotations
from typing import Any, Dict
import copy
import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import trange

from losses import (
    LossContext,
    make_loss,
    pointwise_loss_members,
    _cholesky_C,
    _apply_Cinv,
    _safe_cholesky,
)
from models.nn_models import MLPFModel, RepulsiveMLPFModel
from models.ntk import run_ntk_stage

def select_training_target(ds: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(ds["y"], dtype=np.float32)
    target = str(ds.get("target", cfg.get("data", {}).get("target", "y")))

    # L1 case: y_pseudo, y_theory, etc.
    if y.ndim == 1:
        return y

    # L2 case: y_l2 with shape (Ndata, N_l2_replicas)
    if y.ndim == 2:
        replica_l2 = int(cfg.get("replica_l2", 0))
        n_l2_replicas = y.shape[1]

        if replica_l2 < 0 or replica_l2 >= n_l2_replicas:
            raise ValueError(
                f"replica_l2={replica_l2} requested, but target={target} "
                f"has only {n_l2_replicas} replicas."
            )

        return y[:, replica_l2]

    raise ValueError(
        f"Target {target} must have shape (Ndata,) or "
        f"(Ndata, N_l2_replicas), got {y.shape}"
    )


def _use_l2_cross_validation(ds: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """
    Enable cross validation only for the L2 chi2 case.

    The default is False, so existing runcards keep the old behavior.
    """
    cv_cfg = cfg.get("nn", {}).get("cross_validation", {})
    if not bool(cv_cfg.get("enabled", False)):
        return False

    loss_name = str(cfg.get("loss", {}).get("name", "")).lower()
    target = str(ds.get("target", cfg.get("data", {}).get("target", "y"))).lower()
    y_loaded = np.asarray(ds["y"])

    return (
        target == "y_l2"
        and y_loaded.ndim == 2
        and loss_name == "chi2"
    )


def train_nn_forward(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    # ----------------------------
    # Load data from runcard
    # ----------------------------
    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32
    meta = ds.get("meta", {})
    xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()
    xgrid_ext = (
        meta.get("xgrid_ext").astype(np.float64)
        if "xgrid_ext" in meta
        else np.array([], dtype=np.float64)
    )
    xt3_true_ext = np.asarray(meta.get("xt3_ext", []), float).ravel()

    nncfg = cfg.get("nn", {})
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
    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    out_dim = int(nncfg.get("out_dim", 1.0))
    use_het = (loss_name in {"mse_het", "chi2_het"}) and (out_dim == 2)
    jitter = float(cfg.get("kernel", {}).get("jitter", cfg.get("loss", {}).get("jitter", 1e-10)))
    replica = int(cfg.get("replica", 0))

    # Cross validation is opt-in and restricted to the L2 chi2 case.
    # If disabled, the old split seed and old test_size=0.2 are preserved.
    cv_cfg = nncfg.get("cross_validation", {})
    cv_enabled = _use_l2_cross_validation(ds, cfg)
    val_fraction = float(cv_cfg.get("val_fraction", 0.2))
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"nn.cross_validation.val_fraction must be between 0 and 1, got {val_fraction}")
    cv_seed = seed + int(cv_cfg.get("seed_offset", 2000))
    patience = int(nncfg.get("patience", 500))
    min_delta = float(nncfg.get("min_delta", 0.0))

    # NTK config (explicit init or post)
    ntk_cfg = cfg.get("ntk", {})
    ntk_when = ntk_cfg.get("when", "none")

    # reproducibility for model init+split
    torch.manual_seed(seed)
    np.random.seed(seed)

    xgrid = ds["xgrid"].astype(np.float32)  # (Ngrid,)
    W = ds["W"].astype(np.float32)  # (Ndat, Ngrid)
    C = ds["C"].astype(np.float32)  # (Ndat, Ndat)
    y = select_training_target(ds, cfg).astype(np.float32)  # (Ndat,)

    n_data = W.shape[0]
    n_grid = xgrid.shape[0]
    if xt3_true.size != n_grid:
        raise ValueError(
            f"xt3_true must be defined on xgrid: got {xt3_true.size} vs Ngrid={n_grid}"
        )

    # transform variables to troch tensors
    x_torch = torch.tensor(xgrid, dtype=dtype, device=device).unsqueeze(1)  # (Ngrid, 1)
    xgrid_1d = x_torch.squeeze(1)  # (Ngrid,)
    W_torch = torch.tensor(W, dtype=dtype, device=device)  # (Ndat, Ngrid)
    C_torch = torch.tensor(C, dtype=dtype, device=device)  # (Ndat, Ndat)
    y_torch = torch.tensor(y, dtype=dtype, device=device)  # (Ndat,)

    xext_torch = torch.tensor(xgrid_ext, dtype=dtype, device=device).unsqueeze(1)
    xext_1d = xext_torch.squeeze(1)

    # ----------------------------
    # Calculate reference integral, based on "true" xt3 from NNPDF/Theory ID
    # must be defined on full xgrid
    # ----------------------------
    t3_ref_int = float(np.trapz(xt3_true / xgrid, xgrid))

    # ----------------------------
    # Construct model
    # ----------------------------
    model = MLPFModel(
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        out_dim=out_dim,
        scaling=scaling,
        init_alpha=init_alpha,
        init_beta=init_beta,
        transforms=transforms,
    ).to(device=device, dtype=dtype)

    # ------------------------------------------------------------
    # Optional L2 chi2 scan mode:
    # Freeze endpoint-scaling exponents alpha and beta.
    #
    # This keeps alpha=nn.init_alpha and beta=nn.init_beta fixed during
    # training, which is useful when scanning fixed alpha/beta values for
    # L2 chi2 fits.
    #
    # Comment out this block if alpha and beta should again be learned
    # as ordinary hyperparameters/fit parameters in future runs.
    # ------------------------------------------------------------
    # if loss_name == "chi2":
    #     print("Freezing alpha and beta")

    #     # alpha is now stored directly, so negative values are allowed.
    #     if hasattr(model, "alpha"):
    #         model.alpha.requires_grad_(False)

    #     # fallback for old model versions
    #     if hasattr(model, "logalpha"):
    #         model.logalpha.requires_grad_(False)

    #     # beta is still stored logarithmically to keep beta positive.
    #     if hasattr(model, "logbeta"):
    #         model.logbeta.requires_grad_(False)

    #     print(
    #         f"[L2 chi2 scan] fixed alpha={init_alpha:.6f}, "
    #         f"fixed beta={init_beta:.6f}"
    #     )


    # ----------------------------
    # Proper train/val split over DATA points (rows of W)
    # ----------------------------
    idx_all = np.arange(n_data)
    split_test_size = val_fraction if cv_enabled else 0.2
    split_seed = cv_seed if cv_enabled else (seed + replica * 1000)
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=split_test_size,
        random_state=split_seed,
    )

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    W_tr = W_torch[train_idx_t, :]  # (Ntr,Ngrid)
    W_val = W_torch[val_idx_t, :]  # (Nval,Ngrid)
    y_tr = y_torch[train_idx_t]  # (Ntr,)
    y_val = y_torch[val_idx_t]  # (Nval,)

    C_tr = C_torch[train_idx_t][:, train_idx_t]  # (Ntr,Ntr)

    C_val = C_torch[val_idx_t][:, val_idx_t]  # (Nval,Nval)

    # Cholesky factors for stable chi2
    L_tr = _cholesky_C(C_tr, jitter)
    L_val = _cholesky_C(C_val, jitter)

    # training loss context should use TRAIN y/C/W, but full xgrid for sumrule
    ctx = LossContext(
        W=W_tr,
        C=C_tr,
        y=y_tr,
        xgrid=xgrid_1d,
        t3_ref_int=t3_ref_int,
        jitter=jitter,
        L=L_tr,
    )
    loss_fn, extra_params = make_loss(cfg, ctx, device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    params += [p for p in extra_params.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    loss_hist = []
    val_chi2_hist = []
    log_every = max(1, epochs // 20)

    best_val_chi2 = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = epochs - 1

    # ----------------------------
    # Prediction grid selection
    # ----------------------------
    x_pred_torch = xext_torch if xgrid_ext.size > 0 else x_torch
    x_pred_np = xgrid_ext.astype(np.float64) if xgrid_ext.size > 0 else xgrid.astype(np.float64)

    res: Dict[str, Any] = {}

    # ----------------------------
    # NTK-GP at initialization (optional)
    # ----------------------------
    if str(ntk_when).lower() == "init":
        res.update(
            run_ntk_stage(
                stage="init",
                model=model,
                xgrid_torch=x_torch,
                xgrid_np=xgrid.astype(np.float64),
                W_train=W_tr,
                y_train=y_tr,
                C_train=C_tr,
                x_pred_torch=x_pred_torch,
                x_pred_np=x_pred_np,
                cfg=cfg,
            )
        )

    # ----------------------------
    # Training loop
    # ----------------------------

    for ep in range(epochs):
        # -------- training step --------
        model.train()
        opt.zero_grad()
        out = model(x_torch)
        f_grid = out["f_grid"]  # (Ngrid, 2) if out_dim=2
        f_mu = out["f_grid"]
        if f_mu.ndim == 2:  # e.g. (Ngrid,1) or (Ngrid,2)
            f_mu = f_mu[:, 0]
        f_mu = f_mu.reshape(-1)

        # Logvar on grid (try both conventions)
        f_logvar = None
        if "logvar_f_grid" in out:
            f_logvar = out["logvar_f_grid"].reshape(-1)
        elif out["f_grid"].ndim == 2 and out["f_grid"].shape[1] == 2:
            f_logvar = out["f_grid"][:, 1].reshape(-1)

        extra = {"f_mu": f_mu}
        if f_logvar is not None:
            extra["f_logvar"] = f_logvar

        y_pred_tr = W_tr @ f_mu  # (Ntr,)

        loss = loss_fn(y_pred_tr, extra)
        loss.backward()
        opt.step()

        # -------- val step --------
        # val uses L_val (cached once in nn_train)
        model.eval()
        with torch.no_grad():
            out_v = model(x_torch)

            f_mu_v = out_v["f_grid"]
            if f_mu_v.ndim == 2:  # (Ngrid,1) or (Ngrid,2)
                f_mu_v = f_mu_v[:, 0]
            f_mu_v = f_mu_v.reshape(-1)

            f_logvar_v = None
            if "logvar_f_grid" in out_v:
                f_logvar_v = out_v["logvar_f_grid"].reshape(-1)
            elif out_v["f_grid"].ndim == 2 and out_v["f_grid"].shape[1] == 2:
                f_logvar_v = out_v["f_grid"][:, 1].reshape(-1)

            y_pred_val = W_val @ f_mu_v  # (Nval,)
            r_val = (y_pred_val - y_val).reshape(-1)

            # chi2 using provided C_val (as before)
            chi2_val = float(r_val @ _apply_Cinv(L_val, r_val))
            chi2_val_pt = chi2_val / float(len(val_idx))

            # sum rule check on mean curve
            I_mid_v = torch.trapz(f_mu_v / xgrid_1d, xgrid_1d).item()
            delta_sr_v = I_mid_v - t3_ref_int

            # OPTIONAL: heteroscedastic NLL on val (diagonal propagation)
            nll_val_mean = None
            if use_het and (f_logvar_v is not None):
                lcfg = cfg.get("loss", {})
                eps = float(lcfg.get("eps", 1e-12))
                logvar_clip = lcfg.get("logvar_clip", (-20.0, 5.0))
                full_cov = bool(lcfg.get("full_cov", False))
                full_cov_jitter_abs = float(lcfg.get("full_cov_jitter_abs", eps))
                full_cov_jitter_rel = float(lcfg.get("full_cov_jitter_rel", 1e-8))
                lo, hi = float(logvar_clip[0]), float(logvar_clip[1])

                f_logvar_v = f_logvar_v.clamp(lo, hi)
                var_f_v = torch.exp(f_logvar_v).clamp_min(eps)  # (Ngrid,)
                if full_cov:
                    C_val_het = (W_val * var_f_v.unsqueeze(0)) @ W_val.T
                    L_val_het, _ = _safe_cholesky(
                        C_val_het,
                        jitter_abs=full_cov_jitter_abs,
                        jitter_rel=full_cov_jitter_rel,
                    )
                    alpha = _apply_Cinv(L_val_het, r_val)
                    quad = r_val @ alpha
                    logdet = 2.0 * torch.sum(torch.log(torch.diag(L_val_het)))
                    nll_val_mean = float(
                        (0.5 * (quad + logdet) / float(r_val.shape[0]))
                        .detach()
                        .cpu()
                        .item()
                    )
                else:
                    var_y_val = (W_val * W_val) @ var_f_v  # (Nval,)
                    var_y_val = var_y_val.clamp_min(eps)
                    nll_val = 0.5 * ((r_val * r_val) / var_y_val + torch.log(var_y_val))
                    nll_val_mean = float(nll_val.mean().detach().cpu().item())

        val_chi2_hist.append(float(chi2_val_pt))

        if cv_enabled:
            # Strict improvement by at least min_delta.
            if chi2_val_pt < best_val_chi2 - min_delta:
                best_val_chi2 = float(chi2_val_pt)
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # if epochs_without_improvement >= patience:
            #     stopped_early = True
            #     stopped_epoch = ep
            #     break

        if (ep + 1) % log_every == 0:
            loss_hist.append(float(loss.detach().cpu().item()))

    if cv_enabled and best_state is not None:
        model.load_state_dict(best_state)
    
    if hasattr(model, "alpha") and hasattr(model, "beta"):
        with torch.no_grad():
            final_alpha = float(model.alpha.detach().cpu())
            final_beta = float(model.beta.detach().cpu())

        alpha_status = "trained" if model.alpha.requires_grad else "fixed"
        beta_status = "trained" if model.beta.requires_grad else "fixed"

        print(
            f"[NN scaling] final alpha={final_alpha:.6f} ({alpha_status}), "
            f"final beta={final_beta:.6f} ({beta_status})"
        )

    # ----------------------------
    # NTK-GP after full training (optional)
    # ----------------------------
    if str(ntk_when).lower() == "post":
        res.update(
            run_ntk_stage(
                stage="post",
                model=model,
                xgrid_torch=x_torch,
                xgrid_np=xgrid.astype(np.float64),
                W_train=W_tr,
                y_train=y_tr,
                C_train=C_tr,
                x_pred_torch=x_pred_torch,
                x_pred_np=x_pred_np,
                cfg=cfg,
            )
        )

        # Promote post-NTK outputs to the primary curve summary so downstream
        # consumers see the empirical NTK GP rather than the raw trained NN.
        if "post_ntk_f_mean" in res:
            res["xgrid"] = np.asarray(res.get("post_ntk_xgrid", x_pred_np), dtype=np.float64)
            res["f_grid_mean"] = np.asarray(res["post_ntk_f_mean"], dtype=np.float64)
            res["f_grid_var"] = np.asarray(res.get("post_ntk_f_var", np.zeros_like(res["f_grid_mean"])), dtype=np.float64)
            if "post_ntk_f_cov" in res:
                res["f_grid_cov"] = np.asarray(res["post_ntk_f_cov"], dtype=np.float64)
            res["f_grid_sigma"] = np.asarray(
                res.get("post_ntk_f_std", np.sqrt(np.maximum(res["f_grid_var"], 0.0))),
                dtype=np.float64,
            )

    # ----------------------------
    # Return full-grid prediction
    # ----------------------------
    model.eval()
    # print("x pred shape:", x_pred_torch.shape)
    # print("x shape:", x_torch.shape)
    # print("model eval")
    with torch.no_grad():
        x_pred_torch = xext_torch if xgrid_ext.size > 0 else x_torch
        x_pred_np = (
            xgrid_ext.astype(np.float64)
            if xgrid_ext.size > 0
            else xgrid.astype(np.float64)
        )

        # CHANGED: evaluate on chosen grid
        out_full = model(x_pred_torch)
        

        f_grid_full = out_full["f_grid"]
        f_mu_full = f_grid_full[:, 0].reshape(-1) if f_grid_full.ndim == 2 else f_grid_full.reshape(-1)

        f_var = None
        f_cov = None
        f_sigma = None
        if use_het and ("logvar_f_grid" in out_full):
            f_logvar_full = out_full["logvar_f_grid"].reshape(-1)
            f_var = torch.exp(f_logvar_full).detach().cpu().numpy().astype(np.float64)
            f_cov = np.diag(f_var)
            f_sigma = np.sqrt(np.maximum(np.diag(f_cov), 1e-18))
        elif use_het and f_grid_full.ndim == 2 and f_grid_full.shape[1] == 2:
            f_logvar_full = f_grid_full[:, 1].reshape(-1)
            f_var = torch.exp(f_logvar_full).detach().cpu().numpy().astype(np.float64)
            f_cov = np.diag(f_var)
            f_sigma = np.sqrt(np.maximum(np.diag(f_cov), 1e-18))

        f_mean = f_mu_full.detach().cpu().numpy().astype(np.float64)

        if "f_grid_mean" not in res:
            res["xgrid"] = x_pred_np
            res["f_grid_mean"] = f_mean
        res.update({
            "loss_history": np.array(loss_hist, dtype=np.float64),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "cv": {
                "enabled": bool(cv_enabled),
                "val_fraction": float(split_test_size),
                "split_seed": int(split_seed),
                "patience": int(patience),
                "min_delta": float(min_delta),
                "best_epoch": int(best_epoch),
                "best_val_chi2_per_point": float(best_val_chi2) if np.isfinite(best_val_chi2) else None,
                "stopped_early": bool(stopped_early),
                "stopped_epoch": int(stopped_epoch),
                "val_chi2_history": np.array(val_chi2_hist, dtype=np.float64),
            },
        })

        if f_var is not None and "f_grid_var" not in res:
            res["f_grid_var"] = f_var
        if f_cov is not None and "f_grid_cov" not in res:
            res["f_grid_cov"] = f_cov
        if f_sigma is not None and "f_grid_sigma" not in res:
            res["f_grid_sigma"] = f_sigma

        assert res["xgrid"].shape[0] == res["f_grid_mean"].shape[0]
        # print("f_var: ", f_var)
    return res

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

    member_post_ntk_means = []
    member_post_ntk_vars = []

    out_last = None

    for replica_l2 in range(n_l2_replicas):
        for i in trange(n_members, desc=f"Training L2 replica {replica_l2}"):
            cfg_i = dict(cfg)

            fit_id = replica_l2 * n_members + i

            cfg_i["seed"] = base_seed + seed_offset + fit_id
            cfg_i["replica"] = i
            cfg_i["replica_l2"] = replica_l2

            out_i = train_nn_forward(ds, cfg_i)
            out_last = out_i

            member_means.append(out_i["f_grid_mean"])
            member_vars.append(out_i.get("f_grid_var", None))
            member_losses.append(out_i["loss_history"])

            member_replica_ids.append(i)
            member_l2_ids.append(replica_l2)

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