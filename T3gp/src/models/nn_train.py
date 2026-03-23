from __future__ import annotations
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import trange

from losses import LossContext, make_loss, _cholesky_C, _apply_Cinv, _safe_cholesky
from models.nn_models import MLPFModel
from models.ntk import run_ntk_stage


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
    jitter = float(cfg.get("kernel", {}).get("jitter", 1e-10))
    replica = int(cfg.get("replica", 0))

    # NTK config (init + mid/post)
    ntk_cfg = cfg.get("ntk", {})
    ntk_when = ntk_cfg.get("when", "none")
    # Optional: compute at an intermediate epoch (0-indexed).
    ntk_epoch = ntk_cfg.get("epoch", None)
    ntk_epoch = int(ntk_epoch) if ntk_epoch is not None else None

    # reproducibility for model init+split
    torch.manual_seed(seed)
    np.random.seed(seed)

    xgrid = ds["xgrid"].astype(np.float32)  # (Ngrid,)
    W = ds["W"].astype(np.float32)  # (Ndat, Ngrid)
    C = ds["C"].astype(np.float32)  # (Ndat, Ndat)
    y = ds["y"].astype(np.float32)  # (Ndat,)

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

    # ----------------------------
    # Proper train/val split over DATA points (rows of W)
    # ----------------------------
    idx_all = np.arange(n_data)
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=seed + replica * 1000,
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

    params = list(model.parameters()) + list(extra_params.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    loss_hist = []
    log_every = max(1, epochs // 20)

    # ----------------------------
    # Prediction grid selection
    # ----------------------------
    x_pred_torch = xext_torch if xgrid_ext.size > 0 else x_torch
    x_pred_np = xgrid_ext.astype(np.float64) if xgrid_ext.size > 0 else xgrid.astype(np.float64)

    res: Dict[str, Any] = {}

    # ----------------------------
    # NTK at initialization (optional)
    # ----------------------------
    if str(ntk_when).lower() in ("init", "both") or (isinstance(ntk_when, (list, tuple, set)) and "init" in {str(x).lower() for x in ntk_when}):
        if C_tr is None and bool(ntk_cfg.get("gp", {}).get("use_data_cov", True)):
            # If user wants C as noise but dataset has no C, they'll get a clear error.
            pass
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

        if (ep + 1) % log_every == 0:
            loss_hist.append(float(loss.detach().cpu().item()))

        # ----------------------------
        # NTK at a chosen epoch (optional)
        # Linearize at the current weights after this epoch's update.
        # ----------------------------
        if ntk_epoch is not None and ep == ntk_epoch and str(ntk_when).lower() in ("both"):
            res.update(
                run_ntk_stage(
                    stage="mid",
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
    # NTK after full training (optional)
    # ----------------------------
    if str(ntk_when).lower() in ("post", "both") or ((isinstance(ntk_when, (list, tuple, set)) and "post" in {str(x).lower() for x in ntk_when})):
        print("ntk post mode")
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

        res.update({
            "xgrid": x_pred_np,
            "f_grid_mean": f_mean,
            "loss_history": np.array(loss_hist, dtype=np.float64),
            "train_idx": train_idx,
            "val_idx": val_idx,
        })

        if f_var is not None:
            res["f_grid_var"] = f_var
        if f_cov is not None:
            res["f_grid_cov"] = f_cov
        if f_sigma is not None:
            res["f_grid_sigma"] = f_sigma

        assert res["xgrid"].shape[0] == res["f_grid_mean"].shape[0]
        # print("f_var: ", f_var)
    return res


def train_nn_ensemble_forward(
    ds: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    ens = cfg.get("nn", {}).get("ensemble", {})
    if not bool(ens.get("enabled", True)):
        return train_nn_forward(ds, cfg)
    
    member_post_ntk_means = []
    member_post_ntk_vars  = []

    # Treat ensemble size as replicas
    n_members = int(ens.get("n_members", 20))
    seed_offset = int(ens.get("seed_offset", 1000))
    save_member_preds = bool(ens.get("save_member_preds", False))
    subsample_members = ens.get("subsample_members", None)
    subsample_members = (
        int(subsample_members) if subsample_members is not None else None
    )

    base_seed = int(cfg.get("seed", 0))

    member_means = []
    member_vars = []
    member_losses = []
    out_last = None

    for i in trange(n_members, desc="Training ensemble members"):
        cfg_i = dict(cfg)
        # make both init and split different per member
        cfg_i["seed"] = base_seed + seed_offset + i
        cfg_i["replica"] = i  # controls train/val split random_state

        out_i = train_nn_forward(ds, cfg_i)
        out_last = out_i

        member_means.append(out_i["f_grid_mean"])
        member_losses.append(out_i["loss_history"])
        member_post_ntk_means.append(out_i.get("post_ntk_f_mean", None))
        member_post_ntk_vars.append(out_i.get("post_ntk_f_var", None))

        # might be zero, depending on loss type
        v = out_i.get("f_grid_var", None)
        member_vars.append(v)

    replicas = np.stack(member_means, axis=0)  # (S, Ngrid)
    xgrid = out_last["xgrid"]

    # Only keep vars if EVERY member provided them
    have_all_vars = all(v is not None for v in member_vars)
    vars_replicas = np.stack(member_vars, axis=0) if have_all_vars else None

    have_all_post = all(v is not None for v in member_post_ntk_means) and all(v is not None for v in member_post_ntk_vars)
    post_ntk_replicas = np.stack(member_post_ntk_means, axis=0) if have_all_post else None
    post_ntk_vars     = np.stack(member_post_ntk_vars,  axis=0) if have_all_post else None

    # OPTIONAL SUBSAMPLING OF MEMBERS
    if subsample_members is not None and replicas.shape[0] > subsample_members:
        rng = np.random.default_rng(base_seed + seed_offset)
        keep = rng.choice(replicas.shape[0], subsample_members, replace=False)
        replicas = replicas[keep]

        # optionally subsample losses too (keeps alignment)
        member_losses = [member_losses[k] for k in keep.tolist()]
        if save_member_preds:
            member_means = [member_means[k] for k in keep.tolist()]
        if vars_replicas is not None:
            vars_replicas = vars_replicas[keep]

    res = {
        "xgrid": xgrid,
        "replicas": replicas,  # (S, Ngrid)
        "vars_replicas": vars_replicas,  # (S, Ngrid) or None
        "mean_curve": replicas.mean(axis=0),
        "lo68": np.percentile(replicas, 16.0, axis=0),
        "hi68": np.percentile(replicas, 84.0, axis=0),
        "lo95": np.percentile(replicas, 2.5, axis=0),
        "hi95": np.percentile(replicas, 97.5, axis=0),
        "member_losses": member_losses,
    }

    # Nice-to-have: percentiles for aleatoric var too (if present)
    if vars_replicas is not None:
        res["var_mean_curve"] = vars_replicas.mean(axis=0)
        res["var_lo68"] = np.percentile(vars_replicas, 16.0, axis=0)
        res["var_hi68"] = np.percentile(vars_replicas, 84.0, axis=0)

    if save_member_preds:
        res["member_means"] = member_means
        if vars_replicas is not None:
            res["member_vars"] = vars_replicas
    res["post_ntk_replicas"] = post_ntk_replicas
    res["post_ntk_vars_replicas"] = post_ntk_vars
    
    return res
