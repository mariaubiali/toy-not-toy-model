from __future__ import annotations
from typing import Any, Dict

import numpy as np
import torch

from losses import LossContext, make_loss
from models.nn_models import MLPFModel


def train_nn_forward(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32

    xgrid = ds["xgrid"].astype(np.float32)   # (Ngrid,)
    W = ds["W"].astype(np.float32)           # (Ndat, Ngrid)
    C = ds["C"].astype(np.float32)           # (Ndat, Ndat)
    y = ds["y"].astype(np.float32)           # (Ndat,)

    x_t = torch.from_numpy(xgrid).view(-1, 1).to(device=device, dtype=dtype)
    W_t = torch.from_numpy(W).to(device=device, dtype=dtype)
    C_t = torch.from_numpy(C).to(device=device, dtype=dtype)
    y_t = torch.from_numpy(y).to(device=device, dtype=dtype)

    meta = ds.get("meta", {})
    if "xt3_true" in meta:
        xt3_true = np.asarray(meta["xt3_true"], float).ravel()

    nncfg = cfg.get("nn", {})
    hidden = nncfg.get("hidden", [64, 64])
    activation = str(nncfg.get("activation", "tanh"))
    lr = float(nncfg.get("lr", 1e-3))
    epochs = int(nncfg.get("epochs", 3000))
    weight_decay = float(nncfg.get("weight_decay", 0.0))
    seed = int(cfg.get("seed", 0))
    dropout=nncfg.get("dropout", 0.0),

    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    need_logvar = ("het" in loss_name)

    out_dim = 2 if need_logvar else 1

    model = MLPFModel(
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        out_dim=out_dim,
    ).to(device=device, dtype=dtype)

    jitter = float(cfg.get("kernel", {}).get("jitter", 1e-10))
    # 1D x-grid for the T3_Beta sum-rule term
    xgrid_t = x_t.squeeze(1)
    xt3_true = np.asarray(meta["xt3_true"], float).ravel()
    t3_ref_int = np.trapz(xt3_true / xgrid_t, xgrid_t)
    ctx = LossContext(W=W_t, C=C_t, y=y_t, xgrid=xgrid_t, t3_ref_int=t3_ref_int, jitter=jitter)
    loss_fn, extra_params = make_loss(cfg, ctx, device=device, dtype=dtype)

    # optimizer includes model params + any extra loss params (e.g. learned noise)
    params = list(model.parameters()) + list(extra_params.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    loss_hist = []
    for ep in range(epochs):
        opt.zero_grad()

        out = model(x_t)
        f_grid = out["f_grid"]                # (Ngrid,)
        y_pred = W_t @ f_grid                 # (Ndat,)

        loss = loss_fn(y_pred, out)
        loss.backward()
        opt.step()

        if (ep + 1) % max(1, epochs // 20) == 0:
            loss_hist.append(float(loss.detach().cpu().numpy()))

    # predict on x_star for plotting (deterministic NN)
    x_star_n = int(cfg.get("eval", {}).get("x_star_n", 400))
    x_star = np.linspace(0.0, 1.0, x_star_n, dtype=np.float32)
    x_star_t = torch.from_numpy(x_star).view(-1, 1).to(device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        out_star = model(x_star_t)
        f_star = out_star["f_grid"].detach().cpu().numpy()

        res = {
            "x_star": x_star.astype(np.float64),
            "f_star_mean": f_star.astype(np.float64),
            "loss_history": np.array(loss_hist, dtype=np.float64),
        }

        if "logvar_f_grid" in out_star:
            logvar_star = out_star["logvar_f_grid"].detach().cpu().numpy()
            res["f_star_var"] = np.exp(logvar_star).astype(np.float64)

    return res

def train_nn_ensemble_forward(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ens = cfg.get("nn", {}).get("ensemble", {})
    enabled = bool(ens.get("enabled", True))
    if not enabled:
        return train_nn_forward(ds, cfg)

    n_members = int(ens.get("n_members", 20))
    seed_offset = int(ens.get("seed_offset", 1000))
    save_member_preds = bool(ens.get("save_member_preds", False))
    subsample_members = ens.get("subsample_members", None)
    subsample_members = int(subsample_members) if subsample_members is not None else None

    base_seed = int(cfg.get("seed", 0))

    member_curves = []
    member_losses = []

    for i in range(n_members):
        cfg_i = dict(cfg)
        cfg_i["seed"] = base_seed + seed_offset + i

        out_i = train_nn_forward(ds, cfg_i)
        member_curves.append(out_i["f_star_mean"])
        member_losses.append(out_i["loss_history"])

    replicas = np.stack(member_curves, axis=0)  # (S, N*)
    x_star = out_i["x_star"]

    if subsample_members is not None and replicas.shape[0] > subsample_members:
        rng = np.random.default_rng(base_seed + seed_offset)
        idx = rng.choice(replicas.shape[0], size=subsample_members, replace=False)
        replicas = replicas[idx]

    mean_curve = replicas.mean(axis=0)
    lo68, hi68 = np.percentile(replicas, [16.0, 84.0], axis=0)
    lo95, hi95 = np.percentile(replicas, [2.5, 97.5], axis=0)

    res = {
        "x_star": x_star,
        "replicas": replicas,
        "mean_curve": mean_curve,
        "lo68": lo68,
        "hi68": hi68,
        "lo95": lo95,
        "hi95": hi95,
        "member_losses": member_losses,
    }

    if save_member_preds:
        res["member_curves"] = member_curves

    return res