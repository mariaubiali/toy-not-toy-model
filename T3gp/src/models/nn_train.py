from __future__ import annotations
from typing import Any, Dict

import numpy as np
import torch

from losses import LossContext, make_loss
from models.nn_models import MLPFModel
from sklearn.model_selection import train_test_split


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
    xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()

    nncfg = cfg.get("nn", {})
    hidden = nncfg.get("hidden", [64, 64])
    activation = str(nncfg.get("activation", "tanh"))
    lr = float(nncfg.get("lr", 1e-3))
    epochs = int(nncfg.get("epochs", 3000))
    weight_decay = float(nncfg.get("weight_decay", 0.0))
    seed = int(cfg.get("seed", 0))
    dropout = float(nncfg.get("dropout", 0.0))
    use_preproc = bool(nncfg.get("use_preproc", True))
    init_alpha = float(nncfg.get("init_alpha", 1.0))
    init_beta  = float(nncfg.get("init_beta", 3.0))
    transforms = cfg.get("transforms", {})

    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    out_dim = 2 if ("het" in loss_name) else 1

    model = MLPFModel(
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        out_dim=out_dim,
        use_preproc=use_preproc,
        init_alpha=init_alpha,
        init_beta=init_beta,
        transforms=transforms,
    ).to(device=device, dtype=dtype)

    jitter = float(cfg.get("kernel", {}).get("jitter", 1e-10))

    replica = int(cfg.get("replica", 0))
    idx_all = np.arange(xgrid.shape[0])

    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=replica * 1000,
    )

    x_train_t = torch.from_numpy(xgrid[train_idx]).view(-1, 1).to(device=device, dtype=dtype)
    x_val_t   = torch.from_numpy(xgrid[val_idx]).view(-1, 1).to(device=device, dtype=dtype)

    W_train_t = W_t[:, train_idx]
    W_val_t   = W_t[:, val_idx]

    xgrid_train_t = x_train_t.squeeze(1)
    t3_ref_int = np.trapz(xt3_true[train_idx] / xgrid_train_t, xgrid_train_t)

    ctx = LossContext(
        W=W_train_t,
        C=C_t,
        y=y_t,
        xgrid=xgrid_train_t,
        t3_ref_int=t3_ref_int,
        jitter=jitter,
    )
    loss_fn, extra_params = make_loss(cfg, ctx, device=device, dtype=dtype)

    params = list(model.parameters()) + list(extra_params.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    loss_hist = []
    Cinv_t = torch.linalg.inv(C_t)

    for ep in range(epochs):
        opt.zero_grad()

        out = model(x_train_t)
        f_train = out["f_grid"]
        y_pred = W_train_t @ f_train

        loss = loss_fn(y_pred, out)
        loss.backward()
        opt.step()

        with torch.no_grad():
            out_val = model(x_val_t)
            f_val = out_val["f_grid"]
            y_val_pred = W_val_t @ f_val
            r_val = (y_val_pred - y_t)
            _ = (r_val @ (Cinv_t @ r_val)).item()

        if (ep + 1) % max(1, epochs // 20) == 0:
            loss_hist.append(float(loss.detach().cpu().numpy()))

    model.eval()
    with torch.no_grad():
        out_full = model(x_t)
        f_full = out_full["f_grid"].detach().cpu().numpy()

        res = {
            "xgrid": xgrid.astype(np.float64),
            "f_grid_mean": f_full.astype(np.float64),
            "loss_history": np.array(loss_hist, dtype=np.float64),
            "train_idx": train_idx,
            "val_idx": val_idx,
        }

        if "logvar_f_grid" in out_full:
            res["f_grid_var"] = np.exp(
                out_full["logvar_f_grid"].detach().cpu().numpy()
            ).astype(np.float64)

    return res


def train_nn_ensemble_forward(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ens = cfg.get("nn", {}).get("ensemble", {})
    if not bool(ens.get("enabled", True)):
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
        cfg_i["seed"] = base_seed + seed_offset * i
        cfg_i["replica"] = i

        out_i = train_nn_forward(ds, cfg_i)
        member_curves.append(out_i["f_grid_mean"])
        member_losses.append(out_i["loss_history"])

    replicas = np.stack(member_curves, axis=0)
    xgrid = out_i["xgrid"]

    if subsample_members is not None and replicas.shape[0] > subsample_members:
        rng = np.random.default_rng(base_seed + seed_offset)
        replicas = replicas[rng.choice(replicas.shape[0], subsample_members, replace=False)]

    res = {
        "xgrid": xgrid,
        "replicas": replicas,
        "mean_curve": replicas.mean(axis=0),
        "lo68": np.percentile(replicas, 16.0, axis=0),
        "hi68": np.percentile(replicas, 84.0, axis=0),
        "lo95": np.percentile(replicas, 2.5, axis=0),
        "hi95": np.percentile(replicas, 97.5, axis=0),
        "member_losses": member_losses,
    }

    if save_member_preds:
        res["member_curves"] = member_curves

    return res
