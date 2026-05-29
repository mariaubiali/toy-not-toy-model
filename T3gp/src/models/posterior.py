from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from typing import Optional

from models.gibbs import Kxy_gibbs_numpy
from models.rbf import Kxy_rbf_numpy
from models.matern import Kxy_matern_numpy
from tqdm import tqdm


def _mvnormal_stable(rng, mean, cov, *, jitter0=1e-10, max_tries=8):
    """
    Sample from N(mean, cov) with increasing diagonal jitter if Cholesky fails.
    """
    cov = 0.5 * (cov + cov.T)
    jitter = float(jitter0)

    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            z = rng.standard_normal(cov.shape[0])
            return mean + L @ z
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # last resort: eigenvalue clip (keeps you moving, but note it slightly changes cov)
    w, V = np.linalg.eigh(cov)
    w_clipped = np.maximum(w, 1e-12)
    L = V @ np.diag(np.sqrt(w_clipped))
    z = rng.standard_normal(cov.shape[0])
    return mean + L @ z


def posterior_fstar(
    theta: np.ndarray,  # (3,) [alpha,l0,sigma2] OR (4,) [alpha,beta,l0,sigma2]
    X_train: np.ndarray,  # (Ngrid,1)  kernel input (maybe logx)
    X_star: np.ndarray,  # (N*,1)     kernel input (maybe logx)
    FK: np.ndarray,  # W: (Ndat, Ngrid)
    CY: np.ndarray,  # C: (Ndat, Ndat)
    y: np.ndarray,  # (Ndat,)
    *,
    kernel: str = "gibbs",
    delta: float = 1e-5,
    x_floor: float = 1e-12,
    jitter_cyt: float = 1e-10,
    jitter_star: float = 1e-10,
    nu: float = 1.5,
    sr_a: np.ndarray | None = None,
    sr_ref: float | None = None,
    sr_tau2: float | None = None,
    pref_mode: str = "prefactor",  # legacy | prefactor_fixed_beta | prefactor_infer_beta
    x_train_phys: np.ndarray | None = None,  # (Ngrid,)
    x_star_phys: np.ndarray | None = None,  # (N*,)
):
    pmode = str(pref_mode).strip().lower()

    # ---- parse theta ----
    if pmode == "prefactor":
        if theta.shape[0] != 4:
            raise ValueError(
                f"pref_mode={pref_mode!r} expects theta=[alpha,beta,l0,sigma2], got {theta}"
            )
        alpha = float(theta[0])
        beta = float(theta[1])
        l0 = float(theta[2])
        sigma2 = float(theta[3])

    elif pmode == "legacy":
        if theta.shape[0] != 3:
            raise ValueError(
                f"pref_mode={pref_mode!r} expects theta=[alpha,l0,sigma2], got {theta}"
            )
        alpha = float(theta[0])
        l0 = float(theta[1])
        sigma2 = float(theta[2])

    elif pmode == "none":
        if theta.shape[0] != 2:
            raise ValueError(
                f"pref_mode={pref_mode!r} expects theta=[l0,sigma2], got {theta}"
            )
        l0 = float(theta[0])
        sigma2 = float(theta[1])
    else:
        raise ValueError(f"Unknown pref_mode={pref_mode!r}")

    # ---- choose physical x for prefactor (important if Logx is used) ----
    if x_train_phys is None:
        x_train_phys = np.asarray(X_train[:, 0], float)
    else:
        x_train_phys = np.asarray(x_train_phys, float).reshape(-1)

    if x_star_phys is None:
        x_star_phys = np.asarray(X_star[:, 0], float)
    else:
        x_star_phys = np.asarray(x_star_phys, float).reshape(-1)

    # ---- build kernel blocks ----
    # legacy: keep alpha scaling inside kernel
    # prefactor: disable internal scaling (alpha_kernel=0) and wrap with pre(x)pre(y)
    alpha_kernel = alpha if pmode == "legacy" else 0.0

    if kernel == "gibbs":
        K0_xx = Kxy_gibbs_numpy(
            X_train, X_train, alpha_kernel, l0, sigma2, delta=delta, x_floor=x_floor
        )
        K0_sx = Kxy_gibbs_numpy(
            X_star, X_train, alpha_kernel, l0, sigma2, delta=delta, x_floor=x_floor
        )
        K0_ss = Kxy_gibbs_numpy(
            X_star, X_star, alpha_kernel, l0, sigma2, delta=delta, x_floor=x_floor
        )
    elif kernel == "rbf":
        K0_xx = Kxy_rbf_numpy(
            X_train,
            X_train,
            alpha_kernel,
            l0,
            sigma2,
            amp="none" if pmode == "prefactor" else "legacy",
            x_floor=x_floor,
        )
        K0_sx = Kxy_rbf_numpy(
            X_star,
            X_train,
            alpha_kernel,
            l0,
            sigma2,
            amp="none" if pmode == "prefactor" else "legacy",
            x_floor=x_floor,
        )
        K0_ss = Kxy_rbf_numpy(
            X_star,
            X_star,
            alpha_kernel,
            l0,
            sigma2,
            amp="none" if pmode == "prefactor" else "legacy",
            x_floor=x_floor,
        )
    elif kernel == "matern":
        K0_xx = Kxy_matern_numpy(
            X_train, X_train, alpha_kernel, l0, sigma2, nu=nu, x_floor=x_floor
        )
        K0_sx = Kxy_matern_numpy(
            X_star, X_train, alpha_kernel, l0, sigma2, nu=nu, x_floor=x_floor
        )
        K0_ss = Kxy_matern_numpy(
            X_star, X_star, alpha_kernel, l0, sigma2, nu=nu, x_floor=x_floor
        )
    else:
        raise ValueError(f"Unknown kernel={kernel!r}")

    # ---- wrap with NN-like prefactor if requested ----
    if pmode in ("prefactor"):
        x_clip = 1e-12
        xtr = np.clip(x_train_phys, x_clip, 1.0 - x_clip)
        xst = np.clip(x_star_phys, x_clip, 1.0 - x_clip)

        # stable for x^alpha at tiny x
        xtr_s = np.maximum(xtr, x_floor)
        xst_s = np.maximum(xst, x_floor)

        pre_tr = (xtr_s**alpha) * ((1.0 - xtr) ** beta)  # (Ngrid,)
        pre_st = (xst_s**alpha) * ((1.0 - xst) ** beta)  # (N*,)

        Kxx = (pre_tr[:, None] * K0_xx) * pre_tr[None, :]
        Ksx = (pre_st[:, None] * K0_sx) * pre_tr[None, :]
        Kss = (pre_st[:, None] * K0_ss) * pre_st[None, :]
    else:
        Kxx, Ksx, Kss = K0_xx, K0_sx, K0_ss

    # ----------------------------
    # Optional sumrule pseudo-observation: augment FK, CY, y
    # ----------------------------
    if sr_a is not None:
        if sr_ref is None or sr_tau2 is None:
            raise ValueError(
                "If sr_a is provided, must also provide sr_ref and sr_tau2."
            )
        sr_a = np.asarray(sr_a, dtype=float).reshape(-1)
        if sr_a.shape[0] != FK.shape[1]:
            raise ValueError(
                f"sr_a must have length Ngrid={FK.shape[1]}, got {sr_a.shape[0]}."
            )

        FK = np.vstack([FK, sr_a[None, :]])
        y = np.concatenate([y, np.array([float(sr_ref)], dtype=float)], axis=0)

        CY_aug = np.zeros((CY.shape[0] + 1, CY.shape[1] + 1), dtype=float)
        CY_aug[: CY.shape[0], : CY.shape[1]] = CY
        CY_aug[-1, -1] = float(sr_tau2)
        CY = CY_aug

    # CYT = FK Kxx FK^T + CY
    CYT = FK @ Kxx @ FK.T + CY
    CYT = 0.5 * (CYT + CYT.T)
    CYT = CYT + jitter_cyt * np.eye(CYT.shape[0])

    L = cholesky(CYT)

    def solve_CYT(B):
        v = solve_triangular(L, B, lower=True)
        return solve_triangular(L.T, v, lower=False)

    v = solve_CYT(y)
    m_star = Ksx @ FK.T @ v

    A = solve_CYT(FK @ Ksx.T)
    K_star = Kss - Ksx @ FK.T @ A

    K_star = 0.5 * (K_star + K_star.T)
    K_star = K_star + jitter_star * np.eye(K_star.shape[0])

    return m_star, K_star


def sample_replicas(
    theta_samples: np.ndarray,  # (S,3)
    X_train: np.ndarray,
    X_star: np.ndarray,
    FK: np.ndarray,
    CY: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "gibbs",
    seed: int = 0,
    delta: float = 1e-5,
    x_floor: float = 1e-12,
    jitter_cyt: float = 1e-10,
    jitter_star: float = 1e-10,
    nu: float = 1.5,  # only for matern kernel
    max_samples: int | None = None,
    sr_a=None,
    sr_ref=None,
    sr_tau2=None,
    pref_mode: str = "legacy",  # "legacy" | "prefactor"
    x_train_phys: Optional[np.ndarray] = None,  # shape (Ntrain,)
    x_star_phys: Optional[np.ndarray] = None,  # shape (Nstar,)
):
    rng = np.random.default_rng(seed)

    if max_samples is not None and theta_samples.shape[0] > max_samples:
        idx = rng.choice(theta_samples.shape[0], size=max_samples, replace=False)
        theta_samples = theta_samples[idx]

    pmode = str(pref_mode).strip().lower()

    S = theta_samples.shape[0]
    Nstar = X_star.shape[0]

    replicas = np.empty((S, Nstar), dtype=float)
    means = np.empty((S, Nstar), dtype=float)
    vars_f = np.empty((S, Nstar), dtype=float)

    for s, theta in enumerate(tqdm(theta_samples, desc="Sampling posterior f*")):
        m_star, K_star = posterior_fstar(
            theta,
            X_train,
            X_star,
            FK,
            CY,
            y,
            kernel=kernel,
            delta=delta,
            x_floor=x_floor,
            jitter_cyt=jitter_cyt,
            jitter_star=jitter_star,
            nu=nu,
            sr_a=sr_a,
            sr_ref=sr_ref,
            sr_tau2=sr_tau2,
            pref_mode=pmode,
            x_train_phys=x_train_phys,
            x_star_phys=x_star_phys,
        )
        replicas[s] = _mvnormal_stable(rng, m_star, K_star, jitter0=jitter_star)
        means[s] = m_star
        vars_f[s] = np.clip(np.diag(K_star), 0.0, np.inf)

    return replicas, means, vars_f
