from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

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
    theta: np.ndarray,          # (3,) -> [alpha, l0, sigma2]
    X_train: np.ndarray,        # (Ngrid,1)
    X_star: np.ndarray,         # (N*,1)
    FK: np.ndarray,             # W: (Ndat, Ngrid)
    CY: np.ndarray,             # C: (Ndat, Ndat)
    y: np.ndarray,              # (Ndat,)
    *,
    kernel: str = "gibbs",
    delta: float = 1e-5,
    x_floor: float = 1e-12,
    jitter_cyt: float = 1e-10,
    jitter_star: float = 1e-10,
    nu: float = 1.5,            # only for matern kernel
):
    alpha, l0, sigma2 = (float(theta[0]), float(theta[1]), float(theta[2]))

    if kernel == "gibbs":
        Kxx = Kxy_gibbs_numpy(X_train, X_train, alpha, l0, sigma2, delta=delta, x_floor=x_floor)
        Ksx = Kxy_gibbs_numpy(X_star,  X_train, alpha, l0, sigma2, delta=delta, x_floor=x_floor)
        Kss = Kxy_gibbs_numpy(X_star,  X_star,  alpha, l0, sigma2, delta=delta, x_floor=x_floor)
    elif kernel == "rbf":
        Kxx = Kxy_rbf_numpy(X_train, X_train, alpha, l0, sigma2)
        Ksx = Kxy_rbf_numpy(X_star,  X_train, alpha, l0, sigma2)
        Kss = Kxy_rbf_numpy(X_star,  X_star, alpha, l0, sigma2)
    elif kernel == "matern":
        Kxx = Kxy_matern_numpy(X_train, X_train, alpha, l0, sigma2, nu=nu, x_floor=x_floor)
        Ksx = Kxy_matern_numpy(X_star,  X_train, alpha, l0, sigma2, nu=nu, x_floor=x_floor)
        Kss = Kxy_matern_numpy(X_star,  X_star, alpha, l0, sigma2, nu=nu, x_floor=x_floor)
    else:
        raise ValueError(f"Unknown kernel={kernel}")

    # CYT = FK Kxx FK^T + CY
    CYT = FK @ Kxx @ FK.T + CY
    CYT = 0.5 * (CYT + CYT.T)
    CYT = CYT + jitter_cyt * np.eye(CYT.shape[0])

    L = cholesky(CYT)

    def solve_CYT(B):
        v = solve_triangular(L, B, lower=True)
        return solve_triangular(L.T, v, lower=False)

    v = solve_CYT(y)                 # CYT^{-1} y
    m_star = Ksx @ FK.T @ v          # (N*,)

    A = solve_CYT(FK @ Ksx.T)        # CYT^{-1}(FK Kxs) -> (Ndat, N*)
    K_star = Kss - Ksx @ FK.T @ A    # (N*, N*)

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
    nu: float = 1.5,            # only for matern kernel
    max_samples: int | None = None,
):
    rng = np.random.default_rng(seed)

    if max_samples is not None and theta_samples.shape[0] > max_samples:
        idx = rng.choice(theta_samples.shape[0], size=max_samples, replace=False)
        theta_samples = theta_samples[idx]

    S = theta_samples.shape[0]
    Nstar = X_star.shape[0]

    replicas = np.empty((S, Nstar), dtype=float)
    means = np.empty((S, Nstar), dtype=float)
    vars_f = np.empty((S, Nstar), dtype=float)

    for s, theta in enumerate(tqdm(theta_samples, desc="Sampling posterior f*")):
        m_star, K_star = posterior_fstar(
            theta, X_train, X_star, FK, CY, y,
            kernel=kernel,
            delta=delta,
            x_floor=x_floor,
            jitter_cyt=jitter_cyt,
            jitter_star=jitter_star,
            nu = nu,
        )
        replicas[s] = _mvnormal_stable(rng, m_star, K_star, jitter0=jitter_star)
        means[s] = m_star
        vars_f[s] = np.clip(np.diag(K_star), 0.0, np.inf)

    return replicas, means, vars_f