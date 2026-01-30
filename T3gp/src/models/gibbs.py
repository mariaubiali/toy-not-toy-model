from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class GibbsKernel1D(Kernel):
    """
    1D Gibbs kernel with linear lengthscale:
        l(x) = l0 * (x + delta)

    Trainable hyperparameters:
        theta = [alpha, l0, sigma2]
    """

    def __init__(
        self,
        alpha: float = -0.2,
        l0: float = 1.0,
        delta: float = 1e-5,
        sigma2: float = 1.0,
        alpha_bounds=(-1.0, 0.0),
        l0_bounds=(1e-6, 1e6),
        sigma2_bounds=(1e-6, 1e6),
        x_floor: float = 1e-12,
    ):
        self.alpha = float(alpha)
        self.l0 = float(l0)
        self.delta = float(delta)
        self.sigma2 = float(sigma2)

        self.alpha_bounds = alpha_bounds
        self.l0_bounds = l0_bounds
        self.sigma2_bounds = sigma2_bounds
        self.x_floor = float(x_floor)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    @property
    def hyperparameter_l0(self):
        return Hyperparameter("l0", "numeric", self.l0_bounds)

    @property
    def hyperparameter_sigma2(self):
        return Hyperparameter("sigma2", "numeric", self.sigma2_bounds)

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.alpha, self.l0, self.sigma2], dtype=float)

    @theta.setter
    def theta(self, theta):
        self.alpha = float(theta[0])
        self.l0 = float(theta[1])
        self.sigma2 = float(theta[2])

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.alpha_bounds, self.l0_bounds, self.sigma2_bounds], dtype=float)

    def _ell(self, X: np.ndarray) -> np.ndarray:
        x = X[:, 0]
        ell = self.l0 * (x + self.delta)
        return np.maximum(ell, 1e-12)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if X.shape[1] != 1:
            raise ValueError("Only 1D inputs (n,1).")

        Y = X if Y is None else np.atleast_2d(Y)
        if Y.shape[1] != 1:
            raise ValueError("Only 1D inputs (m,1).")

        x = X[:, 0][:, None]
        y = Y[:, 0][None, :]

        ell_x = self._ell(X)[:, None]
        ell_y = self._ell(Y)[None, :]

        denom = ell_x**2 + ell_y**2
        diff2 = (x - y) ** 2

        denom_safe = np.maximum(denom, 1e-300)
        pref0 = np.sqrt(2.0 * ell_x * ell_y / denom_safe)
        expo0 = np.exp(-diff2 / denom_safe)
        K0 = self.sigma2 * pref0 * expo0

        x_safe = np.maximum(x, self.x_floor)
        y_safe = np.maximum(y, self.x_floor)

        logxy = np.log(x_safe) + np.log(y_safe)
        log_scale = np.clip(self.alpha * logxy, -700.0, 700.0)
        scale = np.exp(log_scale)

        K = scale * K0

        if not eval_gradient:
            return K

        dK_dsigma2 = K / self.sigma2
        dK_dalpha = K * logxy

        raw_ell_x = self.l0 * (X[:, 0] + self.delta)
        raw_ell_y = self.l0 * (Y[:, 0] + self.delta)
        gx = (raw_ell_x > 1e-12)[:, None].astype(float)
        gy = (raw_ell_y > 1e-12)[None, :].astype(float)

        d_ellx = gx * ell_x
        d_elly = gy * ell_y
        d_denom = 2.0 * ell_x * d_ellx + 2.0 * ell_y * d_elly

        term_pref = 0.5 * (gx + gy - d_denom / denom_safe)
        term_exp = diff2 * d_denom / (denom_safe * denom_safe)
        dlogK0_dlogl0 = term_pref + term_exp

        dK_dlogl0 = K * dlogK0_dlogl0
        dK_dl0 = dK_dlogl0 / self.l0

        grad = np.stack([dK_dalpha, dK_dl0, dK_dsigma2], axis=2)
        return K, grad

    def diag(self, X):
        X = np.atleast_2d(X)
        x = np.maximum(X[:, 0], self.x_floor)
        return self.sigma2 * (x ** (2 * self.alpha))

    def is_stationary(self) -> bool:
        return False


def Kxx_gibbs_pytensor(
    xgrid_t: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l0: pt.TensorVariable,
    sigma2: pt.TensorVariable,
    *,
    delta: float = 1e-5,
    x_floor: float = 1e-12,
):
    """
    Free-standing PyTensor Kxx builder:
    K(x_i, x_j) for xgrid_t = (Ngrid,).
    """
    X = xgrid_t.reshape((-1, 1))
    x = X[:, 0][:, None]
    y = X[:, 0][None, :]

    ell_x = pt.maximum(l0 * (x + delta), 1e-12)
    ell_y = pt.maximum(l0 * (y + delta), 1e-12)

    denom = ell_x**2 + ell_y**2
    diff2 = (x - y) ** 2

    pref = pt.sqrt(2.0 * ell_x * ell_y / denom)
    expo = pt.exp(-diff2 / denom)
    K0 = sigma2 * pref * expo

    x_safe = pt.maximum(x, x_floor)
    y_safe = pt.maximum(y, x_floor)
    scale = (x_safe**alpha) * (y_safe**alpha)

    return scale * K0

def Kxy_gibbs_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    l0: float,
    sigma2: float,
    *,
    delta: float = 1e-5,
    x_floor: float = 1e-12,
) -> np.ndarray:
    """
    Fast NumPy Gibbs kernel matrix K(X, Y).
    X: (N,1), Y: (M,1)
    """
    x = X[:, 0][:, None]
    y = Y[:, 0][None, :]

    ell_x = np.maximum(l0 * (x + delta), 1e-12)
    ell_y = np.maximum(l0 * (y + delta), 1e-12)

    denom = ell_x**2 + ell_y**2
    d2 = (x - y) ** 2

    denom_safe = np.maximum(denom, 1e-300)
    pref = np.sqrt(2.0 * ell_x * ell_y / denom_safe)
    expo = np.exp(-d2 / denom_safe)
    K0 = sigma2 * pref * expo

    x_safe = np.maximum(x, x_floor)
    y_safe = np.maximum(y, x_floor)
    log_scale = alpha * (np.log(x_safe) + np.log(y_safe))
    log_scale = np.clip(log_scale, -700.0, 700.0)

    return np.exp(log_scale) * K0

def build_log_marginal_likelihood_pt(
    xgrid_t: pt.TensorVariable,
    W_t: pt.TensorVariable,
    C_t: pt.TensorVariable,
    y_t: pt.TensorVariable,
    Kxx_fn,
    *,
    jitter: float = 1e-10,
):
    """
    Kernel-agnostic PyTensor log marginal likelihood builder.

    Returns a function lml(alpha, l0, sigma2) -> scalar log p(y|theta),
    where Kxx_fn(xgrid_t, alpha, l0, sigma2) returns (Ngrid, Ngrid).
    """

    def lml(alpha, l0, sigma2):
        Kxx = Kxx_fn(xgrid_t, alpha, l0, sigma2)

        CYT = W_t @ Kxx @ W_t.T + C_t
        CYT = 0.5 * (CYT + CYT.T)
        CYT = CYT + jitter * pt.eye(CYT.shape[0])

        L = pt.linalg.cholesky(CYT)

        v = pt.linalg.solve_triangular(L, y_t, lower=True)
        quad = pt.dot(v, v)

        logdet = 2.0 * pt.sum(pt.log(pt.diag(L)))
        n = y_t.shape[0]
        return -0.5 * quad - 0.5 * logdet - 0.5 * n * pt.log(2 * np.pi)

    return lml










