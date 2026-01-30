from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class RBFKernel1D(Kernel):
    """
    1D RBF kernel with optional small-x rescaling:

        K0(x,y) = sigma2 * exp( - (x-y)^2 / (2*ell^2) )
        K(x,y)  = (x^alpha)(y^alpha) * K0(x,y)

    Trainable hyperparameters (theta):
        theta = [alpha, log_ell, log_sigma2]
    """

    def __init__(
        self,
        alpha=-0.2,
        ell=1.0,
        sigma2=1.0,
        alpha_bounds=(-1.0, 0.0),
        ell_bounds=(1e-6, 1e6),
        sigma2_bounds=(1e-6, 1e6),
        x_floor=1e-12,
    ):
        self.alpha = float(alpha)
        self.ell = float(ell)
        self.sigma2 = float(sigma2)

        self.alpha_bounds = alpha_bounds
        self.ell_bounds = ell_bounds
        self.sigma2_bounds = sigma2_bounds
        self.x_floor = float(x_floor)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    @property
    def hyperparameter_ell(self):
        return Hyperparameter("ell", "numeric", self.ell_bounds)

    @property
    def hyperparameter_sigma2(self):
        return Hyperparameter("sigma2", "numeric", self.sigma2_bounds)

    @property
    def theta(self):
        # NOTE: alpha is stored directly; ell and sigma2 in log-space
        return np.array([self.alpha, np.log(self.ell), np.log(self.sigma2)])

    @theta.setter
    def theta(self, theta):
        self.alpha = float(theta[0])
        self.ell = float(np.exp(theta[1]))
        self.sigma2 = float(np.exp(theta[2]))

    @property
    def bounds(self):
        return np.array(
            [
                self.alpha_bounds,
                np.log(self.ell_bounds),
                np.log(self.sigma2_bounds),
            ],
            dtype=float,
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if X.shape[1] != 1:
            raise ValueError("Only 1D inputs (n,1).")
        Y = X if Y is None else np.atleast_2d(Y)
        if Y.shape[1] != 1:
            raise ValueError("Only 1D inputs (m,1).")

        x = X[:, 0][:, None]          # (n,1)
        y = Y[:, 0][None, :]          # (1,m)

        diff2 = (x - y) ** 2
        ell2 = max(self.ell, 1e-300) ** 2

        # base RBF part K0
        expo0 = np.exp(-0.5 * diff2 / ell2)

        K0 = self.sigma2 * expo0

        # small-x rescaling
        x_safe = np.maximum(x, self.x_floor)
        y_safe = np.maximum(y, self.x_floor)
        scale = (x_safe ** self.alpha) * (y_safe ** self.alpha)

        K = scale * K0

        if not eval_gradient:
            return K

        # ---- gradients w.r.t [alpha, log_ell, log_sigma2]

        eps = 1e-300  # numerical guard (keep if you want safety)

        # alpha
        logx = np.log(x_safe)
        logy = np.log(y_safe)
        dK_dalpha = K * (logx + logy)

        # sigma2 (K linear in sigma2)
        sigma2_safe = max(self.sigma2, eps)
        dK_dsigma2 = K / sigma2_safe

        # ell (raw)
        ell_safe = max(self.ell, eps)
        ell2 = ell_safe * ell_safe
        dK_dell = K * diff2 / (ell2 * ell_safe)   # = K * diff2 / ell^3

        grad = np.stack([dK_dalpha, dK_dell, dK_dsigma2], axis=2)
        return K, grad

    def diag(self, X):
        X = np.atleast_2d(X)
        x = np.maximum(X[:, 0], self.x_floor)
        # For x=y: exp(0)=1, so diag = sigma2 * x^(2 alpha)
        return self.sigma2 * (x ** (2 * self.alpha))

    def is_stationary(self):
        # Base RBF is stationary, but the x^alpha scaling makes it non-stationary
        return False



def Kxx_rbf_pytensor(
    xgrid_t: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l0: pt.TensorVariable,
    sigma2: pt.TensorVariable,
    *,
    x_floor: float = 1e-12,
):
    """
    RBF kernel with small-x rescaling, matching RBFKernel1D:

      K0(x,y) = sigma2 * exp( - (x-y)^2 / (2*l0^2) )
      K(x,y)  = (x^alpha)(y^alpha) * K0(x,y)

    xgrid_t: (Ngrid,)
    returns: (Ngrid, Ngrid)
    """
    X = xgrid_t.reshape((-1, 1))
    x = X[:, 0][:, None]
    y = X[:, 0][None, :]

    diff2 = (x - y) ** 2

    # guard l0 for numerical safety
    l0_safe = pt.maximum(l0, 1e-12)
    ell2 = l0_safe * l0_safe

    K0 = sigma2 * pt.exp(-0.5 * diff2 / ell2)

    # small-x rescaling (stable)
    x_safe = pt.maximum(x, x_floor)
    y_safe = pt.maximum(y, x_floor)
    scale = (x_safe**alpha) * (y_safe**alpha)

    return scale * K0

def Kxy_rbf_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    l0: float,
    sigma2: float,
    *,
    x_floor: float = 1e-12,
) -> np.ndarray:
    """
    Fast NumPy RBF kernel matrix K(X, Y) matching RBFKernel1D:
    K = (x^alpha)(y^alpha) * sigma2 * exp( - (x-y)^2 / (2*l0^2) )

    X: (N,1), Y: (M,1)
    """
    x = X[:, 0][:, None]
    y = Y[:, 0][None, :]

    diff2 = (x - y) ** 2
    l0_safe = max(float(l0), 1e-12)
    ell2 = l0_safe * l0_safe

    K0 = float(sigma2) * np.exp(-0.5 * diff2 / ell2)

    x_safe = np.maximum(x, x_floor)
    y_safe = np.maximum(y, x_floor)
    scale = (x_safe ** float(alpha)) * (y_safe ** float(alpha))

    return scale * K0

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










