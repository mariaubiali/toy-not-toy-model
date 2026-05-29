from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

try:
    from scipy.special import kv as besselk
    from scipy.special import gamma
except ImportError as e:
    raise ImportError(
        "This Matérn kernel implementation requires SciPy (scipy.special.kv, gamma). "
        "Install with: pip install scipy"
    ) from e


class MaternKernel1D(Kernel):
    """
    1D Matérn kernel with fixed nu and multiplicative power-law scaling:

        k(x, y) = (x^alpha) (y^alpha) * sigma2 * Matern_nu( |x-y| / l )

    where
        Matern_nu(r) = (2^{1-nu}/Gamma(nu)) * (z^nu K_nu(z)),
        z = sqrt(2*nu) * r / l

    Trainable hyperparameters:
        theta = [alpha, l, sigma2]

    Fixed parameter:
        nu (smoothness)
    """

    def __init__(
        self,
        *,
        nu: float = 1.5,  # fixed
        alpha: float = -0.2,
        l: float = 1e-5,
        sigma2: float = 1.0,
        alpha_bounds=(-1.0, 0.0),
        l_bounds=(1e-6, 1e6),
        sigma2_bounds=(1e-6, 1e6),
        x_floor: float = 1e-12,
    ):
        if nu <= 0:
            raise ValueError("nu must be > 0.")

        self.nu = float(nu)  # fixed
        self.alpha = float(alpha)
        self.l = float(l)
        self.sigma2 = float(sigma2)

        self.alpha_bounds = alpha_bounds
        self.l_bounds = l_bounds
        self.sigma2_bounds = sigma2_bounds
        self.x_floor = float(x_floor)

        # constant prefactor for Matérn
        self._matern_c = (2.0 ** (1.0 - self.nu)) / gamma(self.nu)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    @property
    def hyperparameter_l(self):
        return Hyperparameter("l", "numeric", self.l_bounds)

    @property
    def hyperparameter_sigma2(self):
        return Hyperparameter("sigma2", "numeric", self.sigma2_bounds)

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.alpha, self.l, self.sigma2], dtype=float)

    @theta.setter
    def theta(self, theta):
        self.alpha = float(theta[0])
        self.l = float(theta[1])
        self.sigma2 = float(theta[2])

    @property
    def bounds(self) -> np.ndarray:
        return np.array(
            [self.alpha_bounds, self.l_bounds, self.sigma2_bounds], dtype=float
        )

    def _scale(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (scale, logxy) where:
            scale = exp(alpha * (log x + log y))
            logxy  = log x + log y
        """
        x_safe = np.maximum(x, self.x_floor)
        y_safe = np.maximum(y, self.x_floor)
        logxy = np.log(x_safe) + np.log(y_safe)
        log_scale = np.clip(self.alpha * logxy, -700.0, 700.0)
        return np.exp(log_scale), logxy

    def _matern_base_and_dbase_dl(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes:
            base(r) = Matern_nu(r/l)  (unit-variance, i.e., base(0)=1)
            dbase/dl

        r: pairwise distances, shape (N,M), nonnegative
        """
        l = self.l
        nu = self.nu

        # z = sqrt(2 nu) * r / l
        z = (np.sqrt(2.0 * nu) * r) / l

        base = np.ones_like(z)

        # For z>0 use definition; for z=0, limit is 1
        mask = z > 0.0
        if np.any(mask):
            zm = z[mask]
            # f(z)= z^nu K_nu(z)
            # besselk is modified bessel function of 2nd kind
            f = (zm**nu) * besselk(nu, zm)
            base_m = self._matern_c * f
            base[mask] = base_m

        # d(base)/dl:
        # base = c * f(z), z = a*r/l, dz/dl = -z/l
        # f'(z) = nu z^{nu-1} K_nu(z) + z^nu K_nu'(z)
        # K_nu'(z) = -0.5 (K_{nu-1}(z) + K_{nu+1}(z))
        # => f'(z) = nu z^{nu-1}K_nu(z) - 0.5 z^nu (K_{nu-1}+K_{nu+1})
        # d(base)/dl = c * f'(z) * dz/dl = c * f'(z) * (-z/l)
        dbase_dl = np.zeros_like(z)

        if np.any(mask):
            zm = z[mask]
            Kn = besselk(nu, zm)
            Knm1 = besselk(nu - 1.0, zm)
            Knp1 = besselk(nu + 1.0, zm)

            fprime = nu * (zm ** (nu - 1.0)) * Kn - 0.5 * (zm**nu) * (Knm1 + Knp1)
            dbase_dz = self._matern_c * fprime
            dz_dl = -zm / l
            dbase_dl[mask] = dbase_dz * dz_dl

        return base, dbase_dl

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if X.shape[1] != 1:
            raise ValueError("Only 1D inputs (n,1).")

        Y = X if Y is None else np.atleast_2d(Y)
        if Y.shape[1] != 1:
            raise ValueError("Only 1D inputs (m,1).")

        x = X[:, 0][:, None]
        y = Y[:, 0][None, :]

        r = np.abs(x - y)

        base, dbase_dl = self._matern_base_and_dbase_dl(r)

        scale, logxy = self._scale(x, y)

        K0 = self.sigma2 * base
        K = scale * K0

        if not eval_gradient:
            return K

        # gradients w.r.t [alpha, l, sigma2]
        dK_dsigma2 = K / self.sigma2
        dK_dalpha = K * logxy

        # dK/dl = scale * sigma2 * dbase/dl
        dK_dl = scale * self.sigma2 * dbase_dl

        grad = np.stack([dK_dalpha, dK_dl, dK_dsigma2], axis=2)
        return K, grad

    def diag(self, X):
        X = np.atleast_2d(X)
        x = np.maximum(X[:, 0], self.x_floor)
        # base(0)=1 => diag = sigma2 * x^(2 alpha)
        return self.sigma2 * (x ** (2.0 * self.alpha))

    def is_stationary(self) -> bool:
        # alpha scaling makes it nonstationary unless alpha==0
        return False


def Kxx_matern_pytensor(
    xgrid_t: pt.TensorVariable,
    alpha: pt.TensorVariable,
    l: pt.TensorVariable,
    sigma2: pt.TensorVariable,
    *,
    nu: float = 1.5,  # fixed (restricted)
    amp: str = "legacy",  # "legacy" or "prefactor"
    beta: pt.TensorVariable | None = None,  # only used if amp=="prefactor"
    x_floor: float = 1e-12,
):
    """
    Free-standing PyTensor Kxx builder (NumPy-like closed forms):
    K(x_i, x_j) for xgrid_t = (Ngrid,).

    Restricts nu to {0.5, 1.5, 2.5} (fixed).
    Kernel:
        k(x,y) = (x^alpha)(y^alpha) * sigma2 * Matern_nu(|x-y|/l)
    """
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("nu must be one of {0.5, 1.5, 2.5}.")

    X = xgrid_t.reshape((-1, 1))
    x = X[:, 0][:, None]
    y = X[:, 0][None, :]

    r = pt.abs(x - y)

    # ---- Matérn core (unit variance) -----------------------------------------
    # simplified implementation for certain Bessel functions
    if nu == 0.5:
        z = r / l
        base = pt.exp(-z)
    elif nu == 1.5:
        z = pt.sqrt(3.0) * r / l
        base = (1.0 + z) * pt.exp(-z)
    else:  # nu == 2.5
        z = pt.sqrt(5.0) * r / l
        base = (1.0 + z + (z * z) / 3.0) * pt.exp(-z)

    K0 = sigma2 * base

    if amp == "none":
        return K0

    # ---- alpha scaling --------------------------------
    x_safe = pt.maximum(x, x_floor)
    y_safe = pt.maximum(y, x_floor)

    if amp == "legacy":

        scale = (x_safe**alpha) * (y_safe**alpha)
        return K0

    if amp == "prefactor":
        if beta is None:
            raise ValueError("beta must be provided for amp='prefactor'")
        pre_x = (x_safe**alpha) * ((1.0 - x_safe) ** beta)
        pre_y = (y_safe**alpha) * ((1.0 - y_safe) ** beta)
        scale = pre_x * pre_y

        return scale * K0

    raise ValueError(f"Unknown amp={amp!r}")


def Kxy_matern_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    l: float,
    sigma2: float,
    *,
    nu: float = 1.5,  # fixed (restricted)
    amp: str = "legacy",  # "legacy" or "prefactor"
    beta: float | None = None,  # only used if amp=="prefactor"
    x_floor: float = 1e-12,
) -> np.ndarray:
    """
    Fast NumPy Matérn kernel matrix K(X, Y),
    Restricts nu to {0.5, 1.5, 2.5} (fixed).

    X: (N,1), Y: (M,1)
    k(x,y) = (x^alpha)(y^alpha) * sigma2 * Matern_nu(|x-y|/l)
    """
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("nu must be one of {0.5, 1.5, 2.5}.")

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape[1] != 1 or Y.shape[1] != 1:
        raise ValueError("Only 1D inputs (n,1) and (m,1).")

    x = X[:, 0][:, None]
    y = Y[:, 0][None, :]

    r = np.abs(x - y)

    # ---- Matérn core (unit variance) -----------------------------------------
    if nu == 0.5:
        z = r / l
        base = np.exp(-z)
    elif nu == 1.5:
        z = np.sqrt(3.0) * r / l
        base = (1.0 + z) * np.exp(-z)
    else:  # nu == 2.5
        z = np.sqrt(5.0) * r / l
        base = (1.0 + z + (z * z) / 3.0) * np.exp(-z)

    K0 = sigma2 * base

    if amp == "none":
        return K0

    # ---- alpha scaling --------------------------------
    x_safe = np.maximum(x, x_floor)
    y_safe = np.maximum(y, x_floor)

    if amp == "legacy":

        log_scale = alpha * (np.log(x_safe) + np.log(y_safe))
        log_scale = np.clip(log_scale, -700.0, 700.0)

        return K0

    if amp == "prefactor":
        if beta is None:
            raise ValueError("beta must be provided for amp='prefactor'")
        pre_x = (x_safe**alpha) * ((1.0 - x_safe) ** beta)
        pre_y = (y_safe**alpha) * ((1.0 - y_safe) ** beta)
        scale = pre_x * pre_y
        return scale * K0

    raise ValueError(f"Unknown amp={amp!r}")
