"""
Fixed-scale isotropic low-pass filters for Fourier-space filtering of
fields, following Perrone et al. 2026 (arXiv:2601.06250), which filters a
field at a fixed physical scale ell to split it into a filtered ("bulk")
component and a residual ("turbulent", fluctuating) component.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, jv


class KSpaceFilter:
    """
    Base class for a fixed-scale, isotropic low-pass filter kernel,
    applied to a field by multiplication in Fourier space:
    filtered_hat(k) = W_hat(k) * data_hat(k) (Perrone et al. 2026, Eq. 46).

    Subclasses implement `__call__` as the Fourier-space multiplier
    W_hat(k), normalized so that W_hat(0) = 1 (i.e. the real-space kernel
    integrates to unity and the filtered field preserves the mean).

    Parameters
    ----------
    length : float
        The filter scale ell, in the same physical units as the grid
        (e.g. kpc).
    ndim : int, optional
        Number of dimensions (1, 2, or 3) of the real-space kernel.
        Default is 3.
    """

    def __init__(self, length, ndim=3):
        if ndim not in (1, 2, 3):
            raise ValueError("Invalid number of dimensions! Must be 1, 2, or 3.")
        self.length = length
        self.ndim = ndim

    def __call__(self, k):
        raise NotImplementedError


class GaussianFilter(KSpaceFilter):
    r"""
    Gaussian filter kernel at fixed scale `length` (:math:`\ell`):

    .. math::
        W_\ell(x) = (2\pi\ell^2)^{-n/2}
        \exp\left(-\frac{|x|^2}{2\ell^2}\right)

    in real space (Perrone et al. 2026, Eq. 5), with Fourier-space
    multiplier

    .. math::
        \hat{W}_\ell(k) = \exp\left(-\frac{k^2\ell^2}{2}\right)

    (Perrone et al. 2026, Eq. 42).
    """

    def __call__(self, k):
        k = np.asarray(k, dtype=float)
        return np.exp(-0.5 * (k * self.length) ** 2)


class TopHatFilter(KSpaceFilter):
    r"""
    Spherical top-hat filter kernel at fixed scale `length` (:math:`\ell`):
    uniform over a ball of radius :math:`\ell` in real space
    (Perrone et al. 2026, Eq. 4), with Fourier-space multiplier

    .. math::
        \hat{W}_\ell(k) = \Gamma(n/2 + 1)
        \left(\frac{2}{k\ell}\right)^{n/2} J_{n/2}(k\ell)

    the n-dimensional generalization of Perrone et al. 2026's Eq. 41,
    which gives the n=3 case :math:`\hat{W}_\ell(k) = 3 j_1(k\ell)/(k\ell)`
    (:math:`j_1` the spherical Bessel function) in terms of the ordinary
    Bessel function :math:`J_{3/2}`.
    """

    def __call__(self, k):
        k = np.asarray(k, dtype=float)
        x = k * self.length
        nu = 0.5 * self.ndim
        with np.errstate(divide="ignore", invalid="ignore"):
            w = gamma(nu + 1.0) * (2.0 / x) ** nu * jv(nu, x)
        return np.where(x == 0.0, 1.0, w)
