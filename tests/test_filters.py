import numpy as np

from kspace import FourierAnalysis, GaussianFilter, TopHatFilter


def test_gaussian_filter_matches_closed_form():
    ell = 3.0
    filt = GaussianFilter(ell, ndim=3)
    k = np.geomspace(1e-3, 10.0, 25)
    np.testing.assert_allclose(filt(k), np.exp(-0.5 * (k * ell) ** 2))
    assert filt(0.0) == 1.0


def test_top_hat_filter_matches_spherical_bessel_closed_form():
    # W_hat(k) = 3*j1(k*ell)/(k*ell) for the 3D spherical top hat,
    # with j1(z) = sin(z)/z**2 - cos(z)/z (Perrone et al. 2026, Eq. 41).
    ell = 2.0
    filt = TopHatFilter(ell, ndim=3)
    k = np.geomspace(1e-3, 10.0, 25)
    z = k * ell
    j1 = np.sin(z) / z**2 - np.cos(z) / z
    np.testing.assert_allclose(filt(k), 3.0 * j1 / z, atol=1e-10)
    assert filt(0.0) == 1.0


def test_top_hat_filter_1d_matches_sinc():
    ell = 1.5
    filt = TopHatFilter(ell, ndim=1)
    k = np.geomspace(1e-3, 10.0, 25)
    np.testing.assert_allclose(filt(k), np.sin(k * ell) / (k * ell), atol=1e-10)


def test_top_hat_filter_2d_matches_bessel_j1():
    from scipy.special import j1

    ell = 1.5
    filt = TopHatFilter(ell, ndim=2)
    k = np.geomspace(1e-3, 10.0, 25)
    np.testing.assert_allclose(filt(k), 2.0 * j1(k * ell) / (k * ell), atol=1e-10)


def test_decompose_filtered_reconstructs_field():
    width = np.array([100.0, 100.0, 100.0])
    ddims = [32, 32, 32]
    fa = FourierAnalysis(width, ddims)
    field = np.random.default_rng(0).normal(size=ddims)

    filt = GaussianFilter(10.0, ndim=3)
    filtered, residual = fa.decompose_filtered(field, filt)
    np.testing.assert_allclose(filtered + residual, field, atol=1e-10)
    np.testing.assert_allclose(fa.apply_filter(field, filt), filtered, atol=1e-10)


def test_apply_filter_preserves_mean():
    width = np.array([50.0, 50.0])
    ddims = [40, 40]
    fa = FourierAnalysis(width, ddims)
    field = np.random.default_rng(1).normal(loc=3.0, size=ddims)

    filtered = fa.apply_filter(field, TopHatFilter(5.0, ndim=2))
    np.testing.assert_allclose(filtered.mean(), field.mean(), atol=1e-10)


def test_apply_filter_rejects_dimension_mismatch():
    width = np.array([50.0, 50.0])
    ddims = [16, 16]
    fa = FourierAnalysis(width, ddims)
    field = np.zeros(ddims)
    try:
        fa.apply_filter(field, GaussianFilter(5.0, ndim=3))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched filter ndim")
