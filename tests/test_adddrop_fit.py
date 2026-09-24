"""Tests for the --dark help text and the smoothing-width warning in adddrop_fit.py.

All outputs go to pytest's tmp_path; committed data/ and images/ are not touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import adddrop_fit as af  # noqa: E402
from make_example_sweep import through_port  # noqa: E402

WARN_MARK = "окно сглаживания"
NGL = 1150.0  # um, as in make_example_sweep defaults


def synthetic_trace(kappa2, loss_db, lam1=1545.0, lam2=1555.0, step_pm=1.0):
    """Clean through-port trace on a uniform grid (one sample per bin)."""
    lam = np.arange(lam1, lam2, step_pm * 1e-3)
    t = np.sqrt(1 - kappa2)
    a = 10 ** (-loss_db / 20)
    return lam, through_port(lam, t, a, NGL, phi0=0.7), t, a


def analytic_fwhm_nm(t, a, lam0=1550.0):
    x = t * t * a
    finesse = np.pi * np.sqrt(x) / (1 - x)
    return lam0**2 / (NGL * 1e3) / finesse


# ----------------------------------------------------------------------
# --dark help / docstring
# ----------------------------------------------------------------------
def dark_help():
    for action in af.build_parser()._actions:
        if "--dark" in action.option_strings:
            return action
    raise AssertionError("--dark option not found")


def test_dark_help_matches_behaviour():
    action = dark_help()
    assert action.default == 0.0
    text = " ".join(action.help.split())
    # Old text claimed "None -> фитируется"; the dark level is never fitted.
    assert "None" not in text
    assert "НЕ фитируется" in text
    assert "0.0" in text
    full_help = " ".join(af.build_parser().format_help().split())
    assert "None -> фитируется" not in full_help


def test_dark_is_a_fixed_offset_and_zero_means_none():
    lam = np.linspace(1549, 1551, 200)
    p = [0.9, 0.95, NGL, 0.3, 1.0, 0.0, 0.0]
    base = af.through_model(p, lam, 1550.0, dark=0.0)
    np.testing.assert_array_equal(base, af.through_model(p, lam, 1550.0, dark=None))
    np.testing.assert_allclose(af.through_model(p, lam, 1550.0, dark=0.006),
                               base + 0.006)


def test_docstring_uses_correct_ngL0_flag():
    assert "--ngL0" in af.__doc__
    assert "--ngl0" not in af.__doc__


# ----------------------------------------------------------------------
# resonance width and the smoothing check
# ----------------------------------------------------------------------
def test_narrowest_dip_fwhm_matches_finesse_high_f():
    # FSR/F is the dip FWHM only at high finesse (here F ~ 70).
    lam, T, t, a = synthetic_trace(0.02, 0.2)
    fwhm = af.narrowest_dip_fwhm(lam, T)
    assert fwhm == pytest.approx(analytic_fwhm_nm(t, a), rel=0.1)


def test_narrowest_dip_fwhm_low_finesse_is_wide():
    # F ~ 1.5: the dip is not Lorentzian, FSR/F does not apply; the half-depth
    # width is still hundreds of pm, far above 3 x 60 pm.
    lam, T, *_ = synthetic_trace(0.8, 2.0)
    fwhm = af.narrowest_dip_fwhm(lam, T)
    assert 0.3 < fwhm < 2.09


def test_narrowest_dip_fwhm_no_dips():
    lam = np.linspace(1550, 1551, 500)
    assert af.narrowest_dip_fwhm(lam, np.ones_like(lam)) is None


def test_check_smoothing_narrow_warns_wide_does_not():
    lam, T, *_ = synthetic_trace(0.02, 0.2)
    msg = af.check_smoothing(af.narrowest_dip_fwhm(lam, T), bin_pm=1.0, smooth_pm=60.0)
    assert msg is not None and WARN_MARK in msg and "--smooth 60" in msg

    lam, T, *_ = synthetic_trace(0.8, 2.0)
    assert af.check_smoothing(af.narrowest_dip_fwhm(lam, T),
                              bin_pm=1.0, smooth_pm=60.0) is None


def test_check_smoothing_threshold():
    # FWHM 30 samples on a 1 pm grid: 10 samples is the limit.
    assert af.check_smoothing(0.030, bin_pm=1.0, smooth_pm=10.0) is None
    assert af.check_smoothing(0.030, bin_pm=1.0, smooth_pm=11.0) is not None
    assert af.check_smoothing(0.030, bin_pm=1.0, smooth_pm=0.0) is None
    assert af.check_smoothing(None, bin_pm=1.0, smooth_pm=60.0) is None


def test_load_sweep_equals_bin_then_smooth(tmp_path):
    lam, T, *_ = synthetic_trace(0.3, 1.0, step_pm=0.5)
    csv = tmp_path / "s.csv"
    np.savetxt(csv, np.c_[lam, T], delimiter=",", header="wl,v", comments="")
    ref = af.load_sweep(csv, smooth_pm=20.0)
    lb, Tb = af.load_sweep(csv, smooth_pm=0.0)
    got = af.smooth_trace(lb, Tb, smooth_pm=20.0)
    np.testing.assert_array_equal(ref[0], got[0])
    np.testing.assert_array_equal(ref[1], got[1])


# ----------------------------------------------------------------------
# CLI end to end on realistic synthetic sweeps (noise, ripple, envelope)
# ----------------------------------------------------------------------
def make_sweep(tmp_path, name, kappa2, loss_db):
    out = tmp_path / name
    subprocess.run([sys.executable, str(REPO / "make_example_sweep.py"),
                    "--kappa2", str(kappa2), "--loss", str(loss_db),
                    "--lam1", "1540", "--lam2", "1575", "--out", str(out)],
                   check=True, capture_output=True, cwd=tmp_path)
    return out


def run_cli(tmp_path, csv, *extra):
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(REPO / "adddrop_fit.py"), str(csv),
                        "--dark", "0.006", *extra],
                       check=True, capture_output=True, text=True,
                       cwd=tmp_path, env=env)
    assert (tmp_path / "images" / "fig_adddrop_fit.png").is_file()
    return r.stdout


def test_cli_narrow_resonance_warns(tmp_path):
    csv = make_sweep(tmp_path, "narrow.csv", 0.02, 0.2)
    out = run_cli(tmp_path, csv, "--smooth", "60")
    assert WARN_MARK in out


def test_cli_narrow_resonance_small_smooth_no_warning(tmp_path):
    csv = make_sweep(tmp_path, "narrow.csv", 0.02, 0.2)
    out = run_cli(tmp_path, csv, "--smooth", "3")
    assert WARN_MARK not in out


def test_cli_wide_resonance_no_warning(tmp_path):
    csv = make_sweep(tmp_path, "wide.csv", 0.8, 2.0)
    out = run_cli(tmp_path, csv, "--smooth", "60")
    assert WARN_MARK not in out
