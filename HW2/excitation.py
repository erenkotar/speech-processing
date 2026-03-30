"""
excitation.py
-------------
Excitation signal generator for the LPC synthesis filter.

Two types of excitation are produced (source-filter model):

  Voiced   → periodic impulse train at the target fundamental period T0
  Unvoiced → white Gaussian noise

Both are scaled to unit RMS before being passed to the synthesis filter;
the excitation gain G (stored per phoneme) controls output loudness.

Optionally a simple Rosenberg-style glottal pulse shape can be used
instead of ideal impulses to reduce the harshness of voiced sounds.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Voiced excitation
# ---------------------------------------------------------------------------

def _glottal_pulse(length: int) -> np.ndarray:
    """
    Approximate one period of a Rosenberg glottal pulse (normalised to
    peak amplitude 1).  The pulse has an asymmetric shape with a slow
    rise and a sharper closure.
    """
    if length <= 1:
        return np.array([1.0])
    t = np.linspace(0, 1, length)
    # Rising phase (0 → 0.6): half-cosine
    # Closing phase (0.6 → 1): steeper half-cosine
    pulse = np.where(
        t < 0.6,
        0.5 * (1 - np.cos(np.pi * t / 0.6)),
        0.5 * (1 + np.cos(np.pi * (t - 0.6) / 0.4)),
    )
    return pulse


def voiced_excitation(
    n_samples: int,
    f0: float,
    fs: int,
    use_glottal_pulse: bool = True,
) -> np.ndarray:
    """
    Generate a voiced excitation of length n_samples.

    Parameters
    ----------
    n_samples         : number of output samples
    f0                : fundamental frequency in Hz
    fs                : sampling rate in Hz
    use_glottal_pulse : if True, use a Rosenberg pulse; else ideal impulses

    Returns
    -------
    excitation : np.ndarray, shape (n_samples,)
    """
    if f0 <= 0 or n_samples <= 0:
        return np.zeros(n_samples)

    period = int(round(fs / f0))
    excitation = np.zeros(n_samples)

    if use_glottal_pulse:
        pulse = _glottal_pulse(period)
        pos = 0
        while pos < n_samples:
            end = min(pos + period, n_samples)
            seg_len = end - pos
            excitation[pos:end] += pulse[:seg_len]
            pos += period
    else:
        # Ideal impulse train
        excitation[::period] = 1.0

    return excitation


# ---------------------------------------------------------------------------
# Unvoiced excitation
# ---------------------------------------------------------------------------

def unvoiced_excitation(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate white Gaussian noise for unvoiced sounds.

    Parameters
    ----------
    n_samples : number of output samples
    rng       : optional numpy random Generator (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(n_samples)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def make_excitation(
    n_samples: int,
    voiced: bool,
    f0: float,
    fs: int,
    use_glottal_pulse: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an excitation signal of length n_samples.

    Parameters
    ----------
    n_samples         : output length in samples
    voiced            : True → periodic; False → noise
    f0                : fundamental frequency Hz (used only when voiced=True)
    fs                : sampling rate Hz
    use_glottal_pulse : use shaped glottal pulse (voiced only)
    rng               : random generator (for unvoiced reproducibility)

    Returns
    -------
    excitation : np.ndarray, unit-RMS normalised, shape (n_samples,)
    """
    if n_samples <= 0:
        return np.array([])

    if voiced and f0 > 0:
        exc = voiced_excitation(n_samples, f0, fs, use_glottal_pulse)
    else:
        exc = unvoiced_excitation(n_samples, rng)

    # Normalise to unit RMS
    rms = np.sqrt(np.mean(exc ** 2))
    if rms > 1e-12:
        exc /= rms
    return exc
