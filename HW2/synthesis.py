"""
synthesis.py
------------
LPC synthesis filter and utterance assembler.

Each phoneme is synthesised frame-by-frame.  Within a phoneme, LPC
coefficients are held constant.  Across phoneme boundaries the
coefficients are linearly interpolated (coefficient interpolation)
so that the spectral envelope changes smoothly.

The all-pole synthesis filter implements:

    H(z) = G / A(z)  where  A(z) = 1 - a_1 z^{-1} - ... - a_p z^{-p}

Using scipy.signal.lfilter with:
    b = [G],   a_filter = [1, -a_1, -a_2, ..., -a_p]

Filter state (zi) is carried across frames so that there are no
discontinuities at frame boundaries.
"""

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from excitation import make_excitation


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

FRAME_SHIFT_MS: int = 5   # synthesis frame shift (ms) — controls interpolation resolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_samples(ms: float, fs: int) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _build_filter_poly(a: np.ndarray) -> np.ndarray:
    """
    Return the denominator polynomial for scipy.signal.lfilter.
    Convention: A(z) = 1 - a_1 z^{-1} - ... so the denominator is
    [1, -a_1, -a_2, ..., -a_p].
    """
    return np.concatenate([[1.0], -a])


# ---------------------------------------------------------------------------
# Per-phoneme synthesis
# ---------------------------------------------------------------------------

def synthesize_phoneme(
    entry: dict,
    prosody_item: dict,
    fs: int,
    prev_a: np.ndarray | None = None,
    zi: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple:
    """
    Synthesise one phoneme and return the audio segment plus updated
    filter state.

    Parameters
    ----------
    entry        : inventory entry dict {'a', 'gain', 'voiced', ...}
    prosody_item : {'phoneme', 'duration_ms', 'f0'}
    fs           : sampling rate Hz
    prev_a       : LPC coefficients of the preceding phoneme (for interp.)
    zi           : filter initial conditions (from previous call)
    rng          : random Generator for unvoiced excitation

    Returns
    -------
    segment : np.ndarray   synthesised samples for this phoneme
    zi      : np.ndarray   filter state to pass to the next call
    """
    a      = entry["a"]
    G      = entry["gain"]
    voiced = entry["voiced"]
    f0     = prosody_item["f0"]
    dur_ms = prosody_item["duration_ms"]

    n_total     = _ms_to_samples(dur_ms, fs)
    frame_shift = _ms_to_samples(FRAME_SHIFT_MS, fs)
    n_frames    = max(1, n_total // frame_shift)

    # Initialise filter state from the current LPC polynomial if not provided
    p = len(a)
    a_filter_init = _build_filter_poly(a)
    if zi is None:
        zi = lfilter_zi([G], a_filter_init) * 0.0  # zero initial state

    output_frames = []

    for frame_idx in range(n_frames):
        # Last frame absorbs any rounding remainder
        if frame_idx < n_frames - 1:
            n_samp = frame_shift
        else:
            n_samp = n_total - frame_shift * (n_frames - 1)
        n_samp = max(1, n_samp)

        # Linear interpolation of LPC coefficients across phoneme boundary
        if prev_a is not None:
            alpha   = frame_idx / n_frames
            a_frame = (1.0 - alpha) * prev_a + alpha * a
        else:
            a_frame = a

        # Generate and scale excitation
        exc = make_excitation(n_samp, voiced, f0, fs, rng=rng)
        exc = exc * G

        # All-pole filter (stateful across frames)
        a_filter = _build_filter_poly(a_frame)
        # Resize zi if polynomial length changed (boundary interpolation may change p)
        expected_zi_len = max(len(a_filter), 1) - 1
        if len(zi) != expected_zi_len:
            zi_new = np.zeros(expected_zi_len)
            zi_new[: min(len(zi), expected_zi_len)] = zi[: min(len(zi), expected_zi_len)]
            zi = zi_new

        frame_out, zi = lfilter([G], a_filter, exc, zi=zi)
        output_frames.append(frame_out)

    segment = np.concatenate(output_frames) if output_frames else np.array([])
    return segment, zi


# ---------------------------------------------------------------------------
# Full utterance synthesis
# ---------------------------------------------------------------------------

def synthesize_utterance(
    prosody_sequence: list,
    inventory: dict,
    fs: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthesise a complete utterance from a prosody sequence.

    Parameters
    ----------
    prosody_sequence : list of dicts from prosody.assign_prosody()
    inventory        : phoneme parameter dict from phoneme_inventory
    fs               : sampling rate Hz
    rng              : optional random Generator

    Returns
    -------
    audio : np.ndarray   complete speech waveform (float64)
    """
    if not prosody_sequence:
        return np.array([])

    segments = []
    prev_a   = None
    zi       = None

    for item in prosody_sequence:
        ph    = item["phoneme"]
        entry = inventory[ph]

        segment, zi = synthesize_phoneme(
            entry, item, fs,
            prev_a=prev_a,
            zi=zi,
            rng=rng,
        )
        segments.append(segment)
        prev_a = entry["a"]

    return np.concatenate(segments)
