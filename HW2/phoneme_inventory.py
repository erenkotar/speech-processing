"""
phoneme_inventory.py
--------------------
Pre-built phoneme → LPC parameter table.

Each phoneme entry is derived by constructing an all-pole filter whose
poles correspond to the formant frequencies of that sound (Peterson &
Barney 1952; Klatt 1980).  This avoids the need to record a speaker:
the synthesis will sound synthetic but is correct in its DSP design.

Formant data
  - Vowels      : Peterson & Barney (1952) male averages
  - Consonants  : Klatt (1980) synthesis rules
  - Fricatives  : approximated above 4 kHz (require fs ≥ 16 kHz)

Sampling rate used throughout: FS = 16 000 Hz
LPC order used throughout    : P  = 12
"""

import numpy as np

FS: int = 16_000   # samples per second
P:  int = 12       # LPC predictor order


# ---------------------------------------------------------------------------
# Formant → LPC conversion
# ---------------------------------------------------------------------------

def formants_to_lpc(
    formants: list,
    bandwidths: list,
    fs: int = FS,
    p: int = P,
) -> np.ndarray:
    """
    Build LPC predictor coefficients from a list of formant frequencies
    and bandwidths.

    Each formant F_k (Hz) with bandwidth B_k (Hz) corresponds to a
    complex-conjugate pole pair:

        r_k   = exp(-π B_k / fs)
        θ_k   = 2π F_k / fs
        poles : r_k · exp(±j θ_k)

    contributing a second-order section to A(z):

        A_k(z) = 1 − 2 r_k cos(θ_k) z^{-1} + r_k² z^{-2}

    Parameters
    ----------
    formants   : list of formant frequencies in Hz
    bandwidths : corresponding bandwidths in Hz
    fs         : sampling rate
    p          : desired LPC order (polynomial truncated / zero-padded)

    Returns
    -------
    a : LPC predictor coefficients [a_1, ..., a_p]
        Convention: s(n) = Σ a_k · s(n-k) + G · e(n)
        So A(z)·S(z) = G·E(z) with A(z) = 1 − a_1 z^{-1} − ... − a_p z^{-p}
    """
    A = np.array([1.0])   # running product polynomial in z^{-1} domain

    for F, B in zip(formants, bandwidths):
        F = min(F, fs / 2.0 - 1.0)   # clip to Nyquist
        r = np.exp(-np.pi * B / fs)
        theta = 2.0 * np.pi * F / fs
        section = np.array([1.0, -2.0 * r * np.cos(theta), r ** 2])
        A = np.convolve(A, section)

    # A is now [1, c_1, c_2, ..., c_m]; LPC coefficients are a_k = -c_k
    a = -A[1 : p + 1]                 # take coefficients 1..p
    if len(a) < p:
        a = np.pad(a, (0, p - len(a)))
    return a.astype(np.float64)


# ---------------------------------------------------------------------------
# Phoneme table
# ---------------------------------------------------------------------------
# Format per entry:
#   formants   : [F1, F2, F3]  Hz
#   bw         : [B1, B2, B3]  Hz  (bandwidths)
#   voiced     : bool
#   dur        : average duration in ms
#   f0         : base fundamental frequency Hz (0 for unvoiced)
#
# Formant data sources:
#   Vowels   — Peterson & Barney (1952), male averages
#   Nasals   — Fant (1960)
#   Stops    — Klatt (1980) target values (very approximate for plosives)
#   Fricatives — shaped noise; formants set high to approximate spectral tilt

_RAW: dict = {
    # ---- Vowels ----------------------------------------------------------
    "AA": dict(formants=[730, 1090, 2440], bw=[70,  90, 130], voiced=True,  dur=140, f0=120),
    "AE": dict(formants=[660, 1720, 2410], bw=[70,  90, 120], voiced=True,  dur=130, f0=115),
    "AH": dict(formants=[520, 1190, 2390], bw=[70,  90, 120], voiced=True,  dur=120, f0=120),
    "AO": dict(formants=[570,  840, 2410], bw=[70,  90, 120], voiced=True,  dur=130, f0=115),
    "AW": dict(formants=[640, 1190, 2390], bw=[80, 100, 130], voiced=True,  dur=200, f0=115),
    "AY": dict(formants=[730, 1090, 2440], bw=[80, 100, 130], voiced=True,  dur=200, f0=115),
    "EH": dict(formants=[530, 1840, 2480], bw=[70,  90, 110], voiced=True,  dur=110, f0=115),
    "ER": dict(formants=[490, 1350, 1690], bw=[80, 100, 120], voiced=True,  dur=130, f0=115),
    "EY": dict(formants=[530, 1840, 2480], bw=[70,  90, 110], voiced=True,  dur=160, f0=115),
    "IH": dict(formants=[390, 1990, 2550], bw=[60,  90, 110], voiced=True,  dur=100, f0=120),
    "IY": dict(formants=[270, 2290, 3010], bw=[60,  90, 110], voiced=True,  dur=150, f0=120),
    "OW": dict(formants=[450,  800, 2830], bw=[70,  90, 110], voiced=True,  dur=140, f0=115),
    "OY": dict(formants=[450,  800, 2830], bw=[70,  90, 110], voiced=True,  dur=200, f0=115),
    "UH": dict(formants=[640, 1190, 2390], bw=[80, 100, 120], voiced=True,  dur=100, f0=120),
    "UW": dict(formants=[310,  870, 2250], bw=[60,  80, 110], voiced=True,  dur=150, f0=120),
    # ---- Nasals ----------------------------------------------------------
    "M":  dict(formants=[280,  900, 2200], bw=[100,200,250], voiced=True,  dur=80,  f0=115),
    "N":  dict(formants=[280, 1700, 2600], bw=[100,200,250], voiced=True,  dur=70,  f0=115),
    "NG": dict(formants=[280, 2300, 3000], bw=[100,200,250], voiced=True,  dur=80,  f0=110),
    # ---- Approximants ----------------------------------------------------
    "L":  dict(formants=[340, 1000, 2800], bw=[100,150,200], voiced=True,  dur=70,  f0=115),
    "R":  dict(formants=[460, 1190, 1640], bw=[100,150,180], voiced=True,  dur=80,  f0=115),
    "W":  dict(formants=[300,  610, 2200], bw=[80, 100,150], voiced=True,  dur=80,  f0=115),
    "Y":  dict(formants=[300, 2200, 3000], bw=[80, 100,150], voiced=True,  dur=70,  f0=115),
    # ---- Voiced fricatives -----------------------------------------------
    "V":  dict(formants=[300, 1500, 3500], bw=[80, 300,500], voiced=True,  dur=80,  f0=110),
    "DH": dict(formants=[300, 1500, 3000], bw=[80, 300,500], voiced=True,  dur=80,  f0=110),
    "Z":  dict(formants=[300, 4500, 5500], bw=[80, 300,400], voiced=True,  dur=90,  f0=110),
    "ZH": dict(formants=[300, 2500, 3500], bw=[80, 300,400], voiced=True,  dur=90,  f0=110),
    # ---- Unvoiced fricatives ---------------------------------------------
    "F":  dict(formants=[1500, 3500, 5500], bw=[300,400,500], voiced=False, dur=90,  f0=0),
    "TH": dict(formants=[1500, 3000, 4500], bw=[300,400,500], voiced=False, dur=90,  f0=0),
    "S":  dict(formants=[4500, 5500, 6500], bw=[200,400,500], voiced=False, dur=100, f0=0),
    "SH": dict(formants=[2500, 3500, 4500], bw=[200,400,500], voiced=False, dur=100, f0=0),
    "HH": dict(formants=[ 800, 1200, 2400], bw=[200,300,400], voiced=False, dur=80,  f0=0),
    # ---- Voiced stops (approximated; real stops need burst + murmur) -----
    "B":  dict(formants=[ 300, 1000, 2200], bw=[200,300,400], voiced=True,  dur=80,  f0=110),
    "D":  dict(formants=[ 300, 1700, 2600], bw=[200,300,400], voiced=True,  dur=80,  f0=110),
    "G":  dict(formants=[ 300, 1500, 2500], bw=[200,300,400], voiced=True,  dur=80,  f0=110),
    # ---- Unvoiced stops --------------------------------------------------
    "P":  dict(formants=[ 500, 1500, 2500], bw=[200,300,400], voiced=False, dur=80,  f0=0),
    "T":  dict(formants=[ 500, 1800, 2800], bw=[200,300,400], voiced=False, dur=80,  f0=0),
    "K":  dict(formants=[ 500, 1500, 2500], bw=[200,300,400], voiced=False, dur=80,  f0=0),
    # ---- Affricates ------------------------------------------------------
    "CH": dict(formants=[2500, 3500, 4500], bw=[200,400,500], voiced=False, dur=100, f0=0),
    "JH": dict(formants=[ 300, 2500, 3500], bw=[100,300,400], voiced=True,  dur=100, f0=110),
}


# ---------------------------------------------------------------------------
# Build the runtime inventory
# ---------------------------------------------------------------------------

def build_inventory(fs: int = FS, p: int = P) -> dict:
    """
    Return a dict mapping phoneme string → parameter dict:
        {
            'a'          : np.ndarray  shape (p,)   LPC coefficients
            'gain'       : float                    excitation gain
            'voiced'     : bool
            'duration_ms': float
            'f0'         : float                    base F0 in Hz
        }
    """
    inventory = {}
    for phoneme, data in _RAW.items():
        a = formants_to_lpc(data["formants"], data["bw"], fs=fs, p=p)
        inventory[phoneme] = {
            "a":           a,
            "gain":        1.0,
            "voiced":      data["voiced"],
            "duration_ms": float(data["dur"]),
            "f0":          float(data["f0"]),
        }
    return inventory
