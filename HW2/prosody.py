"""
prosody.py
----------
Rule-based prosody assignment.

For each phoneme in the sequence the module assigns:
  - duration_ms : segment duration in milliseconds
  - f0          : instantaneous fundamental frequency (Hz)

F0 contour
----------
A simple linear declination model is used (Pierrehumbert 1979 style):
  - Starts F0_START_SCALE × base_f0
  - Declines linearly to F0_END_SCALE × base_f0 over the utterance
  - Voiced phonemes only; unvoiced phonemes get f0 = 0

Duration scaling
----------------
A uniform scale factor can be applied to speed up / slow down the output.
Stressed-vowel lengthening is approximated by a fixed +30 ms bonus for
the ARPAbet vowel classes.
"""

import numpy as np

# Utterance-level F0 declination parameters
F0_START_SCALE: float = 1.10   # beginning of utterance
F0_END_SCALE:   float = 0.88   # end of utterance

# Optional global speaking-rate multiplier (1.0 = neutral)
DEFAULT_RATE_SCALE: float = 1.0

# Vowel set (ARPAbet) — these get a slight duration bonus
_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
}
VOWEL_DURATION_BONUS_MS: float = 20.0


def assign_prosody(
    phoneme_sequence: list,
    inventory: dict,
    rate_scale: float = DEFAULT_RATE_SCALE,
) -> list:
    """
    Assign per-phoneme prosodic parameters.

    Parameters
    ----------
    phoneme_sequence : list of ARPAbet phoneme strings
    inventory        : phoneme parameter dict (from phoneme_inventory)
    rate_scale       : global duration multiplier (>1 = slower, <1 = faster)

    Returns
    -------
    prosody : list of dicts, one per phoneme:
        {
            'phoneme'     : str,
            'duration_ms' : float,
            'f0'          : float   (Hz; 0 for unvoiced)
        }
    """
    # Filter to phonemes that exist in inventory
    valid = [ph for ph in phoneme_sequence if ph in inventory]
    n = len(valid)

    prosody = []
    for i, ph in enumerate(valid):
        entry = inventory[ph]

        # --- Duration ---
        dur = entry["duration_ms"]
        if ph in _VOWELS:
            dur += VOWEL_DURATION_BONUS_MS
        dur *= rate_scale

        # --- F0 with linear declination ---
        if entry["voiced"] and entry["f0"] > 0:
            t = i / max(n - 1, 1)                       # normalised position [0,1]
            scale = F0_START_SCALE + (F0_END_SCALE - F0_START_SCALE) * t
            f0 = entry["f0"] * scale
        else:
            f0 = 0.0

        prosody.append(
            {
                "phoneme":      ph,
                "duration_ms":  dur,
                "f0":           f0,
            }
        )

    return prosody
