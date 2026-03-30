"""
text_frontend.py
----------------
Front-end stage: raw text → flat sequence of ARPAbet phoneme strings.

Two G2P strategies are provided:

1. NLTK CMU Pronouncing Dictionary (preferred, install: pip install nltk)
   Covers ~130 000 English words.  Falls back to the built-in dict for
   unknown words.

2. Built-in mini-lexicon (always available, no extra dependencies).
   Sufficient for demonstrating the full synthesis pipeline.

Usage
-----
    from text_frontend import text_to_phonemes
    phones = text_to_phonemes("hello world")
    # ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D']
"""

import re

# ---------------------------------------------------------------------------
# Mini pronunciation lexicon  (ARPAbet, stress digits stripped)
# ---------------------------------------------------------------------------

MINI_LEXICON: dict = {
    "a":        ["AH"],
    "an":       ["AH", "N"],
    "the":      ["DH", "AH"],
    "of":       ["AH", "V"],
    "is":       ["IH", "Z"],
    "in":       ["IH", "N"],
    "it":       ["IH", "T"],
    "and":      ["AE", "N", "D"],
    "to":       ["T", "UW"],
    "be":       ["B", "IY"],
    # ---- Common words ----
    "hello":    ["HH", "AH", "L", "OW"],
    "world":    ["W", "ER", "L", "D"],
    "speech":   ["S", "P", "IY", "CH"],
    "linear":   ["L", "IH", "N", "IY", "ER"],
    "coding":   ["K", "OW", "D", "IH", "NG"],
    "this":     ["DH", "IH", "S"],
    "test":     ["T", "EH", "S", "T"],
    "one":      ["W", "AH", "N"],
    "two":      ["T", "UW"],
    "three":    ["TH", "R", "IY"],
    "four":     ["F", "AO", "R"],
    "five":     ["F", "AY", "V"],
    "six":      ["S", "IH", "K", "S"],
    "seven":    ["S", "EH", "V", "AH", "N"],
    "eight":    ["EY", "T"],
    "nine":     ["N", "AY", "N"],
    "ten":      ["T", "EH", "N"],
    "yes":      ["Y", "EH", "S"],
    "no":       ["N", "OW"],
    "good":     ["G", "UH", "D"],
    "day":      ["D", "EY"],
    "have":     ["HH", "AE", "V"],
    "now":      ["N", "AW"],
    "here":     ["HH", "IY", "R"],
    "there":    ["DH", "EH", "R"],
    "time":     ["T", "AY", "M"],
    "not":      ["N", "AA", "T"],
    "he":       ["HH", "IY"],
    "she":      ["SH", "IY"],
    "we":       ["W", "IY"],
    "they":     ["DH", "EY"],
    "you":      ["Y", "UW"],
    "my":       ["M", "AY"],
    "your":     ["Y", "AO", "R"],
    "how":      ["HH", "AW"],
    "what":     ["W", "AH", "T"],
    "when":     ["W", "EH", "N"],
    "where":    ["W", "EH", "R"],
    "who":      ["HH", "UW"],
    "why":      ["W", "AY"],
    "can":      ["K", "AE", "N"],
    "will":     ["W", "IH", "L"],
    "with":     ["W", "IH", "DH"],
    "for":      ["F", "AO", "R"],
    "from":     ["F", "R", "AH", "M"],
    "at":       ["AE", "T"],
    "by":       ["B", "AY"],
    "or":       ["AO", "R"],
    "on":       ["AO", "N"],
    "all":      ["AO", "L"],
    "are":      ["AA", "R"],
    "has":      ["HH", "AE", "Z"],
    "do":       ["D", "UW"],
    "up":       ["AH", "P"],
    "out":      ["AW", "T"],
    "if":       ["IH", "F"],
    "go":       ["G", "OW"],
    "see":      ["S", "IY"],
    "me":       ["M", "IY"],
    "so":       ["S", "OW"],
    "him":      ["HH", "IH", "M"],
    "her":      ["HH", "ER"],
    "its":      ["IH", "T", "S"],
    "his":      ["HH", "IH", "Z"],
    "our":      ["AW", "ER"],
    "use":      ["Y", "UW", "Z"],
    "but":      ["B", "AH", "T"],
    "as":       ["AE", "Z"],
    "get":      ["G", "EH", "T"],
    "was":      ["W", "AH", "Z"],
    "say":      ["S", "EY"],
    "make":     ["M", "EY", "K"],
    "know":     ["N", "OW"],
    "think":    ["TH", "IH", "NG", "K"],
    "come":     ["K", "AH", "M"],
    "look":     ["L", "UH", "K"],
    "want":     ["W", "AO", "N", "T"],
    "give":     ["G", "IH", "V"],
    "take":     ["T", "EY", "K"],
    "call":     ["K", "AO", "L"],
    "try":      ["T", "R", "AY"],
    "ask":      ["AE", "S", "K"],
    "need":     ["N", "IY", "D"],
    "feel":     ["F", "IY", "L"],
    "become":   ["B", "IH", "K", "AH", "M"],
    "leave":    ["L", "IY", "V"],
    "put":      ["P", "UH", "T"],
    "keep":     ["K", "IY", "P"],
    "let":      ["L", "EH", "T"],
    "begin":    ["B", "IH", "G", "IH", "N"],
    "show":     ["SH", "OW"],
    "hear":     ["HH", "IY", "R"],
    "play":     ["P", "L", "EY"],
    "run":      ["R", "AH", "N"],
    "move":     ["M", "UW", "V"],
    "live":     ["L", "IH", "V"],
    "believe":  ["B", "IH", "L", "IY", "V"],
    "hold":     ["HH", "OW", "L", "D"],
    "bring":    ["B", "R", "IH", "NG"],
    "happen":   ["HH", "AE", "P", "AH", "N"],
    "write":    ["R", "AY", "T"],
    "sit":      ["S", "IH", "T"],
    "stand":    ["S", "T", "AE", "N", "D"],
    "lose":     ["L", "UW", "Z"],
    "pay":      ["P", "EY"],
    "meet":     ["M", "IY", "T"],
    "include":  ["IH", "N", "K", "L", "UW", "D"],
    "set":      ["S", "EH", "T"],
    "learn":    ["L", "ER", "N"],
    "change":   ["CH", "EY", "N", "JH"],
    "lead":     ["L", "IY", "D"],
    "read":     ["R", "IY", "D"],
    "spend":    ["S", "P", "EH", "N", "D"],
    "grow":     ["G", "R", "OW"],
    "open":     ["OW", "P", "AH", "N"],
    "walk":     ["W", "AO", "K"],
    "win":      ["W", "IH", "N"],
    "offer":    ["AO", "F", "ER"],
    "remember": ["R", "IH", "M", "EH", "M", "B", "ER"],
    "love":     ["L", "AH", "V"],
    "decide":   ["D", "IH", "S", "AY", "D"],
    "people":   ["P", "IY", "P", "AH", "L"],
    "into":     ["IH", "N", "T", "UW"],
    "year":     ["Y", "IH", "R"],
    "way":      ["W", "EY"],
    "may":      ["M", "EY"],
    "just":     ["JH", "AH", "S", "T"],
    "about":    ["AH", "B", "AW", "T"],
    "over":     ["OW", "V", "ER"],
    "after":    ["AE", "F", "T", "ER"],
    "also":     ["AO", "L", "S", "OW"],
    "than":     ["DH", "AE", "N"],
    "that":     ["DH", "AE", "T"],
    "more":     ["M", "AO", "R"],
    "other":    ["AH", "DH", "ER"],
    "some":     ["S", "AH", "M"],
    "these":    ["DH", "IY", "Z"],
    "could":    ["K", "UH", "D"],
    "should":   ["SH", "UH", "D"],
    "would":    ["W", "UH", "D"],
    "well":     ["W", "EH", "L"],
    "been":     ["B", "IH", "N"],
    "their":    ["DH", "EH", "R"],
    "first":    ["F", "ER", "S", "T"],
    "new":      ["N", "UW"],
    "old":      ["OW", "L", "D"],
    "same":     ["S", "EY", "M"],
    "each":     ["IY", "CH"],
    "any":      ["EH", "N", "IY"],
    "many":     ["M", "EH", "N", "IY"],
    "then":     ["DH", "EH", "N"],
    "most":     ["M", "OW", "S", "T"],
    "long":     ["L", "AO", "NG"],
    "down":     ["D", "AW", "N"],
    "back":     ["B", "AE", "K"],
    "only":     ["OW", "N", "L", "IY"],
    "little":   ["L", "IH", "T", "AH", "L"],
    "even":     ["IY", "V", "AH", "N"],
    "still":    ["S", "T", "IH", "L"],
    "never":    ["N", "EH", "V", "ER"],
    "before":   ["B", "IH", "F", "AO", "R"],
    "between":  ["B", "IH", "T", "W", "IY", "N"],
    "under":    ["AH", "N", "D", "ER"],
    "very":     ["V", "EH", "R", "IY"],
    "around":   ["AH", "R", "AW", "N", "D"],
    "without":  ["W", "IH", "DH", "AW", "T"],
    "through":  ["TH", "R", "UW"],
    "during":   ["D", "UH", "R", "IH", "NG"],
    "own":      ["OW", "N"],
    "home":     ["HH", "OW", "M"],
    "place":    ["P", "L", "EY", "S"],
    "world":    ["W", "ER", "L", "D"],
    "house":    ["HH", "AW", "S"],
    "hand":     ["HH", "AE", "N", "D"],
    "part":     ["P", "AA", "R", "T"],
    "case":     ["K", "EY", "S"],
    "week":     ["W", "IY", "K"],
    "company":  ["K", "AH", "M", "P", "AH", "N", "IY"],
    "system":   ["S", "IH", "S", "T", "AH", "M"],
    "program":  ["P", "R", "OW", "G", "R", "AE", "M"],
    "question": ["K", "W", "EH", "S", "CH", "AH", "N"],
    "work":     ["W", "ER", "K"],
    "government":["G", "AH", "V", "ER", "N", "M", "AH", "N", "T"],
    "number":   ["N", "AH", "M", "B", "ER"],
    "night":    ["N", "AY", "T"],
    "point":    ["P", "OY", "N", "T"],
    "water":    ["W", "AO", "T", "ER"],
    "room":     ["R", "UW", "M"],
    "mother":   ["M", "AH", "DH", "ER"],
    "area":     ["EH", "R", "IY", "AH"],
    "money":    ["M", "AH", "N", "IY"],
    "story":    ["S", "T", "AO", "R", "IY"],
    "fact":     ["F", "AE", "K", "T"],
    "month":    ["M", "AH", "N", "TH"],
    "lot":      ["L", "AA", "T"],
    "right":    ["R", "AY", "T"],
    "study":    ["S", "T", "AH", "D", "IY"],
    "book":     ["B", "UH", "K"],
    "eye":      ["AY"],
    "life":     ["L", "AY", "F"],
    "kid":      ["K", "IH", "D"],
    "face":     ["F", "EY", "S"],
    "state":    ["S", "T", "EY", "T"],
    "family":   ["F", "AE", "M", "AH", "L", "IY"],
    "group":    ["G", "R", "UW", "P"],
    "city":     ["S", "IH", "T", "IY"],
    "community":["K", "AH", "M", "Y", "UW", "N", "IH", "T", "IY"],
    "name":     ["N", "EY", "M"],
    "president":["P", "R", "EH", "Z", "IH", "D", "AH", "N", "T"],
    "team":     ["T", "IY", "M"],
    "minute":   ["M", "IH", "N", "AH", "T"],
    "idea":     ["AY", "D", "IY", "AH"],
    "body":     ["B", "AA", "D", "IY"],
    "information":["IH","N","F","ER","M","EY","SH","AH","N"],
    "next":     ["N","EH","K","S","T"],
    "early":    ["ER","L","IY"],
    "important":["IH","M","P","AO","R","T","AH","N","T"],
    "problem":  ["P","R","AA","B","L","AH","M"],
    "example":  ["IH","G","Z","AE","M","P","AH","L"],
    "big":      ["B","IH","G"],
    "high":     ["HH","AY"],
    "small":    ["S","M","AO","L"],
    "different":["D","IH","F","ER","AH","N","T"],
    "large":    ["L","AA","R","JH"],
    "next":     ["N","EH","K","S","T"],
    "public":   ["P","AH","B","L","IH","K"],
    "real":     ["R","IY","L"],
    "best":     ["B","EH","S","T"],
    "free":     ["F","R","IY"],
    "sure":     ["SH","UH","R"],
    "top":      ["T","AA","P"],
    "line":     ["L","AY","N"],
    "end":      ["EH","N","D"],
    "air":      ["EH","R"],
    "black":    ["B","L","AE","K"],
    "white":    ["W","AY","T"],
    "turn":     ["T","ER","N"],
    "car":      ["K","AA","R"],
    "north":    ["N","AO","R","TH"],
    "south":    ["S","AW","TH"],
    "east":     ["IY","S","T"],
    "west":     ["W","EH","S","T"],
    "red":      ["R","EH","D"],
    "blue":     ["B","L","UW"],
    "green":    ["G","R","IY","N"],
    "mean":     ["M","IY","N"],
    "able":     ["EY","B","AH","L"],
    "bad":      ["B","AE","D"],
    "man":      ["M","AE","N"],
    "woman":    ["W","UH","M","AH","N"],
    "child":    ["CH","AY","L","D"],
    "king":     ["K","IH","NG"],
    "queen":    ["K","W","IY","N"],
    "cat":      ["K","AE","T"],
    "dog":      ["D","AO","G"],
    "bird":     ["B","ER","D"],
    "fish":     ["F","IH","SH"],
    "food":     ["F","UW","D"],
    "fire":     ["F","AY","ER"],
    "ice":      ["AY","S"],
    "sun":      ["S","AH","N"],
    "moon":     ["M","UW","N"],
    "star":     ["S","T","AA","R"],
    "sky":      ["S","K","AY"],
    "sea":      ["S","IY"],
    "land":     ["L","AE","N","D"],
    "tree":     ["T","R","IY"],
    "flower":   ["F","L","AW","ER"],
    "stone":    ["S","T","OW","N"],
    "music":    ["M","Y","UW","Z","IH","K"],
    "voice":    ["V","OY","S"],
    "sound":    ["S","AW","N","D"],
    "human":    ["HH","Y","UW","M","AH","N"],
    "machine":  ["M","AH","SH","IY","N"],
    "language":  ["L","AE","NG","G","W","AH","JH"],
    "natural":  ["N","AE","CH","ER","AH","L"],
    "process":  ["P","R","AA","S","EH","S"],
    "signal":   ["S","IH","G","N","AH","L"],
    "filter":   ["F","IH","L","T","ER"],
    "model":    ["M","AA","D","AH","L"],
    "method":   ["M","EH","TH","AH","D"],
    "analysis":  ["AH","N","AE","L","AH","S","IH","S"],
    "synthesis": ["S","IH","N","TH","AH","S","IH","S"],
    "frequency": ["F","R","IY","K","W","AH","N","S","IY"],
    "spectrum":  ["S","P","EH","K","T","R","AH","M"],
    "predict":   ["P","R","IH","D","IH","K","T"],
    "digital":   ["D","IH","JH","AH","T","AH","L"],
    "audio":     ["AO","D","IY","OW"],
    "text":      ["T","EH","K","S","T"],
    "phone":     ["F","OW","N"],
    "noise":     ["N","OY","Z"],
    "pitch":     ["P","IH","CH"],
    "rate":      ["R","EY","T"],
    "frame":     ["F","R","EY","M"],
    "window":    ["W","IH","N","D","OW"],
    "error":     ["EH","R","ER"],
    "output":    ["AW","T","P","UH","T"],
    "input":     ["IH","N","P","UH","T"],
    "sample":    ["S","AE","M","P","AH","L"],
}

# ---------------------------------------------------------------------------
# Optional NLTK-based G2P
# ---------------------------------------------------------------------------

def _try_nltk_g2p(word: str) -> list | None:
    """Attempt NLTK CMU dict lookup.  Returns None if NLTK unavailable."""
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()
        entries = d.get(word.lower())
        if entries:
            # Strip stress digits and return first pronunciation
            return [re.sub(r"\d", "", p) for p in entries[0]]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_to_phonemes(text: str, use_nltk: bool = True) -> list:
    """
    Convert input text to a flat list of ARPAbet phoneme strings.

    Parameters
    ----------
    text      : raw English text
    use_nltk  : try NLTK cmudict first (falls back to mini lexicon)

    Returns
    -------
    phoneme_sequence : e.g. ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D']
    """
    text = normalize_text(text)
    words = text.split()
    sequence = []
    unknown = []

    for word in words:
        phones = None
        if use_nltk:
            phones = _try_nltk_g2p(word)
        if phones is None:
            phones = MINI_LEXICON.get(word)
        if phones is not None:
            sequence.extend(phones)
        else:
            unknown.append(word)

    if unknown:
        print(f"[frontend] WARNING — unknown words (skipped): {unknown}")

    return sequence
