"""
Feature extraction for the spam dataset.

Dataset columns:
- words: total number of words in the email
- links: number of URLs in the email
- capital_words: number of fully-uppercase words (e.g., "URGENT", "FREE")
- spam_word_count: number of words from a predefined "spam keywords" list
"""

from __future__ import annotations
import re
from typing import Dict, List

# Simple URL detector (covers http(s) and www.* patterns)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Words: keep letters/numbers/apostrophes to avoid counting punctuation as words
WORD_RE = re.compile(r"[A-Za-z0-9']+")

# A small, reasonable spam-keyword list (can be extended)
SPAM_KEYWORDS: List[str] = [
    "free", "win", "winner", "prize", "cash", "money", "urgent", "offer", "limited",
    "discount", "bonus", "claim", "click", "buy", "cheap", "deal", "guarantee",
    "congratulations", "act", "now", "loan", "credit", "exclusive", "reward"
]
SPAM_SET = set(SPAM_KEYWORDS)

def extract_features(email_text: str) -> Dict[str, int]:
    """
    Convert raw email text into the same 4 features used in the provided CSV.

    Returns a dict with keys: words, links, capital_words, spam_word_count
    """
    if email_text is None:
        email_text = ""

    # Find words
    words_list = WORD_RE.findall(email_text)
    words = len(words_list)

    # Count links
    links = len(URL_RE.findall(email_text))

    # Count fully-uppercase words (length>1 to ignore single-letter words like "I")
    capital_words = sum(1 for w in words_list if len(w) > 1 and w.isalpha() and w.isupper())

    # Count spam keywords (case-insensitive)
    spam_word_count = 0
    for w in words_list:
        w_low = w.lower()
        if w_low in SPAM_SET:
            spam_word_count += 1

    return {
        "words": words,
        "links": links,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count,
    }
