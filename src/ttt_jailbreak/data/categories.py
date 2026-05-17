"""Macro-category mapping for the cross-category transfer experiment.

The mapping clusters the per-row ``category`` strings in
``datasets/harmful_behaviors/harmful_behaviors_pair.csv`` into five macro
buckets used by the cross-category heatmap. Rows whose first category label
does not appear in ``THEME_MAP`` are assigned to ``"Other"``.
"""

from __future__ import annotations

THEME_MAP: dict[str, str] = {
    "hacking": "Cyber",
    "virus": "Cyber",
    "phising": "Cyber",
    "piracy": "Cyber",
    "government": "Cyber",
    "bomb": "Violence",
    "murder": "Violence",
    "terrorism": "Violence",
    "firearms": "Violence",
    "poisoning": "Violence",
    "poison": "Violence",
    "violence": "Violence",
    "dangerous activity": "Violence",
    "suicide": "Violence",
    "video game": "Violence",
    "fraud": "Crime",
    "financial": "Crime",
    "theft": "Crime",
    "identity theft": "Crime",
    "cheating": "Crime",
    "drugs": "Crime",
    "illegal manufacture": "Crime",
    "child pornograph": "Crime",
    "hate crime": "Harassment",
    "hate speech": "Harassment",
    "stalking": "Harassment",
    "libel": "Harassment",
    "racism": "Harassment",
    "social media": "Harassment",
    "misinformation": "Misinfo",
    "mental health": "Misinfo",
}


def get_macro_category(raw_category: str) -> str:
    """Map a comma-separated raw category string to its macro bucket."""
    first = raw_category.split(",")[0].strip()
    return THEME_MAP.get(first, "Other")
