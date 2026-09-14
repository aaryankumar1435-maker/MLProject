"""Prefix-based word autocomplete for the fingerspelling UI.

Backed by a static, frequency-ranked word list (data/word_list.txt - top
~9,900 English words, most common first, from the public-domain
first20hours/google-10000-english corpus) rather than a live dictionary
service, so the UI works fully offline and suggestion ranking is
deterministic. Frequency order is preserved from the source file, so
"first N matches" is already "N most common matches" with no scoring step
needed.
"""
from pathlib import Path


class WordCompleter:
    def __init__(self, word_list_path: Path):
        with open(word_list_path, "r", encoding="utf-8") as f:
            # Preserve file order (frequency rank); de-dupe defensively.
            seen = set()
            self.words = []
            for line in f:
                word = line.strip().upper()
                if word and word not in seen:
                    seen.add(word)
                    self.words.append(word)

    def suggest(self, prefix: str, limit: int = 5):
        """Words starting with `prefix`, most-frequent first. Exact-length
        match (i.e. the prefix is already a complete word) is still
        included and ranked normally, not forced to the top, since a
        signer may intend to keep spelling a longer word."""
        prefix = prefix.upper()
        if not prefix:
            return self.words[:limit]
        return [w for w in self.words if w.startswith(prefix)][:limit]

    def is_word(self, word: str) -> bool:
        return word.upper() in self.words
