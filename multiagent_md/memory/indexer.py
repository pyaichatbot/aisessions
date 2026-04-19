from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class TfidfIndex:
    """Dependency-free TF-IDF. Cheap, no model calls, works offline.

    Upgrade path: plug embeddings behind `rank()` without touching callers.
    """
    docs: dict[str, str] = field(default_factory=dict)
    _tf: dict[str, Counter] = field(default_factory=dict)
    _df: Counter = field(default_factory=Counter)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            old = _tokenize(self.docs[doc_id])
            for t in set(old):
                self._df[t] -= 1
                if self._df[t] <= 0:
                    del self._df[t]
        toks = _tokenize(text)
        self.docs[doc_id] = text
        tf = Counter(toks)
        self._tf[doc_id] = tf
        for t in tf:
            self._df[t] += 1

    def rank(self, query: str, topk: int = 6) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        q = _tokenize(query)
        n = len(self.docs)
        scores: dict[str, float] = {}
        for doc_id, tf in self._tf.items():
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log((1 + n) / (1 + self._df[term])) + 1
                s += (tf[term] / max(1, sum(tf.values()))) * idf
            if s > 0:
                scores[doc_id] = s
        return sorted(scores.items(), key=lambda kv: -kv[1])[:topk]
