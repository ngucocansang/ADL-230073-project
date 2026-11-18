"""
A tiny, dependency-free BM25 keyword index.

Dense embeddings are good at meaning ("what jobs can I get?") but weak at exact
tokens such as course codes ("CS311", "MATH202"). BM25 is the opposite, so the
engine fuses both rankings (hybrid retrieval).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_STOPWORDS = set("""a an the of to in on for and or is are was were be by with as at
from that this these those it its what which who whom how why when where do does
did can could should would will i me my we our you your they them their there
about into than then so if not no any all each per""".split())


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # Keep course codes together: "CS 311" / "cs311" -> "cs311"
    text = re.sub(r"\b([a-z]{2,4})\s?(\d{3}[a-z]?)\b", r"\1\2", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
