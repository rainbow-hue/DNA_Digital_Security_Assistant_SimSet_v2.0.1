# -*- coding: utf-8 -*-
"""Tiny, dependency-light linear models (numpy only).

* Vectorizer       feature dict -> dense float32 row (vocabulary with min_df)
* SoftmaxHead      multinomial logistic regression (one label per row)
* MultiLabelHead   one-vs-rest logistic regression (a set of labels per row)

Linear models on interpretable features are deliberate: the weights can be
printed and argued with (see `explain`), they train in seconds, and the whole
model fits in a few MB, small enough to ship to a phone or an edge function.
"""
import json
from collections import Counter

import numpy as np


class Vectorizer:
    def __init__(self, min_df=2, keep_prefixes=("C:", "AC:", "S:", "REQ:", "AST:", "T:", "G:", "J:", "AUD:", "POL", "LEX:", "BIAS", "HAS_FACTS", "RULE")):
        self.min_df = min_df
        self.keep = keep_prefixes
        self.index = {}

    def fit(self, rows):
        df = Counter(k for r in rows for k in r)
        feats = sorted(k for k, c in df.items() if c >= self.min_df or k.startswith(self.keep))
        self.index = {k: i for i, k in enumerate(feats)}
        return self

    def transform(self, rows):
        X = np.zeros((len(rows), len(self.index)), dtype=np.float32)
        for i, r in enumerate(rows):
            for k, v in r.items():
                j = self.index.get(k)
                if j is not None:
                    X[i, j] = v
        return X

    def to_json(self):
        return {"min_df": self.min_df, "features": sorted(self.index, key=self.index.get)}

    @classmethod
    def from_json(cls, d):
        v = cls(min_df=d["min_df"])
        v.index = {k: i for i, k in enumerate(d["features"])}
        return v


def _adam(grad_fn, W, b, iters, lr):
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        gW, gb = grad_fn(W, b)
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return W, b


class SoftmaxHead:
    def __init__(self, classes=None, l2=1e-3, iters=300, lr=0.05):
        self.classes = list(classes or [])
        self.l2, self.iters, self.lr = l2, iters, lr
        self.W = self.b = None

    def fit(self, X, y, sample_weight=None):
        if not self.classes:
            self.classes = sorted(set(y))
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((len(y), len(self.classes)), dtype=np.float32)
        for i, c in enumerate(y):
            Y[i, idx[c]] = 1.0
        sw = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        sw = sw / sw.sum()
        W = np.zeros((X.shape[1], len(self.classes)), dtype=np.float32)
        b = np.zeros(len(self.classes), dtype=np.float32)

        def grad(W, b):
            P = self._softmax(X @ W + b)
            G = (P - Y) * sw[:, None]
            return X.T @ G + self.l2 * W, G.sum(0)

        self.W, self.b = _adam(grad, W, b, self.iters, self.lr)
        return self

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def dist(self, x):
        p = self.predict_proba(x[None, :])[0]
        return {c: float(p[i]) for i, c in enumerate(self.classes)}

    def to_arrays(self, name):
        return {name + ".W": self.W, name + ".b": self.b}, {"type": "softmax", "classes": self.classes}


class MultiLabelHead:
    def __init__(self, labels=None, l2=1e-3, iters=300, lr=0.05):
        self.labels = list(labels or [])
        self.l2, self.iters, self.lr = l2, iters, lr
        self.W = self.b = None

    def fit(self, X, ys, sample_weight=None):
        if not self.labels:
            self.labels = sorted(set(l for s in ys for l in s))
        idx = {c: i for i, c in enumerate(self.labels)}
        Y = np.zeros((len(ys), len(self.labels)), dtype=np.float32)
        for i, s in enumerate(ys):
            for l in s:
                if l in idx:
                    Y[i, idx[l]] = 1.0
        sw = np.ones(len(ys), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        sw = sw / sw.sum()
        W = np.zeros((X.shape[1], len(self.labels)), dtype=np.float32)
        b = np.zeros(len(self.labels), dtype=np.float32)

        def grad(W, b):
            P = 1.0 / (1.0 + np.exp(-(X @ W + b)))
            G = (P - Y) * sw[:, None]
            return X.T @ G + self.l2 * W, G.sum(0)

        self.W, self.b = _adam(grad, W, b, self.iters, self.lr)
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.W + self.b)))

    def dist(self, x):
        p = self.predict_proba(x[None, :])[0]
        return {c: float(p[i]) for i, c in enumerate(self.labels)}

    def to_arrays(self, name):
        return {name + ".W": self.W, name + ".b": self.b}, {"type": "multilabel", "labels": self.labels}


class ModelBundle:
    """All heads + their vectorizers, saved as one .npz plus one .json."""

    def __init__(self):
        self.vecs = {}     # name -> Vectorizer
        self.heads = {}    # name -> (vec_name, head)
        self.extra = {}    # json-serialisable extras (playbooks, thresholds ...)

    def add(self, name, vec_name, head):
        self.heads[name] = (vec_name, head)

    def save(self, prefix):
        arrays, meta = {}, {"vecs": {k: v.to_json() for k, v in self.vecs.items()}, "heads": {}, "extra": self.extra}
        for name, (vn, h) in self.heads.items():
            a, m = h.to_arrays(name)
            arrays.update(a)
            m["vec"] = vn
            meta["heads"][name] = m
        np.savez_compressed(prefix + ".npz", **arrays)
        with open(prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, prefix):
        b = cls()
        with open(prefix + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        arr = np.load(prefix + ".npz")
        b.vecs = {k: Vectorizer.from_json(v) for k, v in meta["vecs"].items()}
        for name, m in meta["heads"].items():
            h = SoftmaxHead(m["classes"]) if m["type"] == "softmax" else MultiLabelHead(m["labels"])
            h.W, h.b = arr[name + ".W"], arr[name + ".b"]
            b.heads[name] = (m["vec"], h)
        b.extra = meta.get("extra", {})
        return b

    def vector(self, vec_name, feats):
        return self.vecs[vec_name].transform([feats])[0]

    def explain(self, head_name, feats, label, top=8):
        """Top features pushing `feats` towards `label` (for debugging / judges)."""
        vn, h = self.heads[head_name]
        vec = self.vecs[vn]
        cls = h.classes if isinstance(h, SoftmaxHead) else h.labels
        j = cls.index(label)
        contrib = []
        for k, v in feats.items():
            i = vec.index.get(k)
            if i is not None:
                contrib.append((float(v * h.W[i, j]), k))
        contrib.sort(reverse=True)
        return contrib[:top]
