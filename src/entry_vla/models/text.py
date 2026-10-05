"""Frozen instruction encoders.

``hashing`` is deterministic, download-free and dependency-free (default for
tests and smoke runs). ``hf`` wraps a frozen HuggingFace encoder (user-run:
loading it downloads weights); embeddings are computed under ``no_grad``.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np


class HashingTextEncoder:
    name = "hashing_bow/v1"
    frozen = True

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class HFTextEncoder:  # pragma: no cover - user-run (downloads weights)
    frozen = True

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.name = device, f"hf:{model_name}"
        self.dim = int(self.model.config.hidden_size)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            b = self.tok(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            return ((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy().astype(np.float32)


def make_text_encoder(cfg: dict):
    kind = cfg.get("type", "hashing")
    if kind == "hashing":
        return HashingTextEncoder(int(cfg.get("dim", 64)))
    if kind == "hf":
        return HFTextEncoder(cfg["model_name"], cfg.get("device", "cpu"))
    raise ValueError(f"unknown text encoder {kind!r}")
