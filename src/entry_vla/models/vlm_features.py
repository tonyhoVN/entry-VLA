"""Frozen Qwen3-VL features for the entry diffusion condition (shared by the cache embedding and deployment).

text_feat (D,): bare instruction text, mean of the last decoder layer over its tokens.
crop_feat (V, D): each crop alone in a one-image chat prompt, mean of the last layer over its image tokens.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class VLMFeatures:
    def __init__(self, model, processor, device: str = "cuda:0"):
        self.model, self.proc, self.device = model, processor, device
        self.proc.tokenizer.padding_side = "right"
        self.img_tok = model.config.image_token_id

    @classmethod
    def load(cls, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16,
                                                                local_files_only=True).to(device).eval()
        return cls(model, proc, device)

    def _encode(self, texts, images=None, image_only=False) -> np.ndarray:
        import torch
        inp = self.proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            h = self.model.model(**inp).last_hidden_state.float()
        m = (inp["input_ids"] == self.img_tok) if image_only else inp["attention_mask"].bool()
        return ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)).cpu().numpy()

    def text(self, instruction: str) -> np.ndarray:
        return self._encode([instruction])[0]

    def crops(self, crops: Sequence[np.ndarray], batch: int = 8) -> np.ndarray:
        from PIL import Image
        prompt = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}], tokenize=False,
                                               add_generation_prompt=False)
        ims = [Image.fromarray(c) for c in crops]
        return np.concatenate([self._encode([prompt] * len(ims[j:j + batch]), images=ims[j:j + batch], image_only=True)
                               for j in range(0, len(ims), batch)])
