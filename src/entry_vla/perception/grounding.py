"""Locate the object a task instruction requires the robot to interact with,
using Qwen3-VL (``Qwen/Qwen3-VL-4B-Instruct`` by default).

Qwen3-VL returns ``bbox_2d`` in coordinates normalised to [0, 1000] relative to
the input image (unlike Qwen2.5-VL, which used absolute pixels); boxes are
converted back to pixels of the given image here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

GROUNDING_PROMPT = (
    'The robot is given the task: "{instruction}".\n'
    "Identify the object in the image that the robot must interact with to complete this task. "
    'Output its bounding box as JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}}]'
)


@dataclass
class Detection:
    label: str
    bbox_xyxy: np.ndarray  # pixels of the queried image
    raw_text: str
    extra: Dict[str, Any] = field(default_factory=dict)


def parse_boxes(text: str, width: int, height: int, scale: float = 1000.0) -> List[Detection]:
    """Parse ``[{"bbox_2d": [...], "label": ...}]`` (optionally in a ```json fence)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else [items]:
        if not isinstance(it, dict) or "bbox_2d" not in it:
            continue
        x1, y1, x2, y2 = (float(v) for v in it["bbox_2d"])
        box = np.array([x1 / scale * width, y1 / scale * height, x2 / scale * width, y2 / scale * height])
        box[[0, 2]] = np.clip(np.sort(box[[0, 2]]), 0, width - 1)
        box[[1, 3]] = np.clip(np.sort(box[[1, 3]]), 0, height - 1)
        out.append(Detection(label=str(it.get("label", "")), bbox_xyxy=box, raw_text=text))
    return out


class Qwen3VLGrounder:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda:0",
                 dtype: str = "bfloat16", max_new_tokens: int = 256, local_files_only: bool = True):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model_id, self.device, self.max_new_tokens = model_id, device, max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=getattr(torch, dtype), local_files_only=local_files_only).to(device).eval()

    def locate(self, image: np.ndarray, instruction: str) -> Optional[Detection]:
        import torch
        from PIL import Image
        pil = Image.fromarray(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": GROUNDING_PROMPT.format(instruction=instruction)}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        dets = parse_boxes(text, pil.width, pil.height)
        return dets[0] if dets else None
