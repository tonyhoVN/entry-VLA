#!/usr/bin/env python3
"""Serve a fine-tuned X-VLA checkpoint to an RLBench client (scripts/xvla/rollout_xvla_rlbench.py).

Transport: multiprocessing.connection on localhost with an authkey (stdlib on both sides: the
simulator env is Python 3.8, X-VLA runs in the `xvla` env). Request:
  {"images": [V x (H, W, 3) uint8 RGB, same view order as training], "proprio": (20,) float,
   "instruction": str}            -> reply {"actions": (num_actions, 20) float32}
  {"cmd": "close"}                -> server exits
Images follow the training pipeline exactly: JPEG (quality 95) round trip as in the export, resize to
224 bicubic, ImageNet normalisation (no colour jitter at test time).

  CUDA_VISIBLE_DEVICES=1 python scripts/xvla/xvla_policy_server.py --xvla-root ../X-VLA \
      --ckpt runs/xvla_close_microwave/ckpt-3000 --port 6010
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xvla-root", default=os.environ.get("XVLA_ROOT", "../X-VLA"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--authkey", default="entry-vla-xvla")
    ap.add_argument("--domain-id", type=int, default=None)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import xvla_rlbench
    domain_id = a.domain_id if a.domain_id is not None else xvla_rlbench.DEFAULT_DOMAIN_ID
    xvla_rlbench.register(a.xvla_root, domain_id)
    import cv2
    import torch
    from multiprocessing.connection import Listener
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    model = XVLA.from_pretrained(a.ckpt).to(a.device).eval()
    proc = XVLAProcessor.from_pretrained(a.ckpt)
    tf = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def prep(rgb):
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        return tf(Image.fromarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)))   # RGB order kept (as in export)

    listener = Listener(("localhost", a.port), authkey=a.authkey.encode())
    print(f"X-VLA server ready on localhost:{a.port} ({a.ckpt}, domain {domain_id})", flush=True)
    while True:
        conn = listener.accept()
        try:
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                if req.get("cmd") == "close":
                    conn.close()
                    listener.close()
                    return
                t0 = time.time()
                imgs = torch.stack([prep(im) for im in req["images"]])[None].to(a.device)
                lang = proc.encode_language([req["instruction"]])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    act = model.generate_actions(
                        input_ids=lang["input_ids"].to(a.device), image_input=imgs,
                        image_mask=torch.ones(1, imgs.shape[1], dtype=torch.bool, device=a.device),
                        domain_id=torch.tensor([domain_id], device=a.device),
                        proprio=torch.as_tensor(np.asarray(req["proprio"], np.float32))[None].to(a.device),
                        steps=a.denoise_steps)
                conn.send({"actions": act[0].float().cpu().numpy(), "latency_s": time.time() - t0})
        finally:
            conn.close()


if __name__ == "__main__":
    main()
