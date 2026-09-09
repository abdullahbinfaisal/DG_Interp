"""
Rank-2 flip test (follow-up to scripts/w2_label_free.py).

For every sketch image, mask the harmful concepts (D(k,c) < -tau_d, no
support/H/R gate, same scope as W2) belonging to the model's OWN 2nd-ranked
predicted class -- not the true label, not the top prediction. Re-check
argmax. Does the prediction flip away from the original top-1 class?

Fully label-free at decision time: the mask is chosen from probs0's own
ranking. True labels are only used afterward, to split flip counts by
whether the original top-1 prediction was correct or wrong.

No GPU forward pass: reuses the cached (probs0, P_masked, y, yhat) tensors
already written by w2_label_free.py to <out>.raw.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/W2_label_free_erm.raw.pt")
    ap.add_argument("--out", default="results/W2_rank2_flip_erm.json")
    ap.add_argument("--tau_d", type=float, default=None, help="only for the header; masks already baked into P_masked")
    args = ap.parse_args()

    blob = torch.load(args.raw, map_location="cpu")
    probs0 = blob["probs0"]          # (N, C)
    P_masked = blob["P_masked"]      # (C, N, C) -- P_masked[k, i, :] = posteriors for image i with class-k harmful concepts masked
    y = blob["y"]                    # (N,)
    yhat = blob["yhat"]              # (N,)
    class_names = blob["class_names"]
    tau_d = blob["tau_d"]
    domain = blob["domain"]

    N, C = probs0.shape
    assert torch.equal(probs0.argmax(dim=1), yhat)

    top_vals, top_idx = probs0.topk(3, dim=1)
    top1_class, top2_class, top3_class = top_idx[:, 0], top_idx[:, 1], top_idx[:, 2]
    top1_conf, top2_conf, top3_conf = top_vals[:, 0], top_vals[:, 1], top_vals[:, 2]
    assert torch.equal(top1_class, yhat)

    # gather P_masked[top2_class[i], i, :] for each i
    idx = top2_class.view(1, N, 1).expand(1, N, C)
    masked_probs = torch.gather(P_masked, 0, idx).squeeze(0)  # (N, C)
    new_pred = masked_probs.argmax(dim=1)

    flipped = new_pred != yhat
    correct0 = y == yhat
    wrong0 = ~correct0

    flipped_to_rank2 = flipped & (new_pred == top2_class)
    flipped_elsewhere = flipped & (new_pred != top2_class)

    def block(mask):
        n = int(mask.sum())
        f = int((flipped & mask).sum())
        f_to_rank2 = int((flipped_to_rank2 & mask).sum())
        f_elsewhere = int((flipped_elsewhere & mask).sum())
        return {
            "n": n,
            "n_flipped": f,
            "flip_rate_pct": (100.0 * f / n) if n else None,
            "flipped_to_rank2_class": f_to_rank2,
            "flipped_to_other_class": f_elsewhere,
        }

    result = {
        "domain": domain,
        "tau_d": tau_d,
        "n_images": N,
        "n_correct0": int(correct0.sum()),
        "n_wrong0": int(wrong0.sum()),
        "correct0_block": block(correct0),
        "wrong0_block": block(wrong0),
        "overall_block": block(torch.ones(N, dtype=torch.bool)),
    }

    print(json.dumps(result, indent=2))

    # raw per-image table
    rows = []
    for i in range(N):
        rows.append({
            "i": i,
            "y": class_names[int(y[i])],
            "yhat": class_names[int(yhat[i])],
            "correct0": bool(correct0[i]),
            "top1_class": class_names[int(top1_class[i])],
            "top1_conf": float(top1_conf[i]),
            "top2_class": class_names[int(top2_class[i])],
            "top2_conf": float(top2_conf[i]),
            "top3_class": class_names[int(top3_class[i])],
            "top3_conf": float(top3_conf[i]),
            "new_pred_after_masking_rank2": class_names[int(new_pred[i])],
            "flipped": bool(flipped[i]),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": result, "rows": rows}, f, indent=2)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
