r"""
Purity-gated follow-up to the rank-2 flip test (scripts/w2_rank2_flip.py).

The naive rank-2 test masks D(k2,c) < -tau_d, where k2 is the model's OWN
runner-up class -- fully label-free at decision time. That flips 43.4% of
ALREADY-CORRECT predictions (results/W2_rank2_flip_erm.json), far more than
"ideally it should" if the image contains clean, non-conflicting support for
its true (top-1) class. The suspected reason: a concept can be harmful-to-k2
and simultaneously supportive-of-k1 at once (a k1-specific detector that,
when active, suppresses k2 as a side effect of supporting k1) -- ablating it
to help k2 costs k1 real support it was actually using.

Refined mask, per image (k1 = current top-1 prediction, k2 = runner-up):

    D(k2,c) < -tau_d   AND   D(k1,c) <= tau_d

i.e. drop any concept from the harmful-to-k2 set that is ALSO supportive of
k1 (same tau_d convention the rest of the project uses for "supportive").
Only concepts that are purely anti-k2 -- not doing double duty for k1 -- get
masked.

Cost note: there are up to n_classes*(n_classes-1) distinct (k1,k2) pairs,
but this does NOT require one forward pass per pair. Encoding (backbone +
SAE encode) is identical for every mask; only the decode-time zeroing
differs. So this runs the encoder once over the domain and looks up each
image's own mask at decode time via evaluate_posteriors_indexed() --
the same cost as a single reconstruction pass, not 39 passes.

Also re-runs the naive (unfiltered) rank-2 mask through the SAME indexed
pathway as a consistency check: it must reproduce w2_rank2_flip.py's
correct0/wrong0 flip counts exactly, or something in this path is wrong.

  $PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
  & $PYEXE scripts\w2_refined_mask.py --preset erm_resnet_3300 --scores processed\FINAL_ERM_ResNet_3300_T3.json --raw results\W2_label_free_erm.raw.pt --out results\W2_refined_mask_erm.json

Smoke test first (~30s, NOT a valid result):
  & $PYEXE scripts\w2_refined_mask.py --out results\_smoke_w2refined.json --limit-batches 2 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from clean_lib.checkpoints import CheckpointManager
from clean_lib.config import get_preset, PRESETS
from clean_lib.eval import MaskedAccuracyEvaluator
from clean_lib.masks import load_scores
from clean_lib.provenance import RunRecorder, set_seed
from clean_lib.sae import SparseAEs, Normalizer

setattr(sys.modules["__main__"], "Normalizer", Normalizer)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="erm_resnet_3300", choices=sorted(PRESETS))
    p.add_argument("--scores", default="processed/FINAL_ERM_ResNet_3300_T3.json")
    p.add_argument("--raw", default="results/W2_label_free_erm.raw.pt",
                    help="cached probs0/yhat from w2_label_free.py; supplies "
                         "each image's own (k1,k2) pair")
    p.add_argument("--domain", default="sketch")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--out", default="results/W2_refined_mask_erm.json")
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    cfg.validate()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists. Pass --force to overwrite.")

    manifest_cfg = cfg.to_dict()
    manifest_cfg["_resolved"] = {
        "scores": args.scores, "raw": args.raw, "domain": args.domain,
        "tau_d": args.tau_d, "limit_batches": args.limit_batches,
    }

    with RunRecorder("w2_refined_mask", config=manifest_cfg, notes=args.notes) as rec:
        if args.limit_batches:
            print(f"[warn] --limit-batches {args.limit_batches}: SMOKE RUN, "
                  f"results are NOT valid")
        set_seed(cfg.seed)

        class_names = list(cfg.pacs.class_names)
        n_classes = len(class_names)

        with rec.step("load_model"):
            manager = CheckpointManager(directory=cfg.backbone_dir)
            manager.load_checkpoints([cfg.ckpt])
            sae_manager = SparseAEs(
                feature_dim=cfg.sae.feature_dim, topk=cfg.sae.topk,
                nb_concepts=cfg.sae.nb_concepts,
                rearrange_string=cfg.sae.rearrange_string,
                checkpointManager=manager, train_envs=list(cfg.train_envs),
                w=cfg.sae.w,
            )
            sae_manager.load_checkpoint(cfg.sae.checkpoint_path)

            evaluator = MaskedAccuracyEvaluator(
                backbone=sae_manager.get_backbone(cfg.ckpt),
                sae=sae_manager.get_sae(cfg.ckpt),
                domains=[args.domain],
                class_names=class_names,
                nb_concepts=cfg.sae.nb_concepts,
                batch_size=cfg.batch_size,
                pacs_root=cfg.pacs.root,
                limit_batches=args.limit_batches,
            )

        with rec.step("load_scores_and_pairs"):
            scores = load_scores(args.scores, class_names=class_names)
            D = scores.D  # (n_classes, n_concepts)

            blob = torch.load(args.raw, map_location="cpu")
            probs0_cached, y_cached, yhat_cached = (
                blob["probs0"], blob["y"], blob["yhat"]
            )
            top_vals, top_idx = probs0_cached.topk(3, dim=1)
            top1_class, top2_class = top_idx[:, 0], top_idx[:, 1]
            assert torch.equal(top1_class, yhat_cached), (
                "cached yhat doesn't match probs0's own argmax -- raw file "
                "mismatch"
            )
            n = top1_class.shape[0]

            pairs = sorted(set(
                (int(top1_class[i]), int(top2_class[i])) for i in range(n)
            ))
            pair_index = {p: idx for idx, p in enumerate(pairs)}
            n_pairs = len(pairs)

            mask_naive = np.zeros((n_pairs, D.shape[1]), dtype=bool)
            mask_refined = np.zeros((n_pairs, D.shape[1]), dtype=bool)
            retained_frac = []
            for (k1, k2), idx in pair_index.items():
                naive = D[k2, :] < -args.tau_d
                refined = naive & (D[k1, :] <= args.tau_d)
                mask_naive[idx] = naive
                mask_refined[idx] = refined
                if naive.sum() > 0:
                    retained_frac.append(float(refined.sum()) / float(naive.sum()))

            mask_id = torch.tensor(
                [pair_index[(int(top1_class[i]), int(top2_class[i]))]
                 for i in range(n)],
                dtype=torch.long,
            )

            print(f"[pairs] {n_pairs} unique (top1,top2) pairs")
            print(f"[purity] mean fraction of naive harmful-to-k2 concepts "
                  f"retained after purity gate: {np.mean(retained_frac):.3f} "
                  f"(median {np.median(retained_frac):.3f})")

        with rec.step("forward_refined"):
            # mask_id is full-domain-length; evaluate_posteriors_indexed only
            # consumes a prefix of it under --limit-batches, so this is safe
            # for both smoke and real runs.
            probs_refined, y = evaluator.evaluate_posteriors_indexed(
                args.domain, torch.from_numpy(mask_refined), mask_id
            )

        with rec.step("forward_naive_sanity"):
            probs_naive, y2 = evaluator.evaluate_posteriors_indexed(
                args.domain, torch.from_numpy(mask_naive), mask_id
            )
            assert torch.equal(y, y2)

        with rec.step("analyze"):
            n_eval = y.shape[0]
            yhat = yhat_cached[:n_eval]
            y_true_cached = y_cached[:n_eval]
            assert torch.equal(y, y_true_cached), (
                "image order mismatch between this run and the cached raw.pt "
                "-- Load_PACS_full is supposed to be deterministic"
            )
            correct0 = y == yhat
            wrong0 = ~correct0

            def flips(probs):
                new_pred = probs.argmax(dim=1)
                flipped = new_pred != yhat
                return new_pred, flipped

            new_pred_naive, flipped_naive = flips(probs_naive)
            new_pred_refined, flipped_refined = flips(probs_refined)

            def block(flipped, new_pred, group_mask):
                m = int(group_mask.sum())
                f = int((flipped & group_mask).sum())
                landed_true = int((flipped & group_mask & (new_pred == y)).sum())
                return {
                    "n": m,
                    "n_flipped": f,
                    "flip_rate_pct": (100.0 * f / m) if m else None,
                    "n_flipped_landed_on_true": landed_true,
                }

            result = {
                "n_images": n_eval,
                "n_pairs": n_pairs,
                "mean_retained_fraction": float(np.mean(retained_frac)),
                "median_retained_fraction": float(np.median(retained_frac)),
                "naive": {
                    "correct0": block(flipped_naive, new_pred_naive, correct0),
                    "wrong0": block(flipped_naive, new_pred_naive, wrong0),
                },
                "refined": {
                    "correct0": block(flipped_refined, new_pred_refined, correct0),
                    "wrong0": block(flipped_refined, new_pred_refined, wrong0),
                },
            }
            print(json.dumps(result, indent=2))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(out_path)
        rec.add_output(out_path)

        raw_path = out_path.with_suffix(".raw.pt")
        torch.save(
            {
                "probs_naive": probs_naive, "probs_refined": probs_refined,
                "y": y, "yhat": yhat, "top1_class": top1_class[:n_eval],
                "top2_class": top2_class[:n_eval], "mask_id": mask_id[:n_eval],
                "pairs": pairs, "class_names": class_names,
                "tau_d": args.tau_d, "domain": args.domain,
            },
            raw_path,
        )
        rec.add_output(raw_path)
        rec.add(n_pairs=n_pairs, n_images=n_eval)

    print(f"\ndone. wrote {out_path}")


if __name__ == "__main__":
    main()
