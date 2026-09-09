r"""
Low-confidence union mask, following up on scripts/w2_refined_mask.py.

Hypothesis under test: for images where the model's top-1 confidence is
already low, mask BOTH classes' harmful concepts --
    D(k1,c) < -tau_d  OR  D(k2,c) < -tau_d
(k1 = top-1 prediction, k2 = top-2 prediction) -- on the theory that once
whatever is actively suppressing EITHER candidate is removed, the decision
falls to "clean" remaining support and should land correctly more often.
High-confidence images are left untouched (empty mask, i.e. baseline
reconstruction) -- this script only intervenes below --tau-conf.

Pre-registered concern (discussed before running, not added after seeing
results): scripts/w2_refined_mask.py showed that ~86% of a one-sided
harmful-to-k2 mask is nearly inert on its own -- almost all its power to
flip a decision lives in the ~14% of concepts that are ALSO supportive of
k1 (a k1-vs-k2 discriminator, not a "pure" suppressor). Masking
harmful-for-k1 in addition doesn't add a second, independent lever; by the
same logic it should hit concepts that are ALSO k2's support, symmetrically.
So the union may strip the k1-vs-k2 discriminating direction from BOTH
sides rather than leaving unconflicted evidence behind. This script checks
that directly: for every intervened-on image, where does the new
prediction land -- back on k1, over to k2, onto the true label, or onto some
THIRD class entirely (the signature of "discriminating signal destroyed,
decision now arbitrary")?

Reuses the same one-pass-with-per-image-mask-lookup machinery as
w2_refined_mask.py (evaluate_posteriors_indexed): encoding is shared across
all masks, only the decode-time zeroing differs per image.

  $PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
  & $PYEXE scripts\w2_lowconf_union_mask.py --preset erm_resnet_3300 --scores processed\FINAL_ERM_ResNet_3300_T3.json --raw results\W2_label_free_erm.raw.pt --tau-conf 0.55 --out results\W2_lowconf_union_erm.json

Smoke test first (~30s, NOT a valid result):
  & $PYEXE scripts\w2_lowconf_union_mask.py --out results\_smoke_w2union.json --limit-batches 2 --force
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
                    help="cached probs0/yhat; supplies top1_conf, k1, k2 per image")
    p.add_argument("--domain", default="sketch")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--tau-conf", type=float, default=0.55,
                    help="intervene only where top1_conf < this")
    p.add_argument("--out", default="results/W2_lowconf_union_erm.json")
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
        "tau_d": args.tau_d, "tau_conf": args.tau_conf,
        "limit_batches": args.limit_batches,
    }

    with RunRecorder("w2_lowconf_union_mask", config=manifest_cfg, notes=args.notes) as rec:
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

        with rec.step("build_masks"):
            scores = load_scores(args.scores, class_names=class_names)
            D = scores.D  # (n_classes, n_concepts)

            blob = torch.load(args.raw, map_location="cpu")
            probs0, y_cached, yhat_cached = (
                blob["probs0"], blob["y"], blob["yhat"]
            )
            top_vals, top_idx = probs0.topk(2, dim=1)
            top1_class, top2_class = top_idx[:, 0], top_idx[:, 1]
            top1_conf = top_vals[:, 0]
            assert torch.equal(top1_class, yhat_cached)
            n = top1_class.shape[0]

            low_conf = top1_conf < args.tau_conf
            print(f"[scope] {int(low_conf.sum())} / {n} images below "
                  f"tau_conf={args.tau_conf} "
                  f"({100 * low_conf.float().mean():.2f}%)")

            pairs = sorted(set(
                (int(top1_class[i]), int(top2_class[i]))
                for i in range(n) if low_conf[i]
            ))
            pair_index = {p: idx for idx, p in enumerate(pairs)}
            EMPTY_ID = len(pairs)  # sentinel row: no concepts masked
            n_rows = len(pairs) + 1

            mask_table = np.zeros((n_rows, D.shape[1]), dtype=bool)
            for (k1, k2), idx in pair_index.items():
                mask_table[idx] = (D[k1, :] < -args.tau_d) | (D[k2, :] < -args.tau_d)
            # mask_table[EMPTY_ID] stays all-False -- untouched pass-through

            mask_id = torch.full((n,), EMPTY_ID, dtype=torch.long)
            for i in range(n):
                if low_conf[i]:
                    mask_id[i] = pair_index[(int(top1_class[i]), int(top2_class[i]))]

            sizes = mask_table[: len(pairs)].sum(axis=1)
            print(f"[pairs] {len(pairs)} unique low-confidence (top1,top2) pairs, "
                  f"mask size mean={sizes.mean():.1f} median={np.median(sizes):.1f} "
                  f"(nb_concepts={D.shape[1]})")

        with rec.step("forward"):
            probs_new, y = evaluator.evaluate_posteriors_indexed(
                args.domain, torch.from_numpy(mask_table), mask_id
            )

        with rec.step("analyze"):
            n_eval = y.shape[0]
            yhat = yhat_cached[:n_eval]
            assert torch.equal(y, y_cached[:n_eval]), (
                "image order mismatch -- Load_PACS_full is supposed to be "
                "deterministic"
            )
            k1 = top1_class[:n_eval]
            k2 = top2_class[:n_eval]
            lc = low_conf[:n_eval]

            new_pred = probs_new.argmax(dim=1)

            # sanity: untouched (high-confidence) images must be byte-identical
            # to baseline -- confirms the empty-mask row is a true no-op.
            untouched_ok = bool(torch.equal(new_pred[~lc], yhat[~lc]))
            print(f"[sanity] high-confidence images unchanged: {untouched_ok}")
            if not untouched_ok:
                n_bad = int((new_pred[~lc] != yhat[~lc]).sum())
                print(f"[sanity] WARNING: {n_bad} high-confidence images changed "
                      f"anyway -- investigate before trusting results")

            correct0 = y == yhat
            wrong0 = ~correct0

            def landing(mask):
                idx = mask.nonzero(as_tuple=True)[0]
                lands = {"stayed_k1": 0, "moved_to_k2": 0, "moved_to_true_other": 0,
                         "moved_to_other_wrong": 0}
                n_true_in_pair = 0
                for i in idx.tolist():
                    np_i, yi, k1i, k2i = int(new_pred[i]), int(y[i]), int(k1[i]), int(k2[i])
                    if yi in (k1i, k2i):
                        n_true_in_pair += 1
                    if np_i == k1i:
                        lands["stayed_k1"] += 1
                    elif np_i == k2i:
                        lands["moved_to_k2"] += 1
                    elif np_i == yi:
                        lands["moved_to_true_other"] += 1
                    else:
                        lands["moved_to_other_wrong"] += 1
                return {
                    "n": int(mask.sum()),
                    "n_true_label_was_in_top2": n_true_in_pair,
                    **lands,
                    "n_flipped_from_k1": int(mask.sum()) - lands["stayed_k1"],
                    "n_landed_on_true": int((new_pred[idx] == y[idx]).sum()),
                }

            result = {
                "tau_conf": args.tau_conf,
                "tau_d": args.tau_d,
                "n_images": n_eval,
                "n_low_conf": int(lc.sum()),
                "sanity_untouched_unchanged": untouched_ok,
                "correct0_low_conf": landing(lc & correct0),
                "wrong0_low_conf": landing(lc & wrong0),
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
                "probs_new": probs_new, "y": y, "yhat": yhat,
                "top1_class": k1, "top2_class": k2, "low_conf": lc,
                "mask_id": mask_id[:n_eval], "pairs": pairs,
                "class_names": class_names, "tau_d": args.tau_d,
                "tau_conf": args.tau_conf, "domain": args.domain,
            },
            raw_path,
        )
        rec.add_output(raw_path)
        rec.add(n_low_conf=int(lc.sum()), n_images=n_eval)

    print(f"\ndone. wrote {out_path}")


if __name__ == "__main__":
    main()
