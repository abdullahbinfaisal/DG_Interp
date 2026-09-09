r"""
W2 (docs/DIRECTIONS.md D4): does the oracle masking gain survive without
labels -- and is that survival real signal, or self-confirmation?

Naive test: mask concepts harmful for the model's OWN predicted label ŷ and
see if accuracy on originally-wrong images improves. This is close to
circular: D(k,c) < 0 means "zeroing c raises p(k)" BY CONSTRUCTION, so masking
concepts harmful-for-ŷ mechanically boosts p(ŷ) whether ŷ is right or wrong.

This script isolates real signal from that artifact with an any-class negative
control: for every originally-misclassified sketch image, mask conditioned on
EVERY class (not just ŷ), and compare how much the TRUE class's probability
moves under "predicted" (k=ŷ) vs "other" (k = the 5 classes that are neither
y nor ŷ, averaged). Oracle (k=y) is reported for context only, not part of
the test. If predicted beats other, the model's own guess carries real
information about the true label beyond "any mask boosts its own class".

Scope agreed before running: `all_harmful`-style bucket, thresholded
D(k,c) < -tau_d directly -- no H, R, or support-floor gate (a deliberate
deviation from the rest of the project's convention; see
docs/RESULTS_LEDGER.md once this is logged). ERM only.

  $PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
  & $PYEXE scripts\w2_label_free.py --preset erm_resnet_3300 --scores processed\FINAL_ERM_ResNet_3300_T3.json --out results\W2_label_free_erm.json

Smoke test first (~30s, NOT a valid result):
  & $PYEXE scripts\w2_label_free.py --out results\_smoke_w2.json --limit-batches 2 --force
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
from scipy.stats import ttest_rel, wilcoxon

from clean_lib.checkpoints import CheckpointManager
from clean_lib.config import get_preset, PRESETS
from clean_lib.eval import MaskedAccuracyEvaluator
from clean_lib.masks import load_scores
from clean_lib.provenance import RunRecorder, set_seed
from clean_lib.sae import SparseAEs, Normalizer

# SAE checkpoint was pickled from a notebook where Normalizer lived in __main__.
setattr(sys.modules["__main__"], "Normalizer", Normalizer)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="erm_resnet_3300", choices=sorted(PRESETS))
    p.add_argument("--scores", default="processed/FINAL_ERM_ResNet_3300_T3.json")
    p.add_argument("--domain", default="sketch")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--out", default="results/W2_label_free_erm.json")
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
        "scores": args.scores, "domain": args.domain, "tau_d": args.tau_d,
        "limit_batches": args.limit_batches,
    }

    with RunRecorder("w2_label_free", config=manifest_cfg, notes=args.notes) as rec:
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

        with rec.step("load_scores"):
            scores = load_scores(args.scores, class_names=class_names)
            # Scope agreed for W2: D < -tau_d only, no H/R/support-floor gate.
            masks = [
                torch.from_numpy(scores.D[k, :] < -args.tau_d)
                for k in range(n_classes)
            ]
            for k, m in enumerate(masks):
                print(f"[mask] class {k} ({class_names[k]}): {int(m.sum())} concepts")

        with rec.step("baseline"):
            probs0, y = evaluator.evaluate_posteriors(args.domain, concept_mask=None)
            yhat = probs0.argmax(dim=1)
            n = y.shape[0]
            base_correct = int((yhat == y).sum())
            print(f"[baseline] {args.domain}: {n} images, {base_correct} correct "
                  f"({base_correct / n * 100:.2f}%)")

        # (K, N, K) posteriors: P_masked[k, i, :] = softmax probs for image i
        # when masked as if its label were class k, regardless of that image's
        # actual true or predicted label.
        P_masked = torch.zeros(n_classes, n, n_classes)
        for k in range(n_classes):
            with rec.step(f"masked_class_{k}"):
                probs_k, y_k = evaluator.evaluate_posteriors(
                    args.domain, concept_mask=masks[k])
                assert torch.equal(y_k, y), (
                    "image order changed between passes -- Load_PACS_full is "
                    "supposed to be deterministic"
                )
                P_masked[k] = probs_k

        with rec.step("analysis"):
            wrong = (yhat != y).nonzero(as_tuple=True)[0]
            print(f"[analysis] {len(wrong)} / {n} originally wrong "
                  f"({len(wrong) / n * 100:.2f}%)")

            arm_names = ["oracle", "predicted", "other"]
            delta_true = {a: [] for a in arm_names}
            delta_masked_class = {a: [] for a in arm_names}
            flip = {a: [] for a in arm_names}

            for i in wrong.tolist():
                yi, yhi = int(y[i]), int(yhat[i])
                other_ks = [k for k in range(n_classes) if k not in (yi, yhi)]

                p0_true = float(probs0[i, yi])
                p0_yhat = float(probs0[i, yhi])

                # oracle: k = yi (true label)
                delta_true["oracle"].append(float(P_masked[yi, i, yi]) - p0_true)
                delta_masked_class["oracle"].append(
                    float(P_masked[yi, i, yi]) - p0_true)
                flip["oracle"].append(int(P_masked[yi, i, :].argmax().item() == yi))

                # predicted: k = yhi (model's own guess)
                delta_true["predicted"].append(
                    float(P_masked[yhi, i, yi]) - p0_true)
                delta_masked_class["predicted"].append(
                    float(P_masked[yhi, i, yhi]) - p0_yhat)
                flip["predicted"].append(
                    int(P_masked[yhi, i, :].argmax().item() == yi))

                # other: the 5 classes that are neither y nor yhat, averaged
                dt_other = [float(P_masked[k, i, yi]) - p0_true for k in other_ks]
                dmc_other = [
                    float(P_masked[k, i, k]) - float(probs0[i, k]) for k in other_ks
                ]
                fl_other = [
                    int(P_masked[k, i, :].argmax().item() == yi) for k in other_ks
                ]
                delta_true["other"].append(float(np.mean(dt_other)))
                delta_masked_class["other"].append(float(np.mean(dmc_other)))
                flip["other"].append(float(np.mean(fl_other)))

            summary = {}
            for a in arm_names:
                dt = np.array(delta_true[a])
                dmc = np.array(delta_masked_class[a])
                fl = np.array(flip[a])
                summary[a] = {
                    "delta_p_true_mean": float(dt.mean()),
                    "delta_p_true_median": float(np.median(dt)),
                    "delta_p_masked_class_mean": float(dmc.mean()),
                    "flip_rate": float(fl.mean()),
                    "n": int(len(dt)),
                }
                s = summary[a]
                print(f"[{a:>9}] mean dp(true)={s['delta_p_true_mean']:+.4f} "
                      f"median={s['delta_p_true_median']:+.4f} "
                      f"mean dp(masked class)={s['delta_p_masked_class_mean']:+.4f} "
                      f"flip-to-correct={s['flip_rate'] * 100:.2f}%")

            # The decisive comparison: predicted vs other, paired per image.
            dt_pred = np.array(delta_true["predicted"])
            dt_other = np.array(delta_true["other"])
            diff = dt_pred - dt_other
            if np.any(diff != 0):
                w_stat, w_p = wilcoxon(diff)
            else:
                w_stat, w_p = float("nan"), 1.0
            t_stat, t_p = ttest_rel(dt_pred, dt_other)

            test = {
                "mean_diff_predicted_minus_other": float(diff.mean()),
                "median_diff_predicted_minus_other": float(np.median(diff)),
                "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
                "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
            }
            print(f"\n[test] predicted vs other, paired dp(true): "
                  f"mean diff={test['mean_diff_predicted_minus_other']:+.4f}  "
                  f"wilcoxon p={w_p:.4g}  paired-t p={t_p:.4g}")

        with rec.step("rank_analysis"):
            # Does the true class systematically get the largest spillover
            # delta among the classes NOT being masked-for, even where it
            # can't beat the masked class's own mechanical self-boost?
            # Chance rate for "true class ranks #1 of the 6 non-masked
            # classes" is 1/6 = 16.7% if there's no real signal.
            rank_stats = {}
            for scheme, k_of in (("oracle", lambda i: int(y[i])),
                                  ("predicted", lambda i: int(yhat[i]))):
                ranks_full = []
                ranks_excl_masked = []
                delta_true_vals = []
                delta_masked_vals = []
                delta_by_rank_excl = [[] for _ in range(n_classes - 1)]

                for i in wrong.tolist():
                    k = k_of(i)
                    yi = int(y[i])
                    delta = (P_masked[k, i, :] - probs0[i, :]).numpy()

                    order_full = np.argsort(-delta)
                    rank_full = int(np.where(order_full == yi)[0][0]) + 1
                    ranks_full.append(rank_full)
                    delta_true_vals.append(float(delta[yi]))
                    delta_masked_vals.append(float(delta[k]))

                    if yi != k:
                        classes_excl = np.array(
                            [c for c in range(n_classes) if c != k])
                        delta_excl = delta[classes_excl]
                        order_excl = np.argsort(-delta_excl)
                        rank_excl = int(
                            np.where(classes_excl[order_excl] == yi)[0][0]) + 1
                        ranks_excl_masked.append(rank_excl)
                        for pos in range(n_classes - 1):
                            delta_by_rank_excl[pos].append(
                                float(delta_excl[order_excl[pos]]))

                ranks_full_arr = np.array(ranks_full)
                ranks_excl_arr = np.array(ranks_excl_masked)
                rank_stats[scheme] = {
                    "rank_full_hist": {
                        str(r): int((ranks_full_arr == r).sum())
                        for r in range(1, n_classes + 1)
                    },
                    "rank_excl_masked_hist": (
                        {
                            str(r): int((ranks_excl_arr == r).sum())
                            for r in range(1, n_classes)
                        } if ranks_excl_arr.size else None
                    ),
                    "true_is_top_of_nonmasked_pct": (
                        float((ranks_excl_arr == 1).mean() * 100)
                        if ranks_excl_arr.size else None
                    ),
                    "chance_pct": 100.0 / (n_classes - 1),
                    "mean_delta_by_rank_excl_masked": [
                        float(np.mean(v)) if v else None
                        for v in delta_by_rank_excl
                    ],
                    "delta_true_mean": float(np.mean(delta_true_vals)),
                    "delta_masked_mean": float(np.mean(delta_masked_vals)),
                }
                s = rank_stats[scheme]
                print(f"\n[rank:{scheme}] rank of true class among all 7 "
                      f"(1=highest delta): {s['rank_full_hist']}")
                if s["rank_excl_masked_hist"] is not None:
                    print(f"[rank:{scheme}] rank of true class among the 6 "
                          f"non-masked classes: {s['rank_excl_masked_hist']}")
                    print(f"[rank:{scheme}] true class is #1 of 6 non-masked "
                          f"{s['true_is_top_of_nonmasked_pct']:.2f}% of the "
                          f"time (chance = {s['chance_pct']:.2f}%)")
                    print(f"[rank:{scheme}] mean delta by non-masked rank "
                          f"position (1=highest): " +
                          ", ".join(f"{v:+.4f}" for v in
                                    s['mean_delta_by_rank_excl_masked']))

        record = {
            "preset": args.preset,
            "scores_path": args.scores,
            "domain": args.domain,
            "tau_d": args.tau_d,
            "support_floor": None,
            "n_images": n,
            "n_wrong": int(len(wrong)),
            "mask_sizes": {
                class_names[k]: int(masks[k].sum()) for k in range(n_classes)
            },
            "arms": summary,
            "test": test,
            "rank_analysis": rank_stats,
            "smoke": bool(args.limit_batches),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(out_path)
        rec.add_output(out_path)

        # Raw per-image, per-masking-class posteriors, so future re-slicing
        # of this question doesn't need another GPU pass. Small: (K, N, K)
        # float32 plus (N, K), (N,), (N,) -- well under 1 MB here.
        raw_path = out_path.with_suffix(".raw.pt")
        torch.save(
            {
                "probs0": probs0, "P_masked": P_masked, "y": y, "yhat": yhat,
                "class_names": class_names, "tau_d": args.tau_d,
                "domain": args.domain,
            },
            raw_path,
        )
        rec.add_output(raw_path)

        rec.add(n_wrong=int(len(wrong)), smoke=bool(args.limit_batches))

    print(f"\ndone. wrote {out_path}")


if __name__ == "__main__":
    main()
