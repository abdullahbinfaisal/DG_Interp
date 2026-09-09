r"""
Group B: masked-accuracy experiments.

  $PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"

  # B1  baseline + reconstruction fidelity          (1 config,  ~1 min)
  & $PYEXE scripts\run_experiments.py --stage B1

  # B2  bucket interventions                        (8 configs, ~4 min)
  & $PYEXE scripts\run_experiments.py --stage B2

  # B3  stratified random controls                  (24 configs, ~12 min)
  & $PYEXE scripts\run_experiments.py --stage B3

  # B4  inert-pair denoising control                (1 config,  ~1 min)
  & $PYEXE scripts\run_experiments.py --stage B4

  # E3E4  D7: does R carry info beyond D's magnitude? (6 configs, ~3 min)
  & $PYEXE scripts\run_experiments.py --stage E3E4

Each configuration writes results/E_<id>.json (schema in docs/EXPERIMENTS.md
section 10) and appends a row to results/summary.csv. Stages are independent and
resumable, which matters because this machine crashes under load: if B3 dies at
seed 2, re-run B3 and only the missing rows are recomputed unless --force.

Smoke-test the whole path first (~1 min, results NOT valid):
  & $PYEXE scripts\run_experiments.py --stage B2 --limit-batches 2 --out-dir results\_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from clean_lib.checkpoints import CheckpointManager
from clean_lib.config import get_preset, PRESETS
from clean_lib.eval import MaskedAccuracyEvaluator
from clean_lib.masks import (
    build_buckets,
    distribution_matched_subset,
    keep_only_random_control,
    load_scores,
    stratified_random_control,
)
from clean_lib.provenance import RunRecorder, set_seed
from clean_lib.sae import SparseAEs, Normalizer

# SAE checkpoint was pickled from a notebook where Normalizer lived in __main__.
setattr(sys.modules["__main__"], "Normalizer", Normalizer)

SUMMARY_COLUMNS = [
    "experiment_id", "stage", "mask_definition", "oracle", "seed",
    "pairs_masked", "masked_D_mass",
    "art_painting_micro", "art_painting_macro",
    "cartoon_micro", "cartoon_macro",
    "photo_micro", "photo_macro",
    "sketch_micro", "sketch_macro",
    "sketch_delta_macro", "sketch_delta_micro",
    "sketch_top_predicted", "sketch_top_predicted_frac",
    "smoke",
]

# Order matters only for readability of the summary table.
B2_BUCKETS = [
    "all_harmful",
    "all_supportive",
    "S-_lo",
    "S-_hi",
    "S+_lo",
    "S+_hi",
    "harmful_invariant",
    "keep_robust_support",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", required=True, choices=["B1", "B2", "B3", "B4", "B5", "E3E4"])
    p.add_argument("--preset", default="erm_resnet_3300", choices=sorted(PRESETS))
    p.add_argument("--scores", default="processed/FINAL_ERM_ResNet_3300_T3.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--tau-h", type=float, default=0.7)
    p.add_argument("--tau-r", type=float, default=0.7)
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--control-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="recompute configurations whose result json already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def result_path(out_dir: Path, experiment_id: str) -> Path:
    safe = experiment_id.replace("/", "_").replace("*", "x")
    return out_dir / f"E_{safe}.json"


def write_result(out_dir: Path, record: dict) -> Path:
    path = result_path(out_dir, record["experiment_id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)
    return path


def append_summary(out_dir: Path, record: dict, baseline: dict | None) -> None:
    """One flat row per configuration. This is what becomes paper tables."""
    acc = record["accuracy"]
    hist = record.get("predicted_label_histogram", {}).get("sketch", {})
    top_lbl, top_frac = "", ""
    if hist:
        tot = sum(hist.values()) or 1
        lbl, cnt = max(hist.items(), key=lambda kv: kv[1])
        top_lbl, top_frac = lbl, f"{cnt / tot:.4f}"

    row = {
        "experiment_id": record["experiment_id"],
        "stage": record.get("stage", ""),
        "mask_definition": record["mask_definition"],
        "oracle": record["oracle"],
        "seed": record.get("seed", ""),
        "pairs_masked": record["pairs_masked"]["total"],
        "masked_D_mass": f"{record['masked_D_mass']['total']:.6g}",
        "sketch_top_predicted": top_lbl,
        "sketch_top_predicted_frac": top_frac,
        "smoke": record.get("smoke", False),
    }
    for dom in ("art_painting", "cartoon", "photo", "sketch"):
        a = acc.get(dom, {})
        row[f"{dom}_micro"] = f"{a.get('micro', float('nan')) * 100:.2f}"
        row[f"{dom}_macro"] = f"{a.get('macro', float('nan')) * 100:.2f}"

    if baseline:
        for key in ("macro", "micro"):
            b = baseline["accuracy"]["sketch"][key]
            v = acc["sketch"][key]
            row[f"sketch_delta_{key}"] = f"{(v - b) * 100:+.2f}"
    else:
        row["sketch_delta_macro"] = ""
        row["sketch_delta_micro"] = ""

    path = out_dir / "summary.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_baseline(out_dir: Path) -> dict | None:
    p = result_path(out_dir, "B1_sae_reconstruction")
    if p.exists():
        return json.loads(p.read_text())
    return None


# ---------------------------------------------------------------------------
# record assembly
# ---------------------------------------------------------------------------


def make_record(
    experiment_id, stage, description, mask_definition, oracle, report, args,
    pairs_total=0, pairs_per_class=None, mass_total=0.0, mass_per_class=None,
    seed=None, notes="",
) -> dict:
    return {
        "experiment_id": experiment_id,
        "stage": stage,
        "description": description,
        "mask_definition": mask_definition,
        "oracle": oracle,
        "oracle_note": ("class-conditional, uses the true label"
                        if oracle else "no label used"),
        "tau_D": args.tau_d,
        "tau_H": args.tau_h,
        "tau_R": args.tau_r,
        "support_floor_stats": args.support_floor,
        "count_filter_in_forward_pass": False,
        "pairs_masked": {"total": int(pairs_total),
                         "per_class": pairs_per_class or {}},
        "masked_D_mass": {"total": float(mass_total),
                          "per_class": mass_per_class or {}},
        "accuracy": report["accuracy"],
        "counts": report["counts"],
        "predicted_label_histogram": report["predicted_label_histogram"],
        "seed": seed,
        "smoke": bool(args.limit_batches),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    cfg.validate()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_cfg = cfg.to_dict()
    manifest_cfg["_resolved"] = {
        "stage": args.stage, "scores": args.scores,
        "tau_d": args.tau_d, "tau_h": args.tau_h, "tau_r": args.tau_r,
        "support_floor": args.support_floor,
        "control_seeds": args.control_seeds,
        "limit_batches": args.limit_batches,
    }

    with RunRecorder(f"run_experiments_{args.stage}", config=manifest_cfg) as rec:
        if args.limit_batches:
            print(f"[warn] --limit-batches {args.limit_batches}: SMOKE RUN, "
                  f"results are NOT valid")
        set_seed(cfg.seed)

        print(f"[cfg] {cfg.name} ckpt={cfg.ckpt} stage={args.stage}")
        print(f"[cfg] eval domains = {cfg.eval_domains}  (sketch is held out)")
        print(f"[cfg] tau_D={args.tau_d:g} tau_H={args.tau_h} tau_R={args.tau_r} "
              f"support_floor={args.support_floor}")

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
                domains=cfg.eval_domains,
                class_names=list(cfg.pacs.class_names),
                nb_concepts=cfg.sae.nb_concepts,
                batch_size=cfg.batch_size,
                pacs_root=cfg.pacs.root,
                limit_batches=args.limit_batches,
            )

        scores = None
        buckets = None
        if args.stage in ("B2", "B3", "B4", "B5", "E3E4"):
            with rec.step("load_scores"):
                scores = load_scores(args.scores,
                                     class_names=list(cfg.pacs.class_names))
                buckets = build_buckets(
                    scores, tau_d=args.tau_d, tau_h=args.tau_h,
                    tau_r=args.tau_r, support_floor=args.support_floor,
                )
                print(f"[scores] {args.scores}")
                for name, b in buckets.items():
                    print("  " + b.summary())

        baseline = load_baseline(out_dir)

        def run_bucket(bucket, experiment_id, stage, description, seed=None):
            path = result_path(out_dir, experiment_id)
            if path.exists() and not args.force:
                print(f"[skip] {experiment_id} exists (use --force to recompute)")
                return
            if bucket.keep_only:
                print(f"\n[run] {experiment_id}: KEEP {bucket.n_pairs} pairs, "
                      f"so {bucket.n_masked} pairs masked, "
                      f"sum|D|masked={bucket.masked_d_mass:.4g}")
            else:
                print(f"\n[run] {experiment_id}: {bucket.n_masked} pairs masked, "
                      f"sum|D|={bucket.masked_d_mass:.4g}")
            report = evaluator.evaluate_masked(bucket.mask)
            evaluator.print_report(report, title=experiment_id)
            record = make_record(
                experiment_id, stage, description, bucket.definition, True,
                report, args,
                # Always the set the mask actually ablates, which for keep-only
                # buckets is the complement of the named bucket.
                pairs_total=bucket.n_masked,
                pairs_per_class=bucket.per_class_masked,
                mass_total=bucket.masked_d_mass,
                mass_per_class=bucket.per_class_masked_d_mass,
                seed=seed,
            )
            record["bucket"] = {
                "name": bucket.name,
                "keep_only": bucket.keep_only,
                "n_pairs_in_bucket": bucket.n_pairs,
                "bucket_d_mass": bucket.d_mass,
                "per_class_pairs_in_bucket": bucket.per_class_pairs,
            }
            rec.add_output(write_result(out_dir, record))
            append_summary(out_dir, record, baseline)

        # ---------------- B1 ----------------
        if args.stage == "B1":
            with rec.step("original"):
                rep = evaluator.evaluate_original()
                evaluator.print_report(rep, title="B1 original model (no SAE)")
                r = make_record("B1_original", "B1",
                                "unmodified backbone, no SAE in the path",
                                "none", False, rep, args)
                rec.add_output(write_result(out_dir, r))
                append_summary(out_dir, r, None)

            with rec.step("sae_reconstruction"):
                rep = evaluator.evaluate_reconstruction()
                evaluator.print_report(rep, title="B1 SAE reconstruction (no mask)")
                r = make_record("B1_sae_reconstruction", "B1",
                                "SAE reconstruction with nothing masked; the "
                                "baseline every delta is measured against",
                                "none", False, rep, args)
                rec.add_output(write_result(out_dir, r))
                append_summary(out_dir, r, None)

        # ---------------- B2 ----------------
        elif args.stage == "B2":
            for name in B2_BUCKETS:
                b = buckets[name]
                with rec.step(f"B2:{name}"):
                    run_bucket(b, f"B2_{name}", "B2",
                               f"mask bucket {name}")

        # ---------------- B3 ----------------
        elif args.stage == "B3":
            for name in B2_BUCKETS:
                if name == "keep_robust_support":
                    # A keep-only mask ablates ~everything; a size-matched random
                    # control is not a meaningful comparison for it.
                    continue
                target = buckets[name]
                if target.n_pairs == 0:
                    print(f"[skip] {name}: empty bucket")
                    continue
                for seed in args.control_seeds:
                    ctrl = stratified_random_control(
                        target, scores, seed=seed,
                        support_floor=args.support_floor)
                    with rec.step(f"B3:{name}:s{seed}"):
                        run_bucket(
                            ctrl, f"B3_ctrl_{name}_seed{seed}", "B3",
                            f"|D|-stratified random control size-matched to {name}",
                            seed=seed,
                        )

        # ---------------- B4 ----------------
        elif args.stage == "B4":
            with rec.step("B4:inert_only"):
                run_bucket(buckets["inert_only"], "B4_inert_only", "B4",
                           "mask only pairs with no measured effect in any source "
                           "domain; isolates dictionary denoising from structure")

            # Does the support floor discard signal or noise? The excluded
            # population carries 46.5% of the grid's |D| mass, 82% of it
            # concentrated in these pairs at a per-pair magnitude comparable to
            # the above-floor non-neutral set. Aggregate mass cannot settle
            # whether that matters -- B4:inert_only is the standing proof that
            # mass and consequence come apart -- so ablate them and look.
            with rec.step("B4:below_floor_gated"):
                run_bucket(buckets["below_floor_gated"], "B4_below_floor_gated",
                           "B4",
                           "mask the gated pairs the support floor excludes; "
                           "tests directly whether a floor of 30 is discarding "
                           "signal or discarding noise")

            # The combined row masks both signs at once, and they cancel, so it
            # is a lower bound. These two are the rows comparable to Table 3.
            for nm in ("below_floor_harmful", "below_floor_supportive"):
                with rec.step(f"B4:{nm}"):
                    run_bucket(buckets[nm], f"B4_{nm}", "B4",
                               f"sign-split of the excluded population: {nm}")

        # ---------------- B5 ----------------
        # Bounds the keep-only oracle. keep_robust_support retains only concepts
        # that support the true class, which deletes every competing-class
        # direction and is close to injecting the label. Two comparisons make it
        # interpretable: the other keep-only variants share that injection
        # component, so differences between them isolate the R axis; and the
        # random keep-only control measures the injection component on its own.
        elif args.stage == "B5":
            for name in ("keep_S+_lo", "keep_all_supportive"):
                b = buckets[name]
                with rec.step(f"B5:{name}"):
                    run_bucket(b, f"B5_{name}", "B5",
                               f"keep-only variant {name}; shares the "
                               f"label-injection component of keep_robust_support")

            tgt = buckets["keep_robust_support"]
            for seed in args.control_seeds:
                ctrl = keep_only_random_control(
                    tgt, scores, seed=seed, support_floor=args.support_floor)
                with rec.step(f"B5:keep_random:s{seed}"):
                    run_bucket(
                        ctrl, f"B5_keep_random_seed{seed}", "B5",
                        "keep-only uniform random per-class size-match to "
                        "robust support; measures how much of that row is the "
                        "label used to build the mask",
                        seed=seed,
                    )

        # ---------------- E3E4 ----------------
        # D7 (docs/DIRECTIONS.md): does R carry information beyond the sign and
        # magnitude of D? For each side, compare masking the named bucket
        # (already covered by B2/B3) against masking a size- and
        # |D|-distribution-matched subset drawn from its opposite-R
        # counterpart. If the matched subset behaves the same as the named
        # bucket, R adds nothing beyond magnitude; if it doesn't, R is doing
        # independent causal work. Direction per class differs by side: harm is
        # enriched at low R, so E3 matches S-_hi's profile out of the larger
        # S-_lo pool; S+_lo's mass is below S+_hi's in every class, so E4 runs
        # the other way, matching S+_lo's profile out of the larger S+_hi pool.
        elif args.stage == "E3E4":
            comparisons = [
                ("harmful", buckets["S-_hi"], buckets["S-_lo"]),
                ("supportive", buckets["S+_lo"], buckets["S+_hi"]),
            ]
            for label, target, pool in comparisons:
                if target.n_pairs == 0 or pool.n_pairs == 0:
                    print(f"[skip] E3E4 {label}: empty target or pool "
                          f"(target={target.n_pairs}, pool={pool.n_pairs})")
                    continue
                for seed in args.control_seeds:
                    matched = distribution_matched_subset(
                        target, pool, scores, seed=seed)
                    with rec.step(f"E3E4:{label}:s{seed}"):
                        run_bucket(
                            matched, f"E3E4_matched_{label}_seed{seed}", "E3E4",
                            f"size- and |D|-distribution-matched subset of "
                            f"{pool.name} to {target.name}'s profile (D7)",
                            seed=seed,
                        )

        rec.add(stage=args.stage, smoke=bool(args.limit_batches))

    print(f"\ndone. results in {out_dir}, summary at {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
