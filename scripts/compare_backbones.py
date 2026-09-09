r"""
Cross-backbone comparison for W1 (docs/ABLATIONS_AND_EXPERIMENTS.md T3.1):
does invariance training (MMD, DANN) actually reduce harmful-invariant
concepts relative to ERM, or does it retain comparably many?

Reads score files and results/summary.csv rows that other scripts already
produced -- no model loading, no GPU forward pass, safe to re-run freely.

  $PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
  & $PYEXE scripts\compare_backbones.py --backbone "erm=processed\FINAL_ERM_ResNet_3300_T3.json:results" --backbone "mmd=processed\FINAL_MMD_ResNet_1800_T3.json:results\mmd" --backbone "dann=processed\FINAL_DANN_ResNet_5000_T3.json:results\dann" --out results\W1_backbone_comparison.json

Each --backbone is NAME=SCORES_PATH:RESULTS_DIR (repeatable, at least 2
required). The FIRST one named is the reference backbone (normally ERM) that
every other backbone is tested against.

Pre-committed interpretation of the Fisher exact test on harmful-invariant
retention (docs/ABLATIONS_AND_EXPERIMENTS.md, the plan): non-significant, or
significant with the other backbone's rate at or above the reference's, =>
"comparably many" (W1 closes in the paper's favor). Significant AND lower =>
"materially fewer" (the claim needs softening). Both outcomes are reportable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import fisher_exact

from clean_lib.masks import build_buckets, load_scores, Scores
from clean_lib.provenance import RunRecorder

BUCKET_NAMES = [
    "harmful_invariant", "S+_hi", "S+_lo", "S-_hi", "S-_lo",
    "all_harmful", "all_supportive",
]

INTERVENTION_ROWS = ["all_harmful", "S-_hi", "harmful_invariant", "keep_robust_support"]


def parse_backbone_arg(raw: str) -> tuple[str, str, str]:
    name, rest = raw.split("=", 1)
    scores_path, results_dir = rest.split(":", 1)
    return name, scores_path, results_dir


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backbone", action="append", required=True,
        help="NAME=SCORES_PATH:RESULTS_DIR, repeatable. First one is the reference.",
    )
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--tau-h", type=float, default=0.7)
    p.add_argument("--tau-r", type=float, default=0.7)
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--out", default="results/W1_backbone_comparison.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# per-backbone statistics
# ---------------------------------------------------------------------------


def high_h_high_r_counts(
    scores: Scores, sign: str, tau_d: float, tau_h: float, tau_r: float,
    support_floor: int,
) -> tuple[int, int]:
    """(denominator, numerator) for "of high-H {sign} pairs, how many are also
    high-R". Mirrors scripts/inspect_scores.py's inline computation (lines
    106-112 there) -- kept standalone here since that script has no
    return-value API and this is a 3-line numpy reduction, not worth wiring
    a shared import for.
    """
    sup = scores.support(support_floor)
    d_test = (scores.D > tau_d) if sign == "supportive" else (scores.D < -tau_d)
    denom = sup & d_test & (scores.H >= tau_h)
    numer = denom & (scores.R >= tau_r)
    return int(denom.sum()), int(numer.sum())


def bucket_sizes(buckets: dict) -> dict:
    return {name: buckets[name].n_pairs for name in BUCKET_NAMES if name in buckets}


def load_summary_rows(results_dir: str) -> dict:
    """experiment_id -> row dict, from results/<dir>/summary.csv."""
    path = Path(results_dir) / "summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["experiment_id"]: row for row in csv.DictReader(fh)}


def intervention_table(results_dir: str) -> dict:
    rows = load_summary_rows(results_dir)
    out = {}
    for bucket in INTERVENTION_ROWS:
        r = rows.get(f"B2_{bucket}")
        out[bucket] = None if r is None else {
            "sketch_delta_macro": r.get("sketch_delta_macro"),
            "sketch_delta_micro": r.get("sketch_delta_micro"),
            "pairs_masked": r.get("pairs_masked"),
        }
    return out


def pct_str(numer: int, denom: int) -> str:
    if not denom:
        return "n/a (empty population)"
    return f"{numer}/{denom} ({numer / denom * 100:.1f}%)"


def fisher_row(ref_denom: int, ref_numer: int, other_denom: int, other_numer: int) -> dict:
    table = [
        [ref_numer, ref_denom - ref_numer],
        [other_numer, other_denom - other_numer],
    ]
    odds_ratio, p_value = fisher_exact(table)
    return {
        "table": table,
        "ref_rate": (ref_numer / ref_denom) if ref_denom else None,
        "other_rate": (other_numer / other_denom) if other_denom else None,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def verdict(fisher: dict) -> str:
    ref_rate, other_rate = fisher["ref_rate"], fisher["other_rate"]
    if ref_rate is None or other_rate is None:
        return "undetermined (empty population)"
    if fisher["p_value"] >= 0.05 or other_rate >= ref_rate:
        return "comparably many"
    return "materially fewer"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    backbones = [parse_backbone_arg(b) for b in args.backbone]
    if len(backbones) < 2:
        raise SystemExit("need at least 2 --backbone entries to compare")
    ref_name = backbones[0][0]

    manifest_cfg = {
        "backbones": [{"name": n, "scores": s, "results_dir": r} for n, s, r in backbones],
        "tau_d": args.tau_d, "tau_h": args.tau_h, "tau_r": args.tau_r,
        "support_floor": args.support_floor, "reference": ref_name,
    }

    with RunRecorder("compare_backbones", config=manifest_cfg) as rec:
        per_backbone = {}
        with rec.step("load_and_bucket"):
            for name, scores_path, results_dir in backbones:
                scores = load_scores(scores_path)
                buckets = build_buckets(
                    scores, tau_d=args.tau_d, tau_h=args.tau_h,
                    tau_r=args.tau_r, support_floor=args.support_floor,
                )
                sup_denom, sup_numer = high_h_high_r_counts(
                    scores, "supportive", args.tau_d, args.tau_h, args.tau_r,
                    args.support_floor)
                harm_denom, harm_numer = high_h_high_r_counts(
                    scores, "harmful", args.tau_d, args.tau_h, args.tau_r,
                    args.support_floor)

                per_backbone[name] = {
                    "scores_path": scores_path,
                    "results_dir": results_dir,
                    "bucket_sizes": bucket_sizes(buckets),
                    "asymmetry": {
                        "supportive": {"denom": sup_denom, "numer": sup_numer},
                        "harmful": {"denom": harm_denom, "numer": harm_numer},
                    },
                    "interventions": intervention_table(results_dir),
                }
                print(f"[{name}] bucket sizes: {per_backbone[name]['bucket_sizes']}")
                print(f"[{name}] of high-H supportive pairs, also high-R: "
                      f"{pct_str(sup_numer, sup_denom)}")
                print(f"[{name}] of high-H harmful pairs, also high-R:    "
                      f"{pct_str(harm_numer, harm_denom)}")

        fisher = {}
        with rec.step("fisher_tests"):
            ref = per_backbone[ref_name]["asymmetry"]
            for name, _, _ in backbones[1:]:
                other = per_backbone[name]["asymmetry"]
                harm = fisher_row(
                    ref["harmful"]["denom"], ref["harmful"]["numer"],
                    other["harmful"]["denom"], other["harmful"]["numer"])
                supp = fisher_row(
                    ref["supportive"]["denom"], ref["supportive"]["numer"],
                    other["supportive"]["denom"], other["supportive"]["numer"])
                harm["verdict"] = verdict(harm)
                supp["verdict"] = verdict(supp)
                fisher[name] = {"harmful": harm, "supportive": supp}

                print(f"\n[fisher] {ref_name} vs {name} -- harmful-invariant retention:")
                print(f"  {ref_name} rate={pct_str(ref['harmful']['numer'], ref['harmful']['denom'])}"
                      f"  {name} rate={pct_str(other['harmful']['numer'], other['harmful']['denom'])}")
                print(f"  odds_ratio={harm['odds_ratio']:.3g}  p={harm['p_value']:.4g}"
                      f"  => {harm['verdict']}")
                print(f"[fisher] {ref_name} vs {name} -- supportive-invariant retention "
                      f"(asymmetry replication check):")
                print(f"  odds_ratio={supp['odds_ratio']:.3g}  p={supp['p_value']:.4g}"
                      f"  => {supp['verdict']}")

        record = {
            "tau_d": args.tau_d, "tau_h": args.tau_h, "tau_r": args.tau_r,
            "support_floor": args.support_floor,
            "reference_backbone": ref_name,
            "per_backbone": per_backbone,
            "fisher_tests": fisher,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(out_path)

        rec.add_output(out_path)
        rec.add(reference=ref_name, backbones=[n for n, _, _ in backbones])

    print(f"\ndone. wrote {out_path}")


if __name__ == "__main__":
    main()
