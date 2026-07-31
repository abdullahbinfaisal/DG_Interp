r"""
Group A: every analysis that needs no GPU and no masking.

  & $PYEXE scripts\analyze_scores.py --scores processed\FINAL_ERM_ResNet_3300_T3.json

Produces, under results/analysis/:
  A1  typology_counts.csv        H x R contingency per D regime, per tau
      asymmetry.csv              the three-threshold support/harm asymmetry
      fig_h_vs_d.png             Fig 1: invariance vs discriminative effect
      fig_hr_heatmap.png         Fig 5: H x R by D regime
  A2  domain_attribution.{json,csv}   argmax_d |D_d| for low-R pairs
  A3  conflict.{json,csv}             class-conflict from D alone
  A4  per_class_counts.csv            per-class support / supportive / harmful
  A5  bucket_masses.csv               pair counts and sum|D| per bucket

Optionally joins per-class sketch accuracy into A4:
  --baseline results\E0_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clean_lib.config import get_preset
from clean_lib.masks import (
    Scores,
    build_buckets,
    conflict_analysis,
    domain_attribution,
    load_scores,
)
from clean_lib.provenance import RunRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="results/analysis")
    p.add_argument("--preset", default="erm_resnet_3300")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--taus", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--baseline", default=None,
                   help="optional E0 baseline json, to join sketch accuracy into A4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


def a1_typology(s: Scores, tau_d: float, taus, floor: int, out: Path):
    sup = s.support(floor)
    regimes = {
        "neutral": np.abs(s.D) <= tau_d,
        "harmful": s.D < -tau_d,
        "supportive": s.D > tau_d,
    }

    rows, asym = [], []
    for tau in taus:
        hiH, hiR = s.H >= tau, s.R >= tau
        for rname, rsel in regimes.items():
            sel = sup & rsel
            n = int(sel.sum())
            for hlab, hm in (("H<tau", ~hiH), ("H>=tau", hiH)):
                for rlab, rm in (("R<tau", ~hiR), ("R>=tau", hiR)):
                    c = int((sel & hm & rm).sum())
                    rows.append({
                        "tau_h": tau, "tau_r": tau, "tau_d": tau_d,
                        "d_regime": rname, "H_bin": hlab, "R_bin": rlab,
                        "count": c, "regime_total": n,
                        "pct_of_regime": (c / n * 100) if n else 0.0,
                    })

        # The section 5.3 asymmetry: among invariant pairs, what fraction are
        # also consistent, split by whether they help or hurt.
        rec = {"tau": tau, "tau_d": tau_d}
        for lab, rsel in (("supportive", regimes["supportive"]),
                          ("harmful", regimes["harmful"])):
            grp = sup & rsel & hiH
            tot = int(grp.sum())
            hi = int((grp & hiR).sum())
            rec[f"high_H_{lab}_n"] = tot
            rec[f"high_H_{lab}_also_high_R"] = hi
            rec[f"high_H_{lab}_pct"] = (hi / tot * 100) if tot else 0.0
        sp, hp = rec["high_H_supportive_pct"], rec["high_H_harmful_pct"]
        rec["asymmetry_ratio"] = (sp / hp) if hp else float("nan")
        asym.append(rec)

    pd.DataFrame(rows).to_csv(out / "typology_counts.csv", index=False)
    df_a = pd.DataFrame(asym)
    df_a.to_csv(out / "asymmetry.csv", index=False)

    print("\n--- A1  support/harm asymmetry (the section 5.3 claim) ---")
    print(df_a[["tau", "high_H_supportive_pct", "high_H_harmful_pct",
                "asymmetry_ratio"]].to_string(index=False,
                                              float_format=lambda v: f"{v:.2f}"))

    # Consistency presupposes broad activation.
    tau0 = taus[0]
    nonneutral = sup & (np.abs(s.D) > tau_d)
    lowH_hiR = int((nonneutral & (s.H < tau0) & (s.R >= tau0)).sum())
    print(f"\n  low-H & high-R non-neutral pairs at tau={tau0}: "
          f"{lowH_hiR} / {int(nonneutral.sum())}  "
          f"(consistency presupposes broad activation)")

    hiH_all = int((sup & (s.H >= tau0)).sum())
    hiH_hiR = int((sup & (s.H >= tau0) & (s.R >= tau0)).sum())
    print(f"  of {hiH_all} high-H pairs, {hiH_hiR} ({hiH_hiR/hiH_all*100:.1f}%) "
          f"also clear tau_R  (H tells you little about R)")

    neutral_hiHR = int((sup & regimes["neutral"] & (s.H >= tau0) & (s.R >= tau0)).sum())
    n_neutral = int((sup & regimes["neutral"]).sum())
    print(f"  {neutral_hiHR}/{n_neutral} ({neutral_hiHR/n_neutral*100:.1f}%) of "
          f"NEUTRAL pairs are high-H & high-R  (R needs the magnitude gate)")


def a1_figures(s: Scores, tau_d: float, tau: float, floor: int, out: Path):
    sup = s.support(floor)

    # Fig 1: H vs D
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k, name in enumerate(s.class_names):
        m = sup[k]
        ax.scatter(s.D[k][m], s.H[k][m], s=9, alpha=0.55, label=name)
    ax.axvline(0, color="0.4", lw=0.8)
    for v in (-tau_d, tau_d):
        ax.axvline(v, color="0.6", lw=0.7, ls=":")
    ax.axhline(tau, color="0.6", lw=0.7, ls="--")
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("discriminative effect  D(k,c)")
    ax.set_ylabel("activation invariance  H(k,c)")
    ax.set_title("Activation invariance does not imply discriminative usefulness")
    ax.legend(fontsize=7, markerscale=1.6, ncol=2)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(out / "fig_h_vs_d.png", dpi=200)
    plt.close(fig)

    # Fig 5: H x R contingency per D regime
    regimes = [
        (f"neutral\n|D| <= {tau_d:g}", np.abs(s.D) <= tau_d),
        (f"harmful\nD < -{tau_d:g}", s.D < -tau_d),
        (f"supportive\nD > {tau_d:g}", s.D > tau_d),
    ]
    hiH, hiR = s.H >= tau, s.R >= tau
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    im = None
    for ax, (title, rsel) in zip(axes, regimes):
        sel = sup & rsel
        n = int(sel.sum())
        cells = np.zeros((2, 2))
        for i, hm in enumerate((~hiH, hiH)):
            for j, rm in enumerate((~hiR, hiR)):
                cells[i, j] = int((sel & hm & rm).sum())
        pct = cells / n * 100 if n else cells
        im = ax.imshow(pct, vmin=0, vmax=100, cmap="viridis")
        ax.set_xticks([0, 1], [f"R < {tau}", f"R $\\geq$ {tau}"])
        ax.set_yticks([0, 1], [f"H < {tau}", f"H $\\geq$ {tau}"])
        ax.set_title(f"{title}   (n={n})", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cells[i,j])}\n{pct[i,j]:.1f}%",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if pct[i, j] > 50 else "black")
    fig.colorbar(im, ax=axes, shrink=0.85, label="share of regime (%)")
    fig.savefig(out / "fig_hr_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"\n  figures -> {out/'fig_h_vs_d.png'} , {out/'fig_hr_heatmap.png'}")


# ---------------------------------------------------------------------------
# A2 / A3 / A4 / A5
# ---------------------------------------------------------------------------


def a2_attribution(s, domains, tau_d, tau, floor, out):
    res = domain_attribution(s, domains, tau_d=tau_d, tau_r=tau, support_floor=floor)
    (out / "domain_attribution.json").write_text(json.dumps(res, indent=2))

    rows = []
    for sign in ("supportive", "harmful"):
        for dom, c in res["overall"][sign].items():
            rows.append({"scope": "all", "sign": sign, "domain": dom, "count": c})
    for cls, d in res["per_class"].items():
        for sign in ("supportive", "harmful"):
            for dom, c in d[sign].items():
                rows.append({"scope": cls, "sign": sign, "domain": dom, "count": c})
    pd.DataFrame(rows).to_csv(out / "domain_attribution.csv", index=False)

    print("\n--- A2  where concentrated effect lives (low-R pairs, argmax_d |D_d|) ---")
    print(f"  low-R supportive n={res['n_low_r_supportive']}, "
          f"harmful n={res['n_low_r_harmful']}")
    print("\n  LOW-R (concentrated) counts:")
    print(pd.DataFrame(res["overall"]).T[domains].to_string())
    print("\n  LOW-R share of each row (%):")
    print(pd.DataFrame(res["overall_pct"]).T[domains].to_string(
        float_format=lambda v: f"{v:.1f}"))

    print("\n  REFERENCE: HIGH-R pairs, i.e. NOT concentrated. This is the null.")
    print("  If low-R skews toward a domain more than high-R does, the")
    print("  concentration claim survives; if the two match, the argmax table is")
    print("  reflecting a global per-domain effect size, not per-concept structure.")
    print(pd.DataFrame(res["reference_high_r_pct"]).T[domains].to_string(
        float_format=lambda v: f"{v:.1f}"))

    print("\n  mean |D_d| over non-neutral pairs, by domain:")
    for d in domains:
        print(f"    {d:<14} {res['mean_abs_D_by_domain'][d]:.3e}")
    print("  active-image exposure over non-neutral pairs, by domain:")
    for d in domains:
        print(f"    {d:<14} {res['active_image_exposure_by_domain'][d]:,.0f}")
    return res


def a3_conflict(s, tau_d, floor, out):
    res = conflict_analysis(s, tau_d=tau_d, support_floor=floor)
    (out / "conflict.json").write_text(json.dumps(res, indent=2))

    pd.DataFrame(res["per_class"]).T.to_csv(out / "conflict.csv")
    print("\n--- A3  class conflict, computed from D alone ---")
    print(f"  concepts considered      : {res['n_concepts_considered']}")
    print(f"  supportive for some class: {res['n_supportive_any']}")
    print(f"  harmful for some class   : {res['n_harmful_any']}")
    print(f"  CONFLICTING              : {res['n_conflicting']} "
          f"({res['pct_conflicting_of_nonneutral']:.1f}% of non-neutral concepts)")
    if res["top_conflicting"]:
        print("  most conflicted concepts (contrast = max_k D - min_k D):")
        for r in res["top_conflicting"][:5]:
            print(f"    concept {r['concept']:>6}  contrast={r['contrast']:.5f}  "
                  f"supports {r['argmax_class']}, harms {r['argmin_class']}")
    return res


def a4_per_class(s, tau_d, floor, out, baseline_path, conflict):
    sup = s.support(floor)
    rows = []
    for k, name in enumerate(s.class_names):
        c = conflict["per_class"][name]
        rows.append({
            "class": name,
            "pairs_above_floor": int(sup[k].sum()),
            "supportive": int((sup[k] & (s.D[k] > tau_d)).sum()),
            "harmful": int((sup[k] & (s.D[k] < -tau_d)).sum()),
            "neutral": int((sup[k] & (np.abs(s.D[k]) <= tau_d)).sum()),
            "harmed_by_conflicting_concept": c["harmed_by_conflicting"],
        })
    df = pd.DataFrame(rows)

    if baseline_path:
        try:
            b = json.loads(Path(baseline_path).read_text())
            per = b["accuracy"]["sketch"]["per_class"]
            df["sketch_acc"] = df["class"].map(lambda n: per.get(n, float("nan")) * 100)
        except Exception as exc:
            print(f"  [warn] could not join baseline {baseline_path}: {exc}")

    df.to_csv(out / "per_class_counts.csv", index=False)
    print("\n--- A4  per-class counts (Table 4) ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def a5_masses(s, tau_d, tau_h, tau_r, floor, out):
    buckets = build_buckets(s, tau_d=tau_d, tau_h=tau_h, tau_r=tau_r,
                            support_floor=floor)
    rows = []
    for name, b in buckets.items():
        row = {"bucket": name, "definition": b.definition,
               "n_pairs": b.n_pairs, "sum_abs_D": b.d_mass}
        for cls, n in b.per_class_pairs.items():
            row[f"n_{cls}"] = n
        for cls, m in b.per_class_d_mass.items():
            row[f"mass_{cls}"] = m
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "bucket_masses.csv", index=False)

    print("\n--- A5  bucket populations and effect mass ---")
    print(df[["bucket", "n_pairs", "sum_abs_D"]].to_string(
        index=False, float_format=lambda v: f"{v:.6g}"))

    # E4 feasibility: can S+_lo's mass cover S+_hi's, per class?
    lo, hi = buckets["S+_lo"], buckets["S+_hi"]
    print("\n  E4 mass-match feasibility (supportive), per class:")
    for cls in s.class_names:
        a, b_ = lo.per_class_d_mass[cls], hi.per_class_d_mass[cls]
        ok = "lo>=hi OK" if a >= b_ else "lo<hi  -> must match in reverse"
        print(f"    {cls:<10} S+_lo={a:.3e}  S+_hi={b_:.3e}   {ok}")
    return buckets


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_cfg = {
        "scores": args.scores, "tau_d": args.tau_d, "taus": args.taus,
        "support_floor": args.support_floor, "out_dir": str(out),
    }

    with RunRecorder("analyze_scores", config=manifest_cfg) as rec:
        s = load_scores(args.scores, class_names=list(cfg.pacs.class_names))
        print(f"[scores] {args.scores}: {s.n_classes} classes x {s.n_concepts} "
              f"concepts, {s.n_domains} source domains")
        print(f"[scores] pairs above support floor {args.support_floor}: "
              f"{int(s.support(args.support_floor).sum())}")

        tau0 = args.taus[0]
        with rec.step("A1_typology"):
            a1_typology(s, args.tau_d, args.taus, args.support_floor, out)
            a1_figures(s, args.tau_d, tau0, args.support_floor, out)

        with rec.step("A2_domain_attribution"):
            a2_attribution(s, cfg.score_domains, args.tau_d, tau0,
                           args.support_floor, out)

        with rec.step("A3_conflict"):
            conflict = a3_conflict(s, args.tau_d, args.support_floor, out)

        with rec.step("A4_per_class"):
            a4_per_class(s, args.tau_d, args.support_floor, out,
                         args.baseline, conflict)

        with rec.step("A5_bucket_masses"):
            a5_masses(s, args.tau_d, tau0, tau0, args.support_floor, out)

        for f in sorted(out.glob("*")):
            rec.add_output(f)

    print(f"\nwrote analysis artifacts to {out}")


if __name__ == "__main__":
    main()
