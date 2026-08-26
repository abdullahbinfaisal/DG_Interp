r"""
Read-only lookups that resolve the open items in docs/critique.md.

No GPU, no masking, no re-scoring: one pass over the clean score file. Every
number printed here is destined for a specific line of the manuscript, so each
block names the critique item it closes.

  python scripts\answer_critique.py --scores processed\FINAL_ERM_ResNet_3300_T3.json

Blocks
  Q1  concept cards for every concept named in the text or a figure   -> P0-6, P2-8
  Q2  most-conflicted concepts by contrast max_k D - min_k D          -> P0-7
  Q3  nonzero-effect pairs below the support floor                    -> P0-10
  Q4  gated invariant population clearing tau_R                       -> P0-9
  Q5  bucket predicates, sizes, masses and set nesting                -> P0-1, P0-11
  Q6  the population funnel, top to bottom                            -> P2-2
  Q7  high-H / low-R exemplar candidates                              -> P2-8 #1
  Q8  support-floor and tau_D sensitivity sweeps                      -> 4.4 thresholds
  Q9  what the inert set contains; why the floor exists               -> 4.3, 5.4
  Q10 low-R exemplars OWNED by the class shown                        -> P2-8 #1

Run a subset with --only, e.g.  --only q1 q10  (or  --only 1 10).

Writes results/analysis/critique_lookups.json alongside the stdout report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from clean_lib.config import get_preset
from clean_lib.masks import Scores, build_buckets, load_scores
from clean_lib.provenance import RunRecorder

# Every concept id that appears in the manuscript, in docs/RESULTS_DRAFT.md, in a
# figure filename, or in an analysis/ folder name. Q1 prints a card for each so
# the captions can be re-quoted from the clean file rather than from folder names.
FIGURE_CONCEPTS = {
    1637: "docs/figures/concept_dogchest.png; analysis HIND folder (chest/forelimb)",
    2979: "RESULTS_DRAFT 5.6 chest archetype, horse/dog",
    4501: "RESULTS_DRAFT 5.6 conflict, person/dog; analysis HIPD folder",
    3128: "RESULTS_DRAFT 5.6 conflict, dog/elephant",
    9839: "analysis LIPD folder, cartoon-concentrated elephant (critique P2-8 #1/#3)",
    1893: "analysis HILD folder 'Lines' (critique P2-8 #2 candidate)",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="results/analysis")
    p.add_argument("--preset", default="erm_resnet_3300")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--tau-h", type=float, default=0.7)
    p.add_argument("--tau-r", type=float, default=0.7)
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--top-n", type=int, default=10,
                   help="how many conflicted concepts to list in Q2")
    p.add_argument("--concepts", type=int, nargs="*", default=None,
                   help="override the Q1 concept list")
    p.add_argument("--only", nargs="*", default=None, metavar="BLOCK",
                   help="run a subset of blocks, e.g. --only q1 q10 (or 1 10). "
                        "Default runs all of q1..q10.")
    return p.parse_args()


def _wanted(only):
    """Normalise --only into a set of block keys; None means everything."""
    if not only:
        return None
    out = set()
    for tok in only:
        tok = str(tok).strip().lower().lstrip("q")
        if not tok.isdigit():
            raise SystemExit(f"--only takes block numbers like q1 or 10, got {tok!r}")
        out.add(f"q{int(tok)}")
    return out


def _rule(title: str, item: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n  resolves: {item}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Q1  concept cards                                              P0-6, P2-8
# ---------------------------------------------------------------------------


def q1_concept_cards(s: Scores, concepts, tau_d, floor, domains):
    _rule("Q1  concept cards", "P0-6 figure polarity/identity, P2-8 caption re-quoting")
    print("Every class row for each concept. 'regime' is at tau_D = "
          f"{tau_d:g}; rows below the support floor of {floor} are marked, since "
          "their D_d and R are not trustworthy.\n")

    out = {}
    for c in concepts:
        note = FIGURE_CONCEPTS.get(c, "")
        print(f"--- concept {c}" + (f"   [{note}]" if note else ""))
        print(f"    {'class':<10} {'H':>7} {'D':>12} {'R':>7} {'n_act':>7} "
              f"{'regime':>11}   per-domain D_d " + str(list(domains)))
        rows = []
        for k, name in enumerate(s.class_names):
            d = float(s.D[k, c])
            n = int(s.H_counts[k, c])
            if d > tau_d:
                regime = "supportive"
            elif d < -tau_d:
                regime = "harmful"
            else:
                regime = "neutral"
            flag = "" if n >= floor else "  (below floor)"
            dd = "  ".join(f"{v:+.3e}" for v in s.R_acts[k, c])
            print(f"    {name:<10} {s.H[k, c]:7.4f} {d:+12.5e} {s.R[k, c]:7.4f} "
                  f"{n:7d} {regime:>11}   [{dd}]{flag}")
            rows.append({
                "class": name, "H": float(s.H[k, c]), "D": d, "R": float(s.R[k, c]),
                "H_counts": n, "regime": regime,
                "D_per_domain": [float(v) for v in s.R_acts[k, c]],
                "above_floor": n >= floor,
            })
        out[str(c)] = {"note": note, "rows": rows}

        # The single question P0-6 asks: which class does this concept support and
        # which does it harm, among above-floor classes only.
        sup = [r["class"] for r in rows if r["regime"] == "supportive" and r["above_floor"]]
        harm = [r["class"] for r in rows if r["regime"] == "harmful" and r["above_floor"]]
        print(f"    => supports {sup or '(none)'} ; harms {harm or '(none)'}"
              f"   [above-floor classes only]")
        out[str(c)]["supports"] = sup
        out[str(c)]["harms"] = harm
    return out


# ---------------------------------------------------------------------------
# Q2  most-conflicted concepts                                          P0-7
# ---------------------------------------------------------------------------


def q2_conflict_ranking(s: Scores, tau_d, floor, top_n):
    _rule("Q2  most-conflicted concepts by contrast max_k D - min_k D",
          "P0-7 the broken sentence in 5.6")
    sup = s.support(floor)
    D = np.where(sup, s.D, np.nan)

    with np.errstate(invalid="ignore"):
        dmax = np.nanmax(D, axis=0)
        dmin = np.nanmin(D, axis=0)
    contrast = dmax - dmin

    # A conflict requires a genuinely supportive class and a genuinely harmful
    # one, both above the floor -- not merely a large spread across neutral cells.
    conflicting = (dmax > tau_d) & (dmin < -tau_d)
    contrast = np.where(conflicting, contrast, -np.inf)

    order = np.argsort(-contrast)[:top_n]
    print(f"  {int(np.isfinite(contrast).sum())} conflicting concepts "
          f"(supports one class above +tau_D, harms another below -tau_D, "
          f"both above the support floor of {floor}).\n")
    print(f"  {'rank':>4} {'concept':>8} {'contrast':>11}   supports (D)"
          f"                      harms (D)")
    rows = []
    for i, c in enumerate(order, 1):
        if not np.isfinite(contrast[c]):
            break
        kmax = int(np.nanargmax(D[:, c]))
        kmin = int(np.nanargmin(D[:, c]))
        sname, hname = s.class_names[kmax], s.class_names[kmin]
        print(f"  {i:>4} {c:>8} {contrast[c]:11.5e}   "
              f"{sname:<10} ({dmax[c]:+.3e})     {hname:<10} ({dmin[c]:+.3e})")
        rows.append({
            "rank": i, "concept": int(c), "contrast": float(contrast[c]),
            "supports": sname, "D_support": float(dmax[c]),
            "harms": hname, "D_harm": float(dmin[c]),
        })
    print("\n  The manuscript sentence must name the top three of these, with ids.")
    return rows


# ---------------------------------------------------------------------------
# Q3  nonzero effect below the floor                                   P0-10
# ---------------------------------------------------------------------------


def q3_below_floor(s: Scores, tau_d, floor):
    _rule("Q3  pairs with nonzero measured effect below the support floor",
          "P0-10 the 1,616 pairs the paper never mentions")
    total = s.H_counts.size
    sup = s.support(floor)
    inert = s.R == 0.0                       # no measured effect in any source domain
    absd = np.abs(s.D)

    below = (~sup) & (~inert)
    n_below = int(below.sum())
    mass_below = float(absd[below].sum())
    n_gated = int((below & (absd > tau_d)).sum())
    mass_gated = float(absd[below & (absd > tau_d)].sum())

    total_mass = float(absd[~inert].sum())

    print(f"  total grid pairs                      {total:>10,}")
    print(f"  inert (R == 0, no effect anywhere)    {int(inert.sum()):>10,}")
    print(f"  above support floor ({floor})            {int(sup.sum()):>10,}")
    print(f"  -> nonzero effect, BELOW floor        {n_below:>10,}"
          f"   sum|D| = {mass_below:.6g}")
    print(f"     of those, |D| > tau_D              {n_gated:>10,}"
          f"   sum|D| = {mass_gated:.6g}")
    print(f"\n  check: {int(inert.sum()):,} + {int(sup.sum()):,} + {n_below:,} = "
          f"{int(inert.sum()) + int(sup.sum()) + n_below:,} "
          f"(should be {total:,}; overlap of inert with above-floor is "
          f"{int((sup & inert).sum()):,})")
    print(f"\n  share of all non-inert effect mass carried below the floor: "
          f"{mass_below / total_mass * 100:.2f}%")
    print("  If this share is small, one sentence in 4.3 closes the item; the "
          "funnel figure gets a labelled box either way.")
    return {
        "n_total": total, "n_inert": int(inert.sum()), "n_above_floor": int(sup.sum()),
        "n_below_floor_nonzero": n_below, "mass_below_floor": mass_below,
        "n_below_floor_gated": n_gated, "mass_below_floor_gated": mass_gated,
        "share_of_noninert_mass_pct": mass_below / total_mass * 100,
        "n_inert_and_above_floor": int((sup & inert).sum()),
    }


# ---------------------------------------------------------------------------
# Q4  the gated invariant population                                    P0-9
# ---------------------------------------------------------------------------


def q4_gated_invariant(s: Scores, tau_d, tau_h, tau_r, floor):
    _rule("Q4  fraction of the invariant population whose effect is consistent",
          "P0-9 the 50.7% drawn from an R-meaningless population")
    sup = s.support(floor)
    hiH, hiR = s.H >= tau_h, s.R >= tau_r
    nonneutral = np.abs(s.D) > tau_d

    ungated_n = int((sup & hiH).sum())
    ungated_hi = int((sup & hiH & hiR).sum())
    gated_n = int((sup & hiH & nonneutral).sum())
    gated_hi = int((sup & hiH & nonneutral & hiR).sum())

    print(f"  UNGATED  (what the draft reports)")
    print(f"    high-H pairs                     {ungated_n:>6}")
    print(f"    also high-R                      {ungated_hi:>6}   "
          f"{ungated_hi / ungated_n * 100:.1f}%")
    neutral_frac = int((sup & hiH & ~nonneutral).sum()) / ungated_n * 100
    print(f"    of which NEUTRAL                 "
          f"{int((sup & hiH & ~nonneutral).sum()):>6}   {neutral_frac:.1f}%"
          f"   <- why the ungated figure is unusable")
    print(f"\n  GATED  (|D| > tau_D, the population the paper says R applies to)")
    print(f"    high-H non-neutral pairs         {gated_n:>6}")
    print(f"    also high-R                      {gated_hi:>6}   "
          f"{gated_hi / gated_n * 100:.1f}%   <- headline figure for 5.3")

    for lab, sel in (("supportive", s.D > tau_d), ("harmful", s.D < -tau_d)):
        grp = sup & hiH & sel
        n, hi = int(grp.sum()), int((grp & hiR).sum())
        print(f"      {lab:<11} {hi:>4}/{n:<4}  {hi / n * 100:.1f}%")

    return {
        "ungated_high_H": ungated_n, "ungated_also_high_R": ungated_hi,
        "ungated_pct": ungated_hi / ungated_n * 100,
        "ungated_neutral_pct": neutral_frac,
        "gated_high_H": gated_n, "gated_also_high_R": gated_hi,
        "gated_pct": gated_hi / gated_n * 100,
    }


# ---------------------------------------------------------------------------
# Q5  bucket predicates and nesting                              P0-1, P0-11
# ---------------------------------------------------------------------------


def q5_buckets(s: Scores, tau_d, tau_h, tau_r, floor):
    _rule("Q5  bucket predicates, sizes and set nesting",
          "P0-1 the 35-vs-32 mislabelling, P0-11 the undefined predicates")
    buckets = build_buckets(s, tau_d=tau_d, tau_h=tau_h, tau_r=tau_r,
                            support_floor=floor)

    print(f"  {'bucket':<20} {'pairs':>7} {'sum|D|':>10}   predicate")
    rows = {}
    for name, b in buckets.items():
        if b.keep_only:
            continue
        print(f"  {name:<20} {b.n_pairs:>7} {b.d_mass:>10.4f}   {b.definition}")
        rows[name] = {"n_pairs": b.n_pairs, "d_mass": b.d_mass,
                      "definition": b.definition}

    # The claim P0-1 turns on: is harmful_invariant a strict subset of S-_hi,
    # and how many pairs separate them.
    shi = buckets["S-_hi"].select
    hinv = buckets["harmful_invariant"].select
    strict = bool((hinv & ~shi).sum() == 0)
    extra = shi & ~hinv
    print(f"\n  NESTING CHECK")
    print(f"    S-_hi  (distributed harm, R only)        {int(shi.sum()):>4} pairs")
    print(f"    harmful_invariant (H and R)              {int(hinv.sum()):>4} pairs")
    print(f"    harmful_invariant subset of S-_hi?       {strict}")
    print(f"    in S-_hi but NOT invariant (low H)       {int(extra.sum()):>4} pairs"
          f"   sum|D| = {float(np.abs(s.D)[extra].sum()):.4f}")
    for k, c in zip(*np.nonzero(extra)):
        print(f"      ({s.class_names[k]}, {c})  H={s.H[k, c]:.4f} "
              f"R={s.R[k, c]:.4f} D={s.D[k, c]:+.4e}")

    # Same question on the supportive side: Table 3 says "distributed support (81)"
    # (R only) while Table 4 says "robust support (76)" (H and R). Same trap.
    sphi = buckets["S+_hi"].select
    rsup = (s.support(floor) & (s.D > tau_d) & (s.H >= tau_h) & (s.R >= tau_r))
    print(f"\n    S+_hi  (distributed support, R only)     {int(sphi.sum()):>4} pairs")
    print(f"    robust support (H and R)                 {int(rsup.sum()):>4} pairs")
    print(f"    in S+_hi but NOT invariant (low H)       {int((sphi & ~rsup).sum()):>4} pairs")
    print("\n  => the manuscript must not call an R-only bucket "
          "'invariant and consistent'.")

    rows["_nesting"] = {
        "S-_hi": int(shi.sum()), "harmful_invariant": int(hinv.sum()),
        "harmful_invariant_is_subset": strict,
        "S-_hi_not_invariant": int(extra.sum()),
        "S+_hi": int(sphi.sum()), "robust_support": int(rsup.sum()),
        "S+_hi_not_invariant": int((sphi & ~rsup).sum()),
    }
    return rows, buckets


# ---------------------------------------------------------------------------
# Q6  the population funnel                                             P2-2
# ---------------------------------------------------------------------------


def q6_funnel(s: Scores, buckets, tau_d, floor):
    _rule("Q6  the population funnel, top to bottom", "P2-2 the funnel figure")
    absd = np.abs(s.D)
    sup = s.support(floor)
    inert = s.R == 0.0
    below = (~sup) & (~inert)
    nonneutral = sup & (absd > tau_d)

    stages = [
        ("class-concept grid", s.H.size, float(absd.sum())),
        ("inert: no effect in any source domain", int(inert.sum()),
         float(absd[inert].sum())),
        ("nonzero effect, below support floor", int(below.sum()),
         float(absd[below].sum())),
        ("above support floor", int(sup.sum()), float(absd[sup].sum())),
        ("  neutral (|D| <= tau_D)", int((sup & ~(absd > tau_d)).sum()),
         float(absd[sup & ~(absd > tau_d)].sum())),
        ("  non-neutral", int(nonneutral.sum()), float(absd[nonneutral].sum())),
        ("    supportive (D > tau_D)", int((nonneutral & (s.D > 0)).sum()),
         float(absd[nonneutral & (s.D > 0)].sum())),
        ("    harmful (D < -tau_D)", int((nonneutral & (s.D < 0)).sum()),
         float(absd[nonneutral & (s.D < 0)].sum())),
    ]
    for name, b in buckets.items():
        if b.keep_only or name in ("all_harmful", "all_supportive", "inert_only"):
            continue
        stages.append((f"      {name}", b.n_pairs, b.d_mass))

    print(f"  {'stage':<48} {'pairs':>9} {'sum|D|':>10}  {'% of grid':>10}")
    rows = []
    for name, n, mass in stages:
        print(f"  {name:<48} {n:>9,} {mass:>10.4f}  {n / s.H.size * 100:>9.3f}%")
        rows.append({"stage": name.strip(), "pairs": n, "d_mass": mass,
                     "pct_of_grid": n / s.H.size * 100})
    print("\n  Percentages here are the ones the abstract quotes. Check 0.028% "
          "against the harmful_invariant row before editing the abstract.")
    return rows


# ---------------------------------------------------------------------------
# Q7  exemplar candidates for the high-H / low-R figure               P2-8 #1
# ---------------------------------------------------------------------------


def q7_exemplars(s: Scores, tau_d, tau_h, tau_r, floor, domains, top_n):
    _rule("Q7  high-H, low-R, non-neutral exemplar candidates",
          "P2-8 #1 the Figure 6 exemplar (9839 turned out to be LOW-H, so unusable)")
    sup = s.support(floor)
    absd = np.abs(s.D)
    sel = sup & (absd > tau_d) & (s.H >= tau_h) & (s.R < tau_r)

    print(f"  {int(sel.sum())} pairs are invariant, non-neutral and inconsistent.")
    print("  A good figure needs high H (flat activation bars), low R (one spiked")
    print("  effect bar), large |D| (so the spike is visible) and plenty of active")
    print("  images. Ranked by (H - R) * log10(1 + |D| / tau_D).\n")

    ks, cs = np.nonzero(sel)
    score = (s.H[ks, cs] - s.R[ks, cs]) * np.log10(1 + absd[ks, cs] / tau_d)
    order = np.argsort(-score)[:top_n]

    print(f"  {'concept':>8} {'class':<10} {'H':>7} {'R':>7} {'D':>12} {'n_act':>7} "
          f"{'peak':<13} share  per-domain |D_d|")
    rows = []
    for i in order:
        k, c = int(ks[i]), int(cs[i])
        dd = np.abs(s.R_acts[k, c])
        pk = int(np.argmax(dd))
        share = dd[pk] / dd.sum() if dd.sum() > 0 else 0.0
        print(f"  {c:>8} {s.class_names[k]:<10} {s.H[k, c]:7.4f} {s.R[k, c]:7.4f} "
              f"{s.D[k, c]:+12.4e} {int(s.H_counts[k, c]):7d} {domains[pk]:<13} "
              f"{share * 100:4.0f}%  [" + "  ".join(f"{v:.2e}" for v in dd) + "]")
        rows.append({
            "concept": c, "class": s.class_names[k], "H": float(s.H[k, c]),
            "R": float(s.R[k, c]), "D": float(s.D[k, c]),
            "H_counts": int(s.H_counts[k, c]), "peak_domain": domains[pk],
            "peak_share": float(share), "regime": "supportive" if s.D[k, c] > 0 else "harmful",
        })

    print("\n  For Figure 6 to also serve 5.6 it should be a SUPPORTIVE pair peaking")
    print("  in cartoon, which is the enrichment 5.6 reports. Candidates matching that:")
    for r in rows:
        if r["regime"] == "supportive" and r["peak_domain"] == "cartoon":
            print(f"    concept {r['concept']} / {r['class']}  H={r['H']:.3f} "
                  f"R={r['R']:.3f} D={r['D']:+.3e}  {r['peak_share']*100:.0f}% in cartoon")
    return rows


# ---------------------------------------------------------------------------
# Q8  support-floor and tau_D sensitivity
# ---------------------------------------------------------------------------


def q8_floor_and_taud(s: Scores, tau_h, tau_r,
                      floors=(10, 20, 30, 50, 100),
                      tau_ds=(1e-5, 1e-4, 1e-3)):
    """
    Two justification sweeps the paper currently lacks.

    The support floor and tau_D are the only two thresholds with no argument
    behind them: tau_H = tau_R = 0.7 has the log2/log3 bound, these have nothing.
    The floor exists so R -- an entropy over three per-domain means -- is
    estimable at all, so the question is whether the 5.3 asymmetry survives
    moving it. tau_D is a pure magnitude gate, so the question is whether the
    asymmetry survives moving that too.
    """
    _rule("Q8  support floor and tau_D sensitivity",
          "justification for the floor (and the 1,629 pairs it excludes) and for tau_D")

    absd = np.abs(s.D)
    inert = s.R == 0.0
    noninert_mass = float(absd[~inert].sum())
    hiH, hiR = s.H >= tau_h, s.R >= tau_r

    print(f"  (A) SUPPORT FLOOR, at tau_D = 1e-4, tau_H = tau_R = {tau_h}\n")
    print(f"  {'floor':>6} {'above':>7} {'excluded':>9} {'excl Σ|D|':>10} {'excl %':>7} "
          f"{'nonneut':>8} {'supp%':>7} {'harm%':>7} {'ratio':>6}")
    rows_a = []
    for f in floors:
        sup = s.support(f)
        below = (~sup) & (~inert)
        nn = sup & (absd > 1e-4)
        rec = {"floor": f, "n_above": int(sup.sum()),
               "n_excluded": int(below.sum()),
               "excluded_mass": float(absd[below].sum()),
               "excluded_mass_pct": float(absd[below].sum()) / noninert_mass * 100,
               "n_nonneutral": int(nn.sum())}
        for lab, sel in (("supp", s.D > 1e-4), ("harm", s.D < -1e-4)):
            grp = sup & sel & hiH
            rec[f"{lab}_pct"] = int((grp & hiR).sum()) / max(int(grp.sum()), 1) * 100
        rec["ratio"] = rec["supp_pct"] / rec["harm_pct"] if rec["harm_pct"] else float("nan")
        print(f"  {f:>6} {rec['n_above']:>7,} {rec['n_excluded']:>9,} "
              f"{rec['excluded_mass']:>10.3f} {rec['excluded_mass_pct']:>6.1f}% "
              f"{rec['n_nonneutral']:>8} {rec['supp_pct']:>6.1f}% {rec['harm_pct']:>6.1f}% "
              f"{rec['ratio']:>6.2f}")
        rows_a.append(rec)
    print("\n  Read the last column. If the asymmetry ratio is stable across floors,")
    print("  the floor is a nuisance parameter and can be defended as such; if it")
    print("  moves, the choice of 30 is load-bearing and must be argued directly.")

    print(f"\n  (B) tau_D, at support floor 30, tau_H = tau_R = {tau_h}\n")
    print(f"  {'tau_D':>8} {'neutral':>8} {'supp':>6} {'harm':>6} "
          f"{'supp%':>7} {'harm%':>7} {'ratio':>6}")
    sup30 = s.support(30)
    rows_b = []
    for td in tau_ds:
        rec = {"tau_d": td,
               "n_neutral": int((sup30 & (absd <= td)).sum()),
               "n_supp": int((sup30 & (s.D > td)).sum()),
               "n_harm": int((sup30 & (s.D < -td)).sum())}
        for lab, sel in (("supp", s.D > td), ("harm", s.D < -td)):
            grp = sup30 & sel & hiH
            rec[f"{lab}_pct"] = int((grp & hiR).sum()) / max(int(grp.sum()), 1) * 100
        rec["ratio"] = rec["supp_pct"] / rec["harm_pct"] if rec["harm_pct"] else float("nan")
        print(f"  {td:>8.0e} {rec['n_neutral']:>8,} {rec['n_supp']:>6} {rec['n_harm']:>6} "
              f"{rec['supp_pct']:>6.1f}% {rec['harm_pct']:>6.1f}% {rec['ratio']:>6.2f}")
        rows_b.append(rec)

    # Is there a natural gap at 1e-4, or is the distribution smooth there?
    v = np.sort(absd[sup30 & (absd > 0)])
    print("\n  |D| distribution over above-floor pairs with nonzero effect "
          f"(n = {v.size:,}):")
    for q in (50, 75, 80, 82, 85, 90, 95, 99):
        print(f"    p{q:<3} = {np.percentile(v, q):.3e}")
    print(f"    tau_D = 1e-4 sits at percentile "
          f"{(v < 1e-4).sum() / v.size * 100:.1f}")
    print("\n  A threshold that lands inside a smooth region is an arbitrary cut and")
    print("  should be defended by the sweep above, not by the distribution.")
    return {"floor_sweep": rows_a, "tau_d_sweep": rows_b}


# ---------------------------------------------------------------------------
# Q9  what "inert" actually contains, and why the floor exists
# ---------------------------------------------------------------------------


def q9_inert_and_floor(s: Scores, tau_d, floor):
    """
    Two things the paper currently asserts without having checked them.

    (a) 5.4 says the 111,530 R == 0 pairs "have zero measured effect in every
        source domain". R == 0 does NOT mean that. R is an entropy over the
        normalised |D_d|, so a pair whose effect sits in exactly ONE source
        domain also scores R = 0. If any R == 0 pair has D != 0, the sentence
        is wrong as written.

    (b) The floor is usually justified by "D_d is noisy at low support". The
        sharper reason is that R is not merely noisy at low support, it is
        BIASED TOWARDS ZERO: a domain in which the concept never fires
        contributes an exact zero to the entropy, so a thinly supported pair
        looks domain-concentrated whether or not it is.
    """
    _rule("Q9  the composition of the inert set, and what the floor is protecting",
          "the 5.4 'zero effect in every source domain' claim, and the floor's rationale")

    absd = np.abs(s.D)
    inert = s.R == 0.0
    n = s.H_counts
    dd = np.abs(s.R_acts)
    n_dom_active = (dd > 0).sum(axis=2)     # source domains with nonzero effect

    print("  (a) WHAT R == 0 ACTUALLY CONTAINS\n")
    truly_zero = inert & (absd == 0)
    one_domain = inert & (absd > 0)
    print(f"    R == 0 pairs                              {int(inert.sum()):>8,}")
    print(f"      of which D is exactly 0                 {int(truly_zero.sum()):>8,}")
    print(f"      of which D != 0 (effect in ONE domain)  {int(one_domain.sum()):>8,}"
          f"   sum|D| = {float(absd[one_domain].sum()):.4f}")
    print(f"      of those, |D| > tau_D                   "
          f"{int((one_domain & (absd > tau_d)).sum()):>8,}"
          f"   sum|D| = {float(absd[one_domain & (absd > tau_d)].sum()):.4f}")
    print(f"    never fires for the class at all (n = 0)  {int((n == 0).sum()):>8,}")
    if int(one_domain.sum()) > 0:
        print("\n    => the 5.4 sentence 'zero measured effect in every source domain'")
        print("       is FALSE for these pairs. Correct wording: 'no effect outside a")
        print("       single source domain', or gate the bucket on |D| == 0 instead.")

    print("\n  (b) WHY THE FLOOR EXISTS: R is biased toward 0 at low support\n")
    print(f"    {'support n(k,c)':<18} {'pairs':>8} {'mean R':>8} {'% with R = 0':>13} "
          f"{'mean domains active':>20}")
    bands = [(1, 4), (5, 9), (10, 19), (20, 29), (30, 49), (50, 99),
             (100, 499), (500, 10 ** 9)]
    rows = []
    for lo, hi in bands:
        sel = (n >= lo) & (n <= hi) & (absd > 0)
        c = int(sel.sum())
        if c == 0:
            continue
        rec = {"n_lo": lo, "n_hi": hi, "pairs": c,
               "mean_R": float(s.R[sel].mean()),
               "pct_R_zero": float((s.R[sel] == 0).mean() * 100),
               "mean_domains_active": float(n_dom_active[sel].mean())}
        lab = f"{lo}–{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"    {lab:<18} {c:>8,} {rec['mean_R']:>8.3f} "
              f"{rec['pct_R_zero']:>12.1f}% {rec['mean_domains_active']:>20.2f}")
        rows.append(rec)
    print(f"\n    The floor sits at n = {floor}. If mean R climbs and '% with R = 0'")
    print("    falls as support grows, then low-R at low support is a sampling")
    print("    artifact rather than genuine domain-contingency -- which is the")
    print("    argument for the floor, and it is about R alone. H and D are pooled")
    print("    over domains and do not suffer this.")

    print("\n  (c) SUPPORT COUNTS ARE PER CLASS, NOT POOLED\n")
    k0 = 0
    print(f"    H_counts has shape {tuple(n.shape)} = (classes, concepts):")
    print(f"    n(k,c) counts images OF CLASS k on which c fires, summed over the")
    print(f"    three source domains. Same concept, different classes, e.g. c = 1637:")
    for k in range(s.n_classes):
        print(f"      n({s.class_names[k]:<9}, 1637) = {int(n[k, 1637]):>5}"
              f"    {'above floor' if n[k, 1637] >= floor else 'below floor'}")
    return {"n_inert": int(inert.sum()), "n_truly_zero": int(truly_zero.sum()),
            "n_one_domain": int(one_domain.sum()),
            "mass_one_domain": float(absd[one_domain].sum()),
            "n_never_fires": int((n == 0).sum()), "support_bands": rows}


# ---------------------------------------------------------------------------
# Q10  low-R exemplars that the displayed class actually OWNS
# ---------------------------------------------------------------------------


def q10_owned_exemplars(s: Scores, tau_d, tau_h, tau_r, floor, domains, top_n):
    """
    Q7 ranked low-R pairs by score alone and every winner turned out to be a
    concept whose home is a different class -- a giraffe latent shown on dog, an
    elephant latent shown on dog. A top-activating grid of such a pair is close
    to uninterpretable, and worse, implies the concept is about a class it is not.

    That is not bad luck. Section 5.5 measures it: 78.3% of harmful pairs involve
    a concept that supports some other class, so conditioning on "harmful" almost
    guarantees the concept lives elsewhere. The fix is to add OWNERSHIP as an
    explicit criterion rather than hoping for it:

        share(k,c) = n(k,c) / sum_k' n(k',c)

    the fraction of the concept's firing that lands on the displayed class. A
    grid is representative when that share is high and the class is the argmax.

    Supportive pairs are listed first because for them the tension dissolves:
    D > 0 means the concept is evidence FOR that class, so it fires on its own
    home by construction.
    """
    _rule("Q10  low-R exemplars owned by the class they would be shown on",
          "Figure 7 exemplar; every Q7 winner was a concept belonging to another class")

    sup = s.support(floor)
    absd = np.abs(s.D)
    n = s.H_counts
    tot_fire = n.sum(axis=0, keepdims=True).clip(min=1)     # per concept
    share = n / tot_fire
    owner = np.argmax(n, axis=0)                            # class per concept

    base = sup & (absd > tau_d) & (s.H >= tau_h) & (s.R < tau_r)
    is_owner = np.zeros_like(base)
    for k in range(s.n_classes):
        is_owner[k] = owner == k

    for label, sel in (("SUPPORTIVE  (recommended: concept fires on its own class)",
                        base & (s.D > tau_d)),
                       ("HARMFUL  (shown for completeness; expect low ownership)",
                        base & (s.D < -tau_d))):
        print(f"\n  {label}")
        ks, cs = np.nonzero(sel)
        if ks.size == 0:
            print("    (none)")
            continue
        rows = []
        for k, c in zip(ks, cs):
            dd = np.abs(s.R_acts[k, c])
            pk = int(np.argmax(dd))
            rows.append(dict(
                concept=int(c), cls=s.class_names[k], H=float(s.H[k, c]),
                R=float(s.R[k, c]), D=float(s.D[k, c]), n=int(n[k, c]),
                share=float(share[k, c]), owns=bool(is_owner[k, c]),
                owner_class=s.class_names[int(owner[c])],
                peak=domains[pk],
                peak_share=float(dd[pk] / dd.sum()) if dd.sum() > 0 else 0.0))
        # ownership first, then how cleanly H and R disagree
        rows.sort(key=lambda r: (r["owns"], r["share"], r["H"] - r["R"]), reverse=True)

        print(f"    {'concept':>8} {'class':<10} {'H':>6} {'R':>6} {'D':>11} "
              f"{'n':>5} {'share':>6} {'owner':<10} {'peak':<13} {'peak%':>6}")
        for r in rows[:top_n]:
            flag = "  <-- owned" if r["owns"] and r["share"] >= 0.5 else ""
            print(f"    {r['concept']:>8} {r['cls']:<10} {r['H']:6.3f} {r['R']:6.3f} "
                  f"{r['D']:+11.3e} {r['n']:>5} {r['share']*100:5.0f}% "
                  f"{r['owner_class']:<10} {r['peak']:<13} {r['peak_share']*100:5.0f}%"
                  f"{flag}")

    print("\n  Pick from the supportive block with share >= 50% and owner == class.")
    print("  If one of those also peaks in cartoon it serves 5.6 as well, since")
    print("  that section reports concentrated support enriched 2.3x in cartoon")
    print("  and currently has no exemplar at all.")
    return None


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_cfg = {
        "scores": args.scores, "tau_d": args.tau_d, "tau_h": args.tau_h,
        "tau_r": args.tau_r, "support_floor": args.support_floor,
        "out_dir": str(out),
    }

    with RunRecorder("answer_critique", config=manifest_cfg) as rec:
        s = load_scores(args.scores, class_names=list(cfg.pacs.class_names))
        print(f"[scores] {args.scores}: {s.n_classes} classes x {s.n_concepts} "
              f"concepts, {s.n_domains} source domains")
        if s.n_domains != 3:
            print("  WARNING: expected 3 source domains. A length-4 R_acts means "
                  "this is a contaminated score file (CLAUDE.md section 5).")

        concepts = args.concepts if args.concepts else sorted(FIGURE_CONCEPTS)
        result = {}
        state = {}          # q5 hands its buckets to q6

        def q5():
            rows, buckets = q5_buckets(s, args.tau_d, args.tau_h, args.tau_r,
                                       args.support_floor)
            state["buckets"] = buckets
            return rows

        def q6():
            # q6 needs q5's buckets; build them quietly if q5 was not selected
            if "buckets" not in state:
                state["buckets"] = build_buckets(
                    s, tau_d=args.tau_d, tau_h=args.tau_h, tau_r=args.tau_r,
                    support_floor=args.support_floor)
            return q6_funnel(s, state["buckets"], args.tau_d, args.support_floor)

        BLOCKS = [
            ("q1", "Q1_concept_cards", lambda: q1_concept_cards(
                s, concepts, args.tau_d, args.support_floor, cfg.score_domains)),
            ("q2", "Q2_conflict_ranking", lambda: q2_conflict_ranking(
                s, args.tau_d, args.support_floor, args.top_n)),
            ("q3", "Q3_below_floor", lambda: q3_below_floor(
                s, args.tau_d, args.support_floor)),
            ("q4", "Q4_gated_invariant", lambda: q4_gated_invariant(
                s, args.tau_d, args.tau_h, args.tau_r, args.support_floor)),
            ("q5", "Q5_buckets", q5),
            ("q6", "Q6_funnel", q6),
            ("q7", "Q7_exemplars", lambda: q7_exemplars(
                s, args.tau_d, args.tau_h, args.tau_r, args.support_floor,
                cfg.score_domains, args.top_n)),
            ("q8", "Q8_floor_and_taud", lambda: q8_floor_and_taud(
                s, args.tau_h, args.tau_r)),
            ("q9", "Q9_inert_and_floor", lambda: q9_inert_and_floor(
                s, args.tau_d, args.support_floor)),
            ("q10", "Q10_owned_exemplars", lambda: q10_owned_exemplars(
                s, args.tau_d, args.tau_h, args.tau_r, args.support_floor,
                cfg.score_domains, args.top_n)),
        ]

        want = _wanted(args.only)
        unknown = (want or set()) - {k for k, _, _ in BLOCKS}
        if unknown:
            raise SystemExit(f"unknown block(s): {sorted(unknown)}")

        for key, step_name, fn in BLOCKS:
            if want is not None and key not in want:
                continue
            with rec.step(step_name):
                out_val = fn()
                if out_val is not None:
                    result[step_name.lower()] = out_val

        # A partial run must not clobber a full one: it would look complete on
        # disk while silently missing blocks.
        dest = out / ("critique_lookups.json" if want is None
                      else f"critique_lookups_{'_'.join(sorted(want))}.json")
        tmp = dest.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"config": manifest_cfg, "results": result}, fh, indent=2)
        tmp.replace(dest)
        rec.add_output(dest)

    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
