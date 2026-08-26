r"""
The five data-driven figures for the manuscript (critique P2-2..P2-5).

  python scripts\make_figures.py --scores processed\FINAL_ERM_ResNet_3300_T3.json

Produces, under docs/figures/:
  fig2_funnel.png           P2-2  the class-concept population, grid -> buckets
  fig3_h_vs_d.png           P2-3  H vs D, symlog, coloured by regime
  fig7_hr_heatmap.png       P2-4  H x R contingency by D regime
  fig8_tau_sensitivity.png  P2-4  the support/harm asymmetry swept over tau
  fig10_interventions.png    P2-5  intervention deltas against matched controls

Figure numbers match the manuscript. 1 (pipeline), 4/5/6 (the three
regime plates), 9 (high-H/low-R exemplar) and 11 (conflict plate) are
author-built.

The three qualitative concept plates (P2-8) and the pipeline schematic (P2-1) are
not produced here: they need image grids and hand layout respectively.

COLOR. One diverging scale is used in every figure, because every figure encodes
the same thing: harmful / neutral / supportive. Harmful and supportive are the two
poles, neutral gray is the midpoint. The pair was checked for colour-vision
deficiency separation (worst adjacent dE 20.6 protan, 26.9 tritan) rather than
chosen by eye. Matched random controls are NOT given a fourth hue -- a fourth
categorical hue fails CVD separation against the blue -- they reuse the target's
hue at a lighter tint plus a hatch, so the distinction survives greyscale printing
and colourblind readers alike.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from clean_lib.config import get_preset
from clean_lib.masks import build_buckets, load_scores
from clean_lib.provenance import RunRecorder

# --- the one palette -------------------------------------------------------
SUPPORT = "#1f6fb2"
HARM = "#c1622a"
NEUTRAL = "#8a8f98"
INERT = "#c8ccd1"
# Paired light tints for the funnel: a parent row and its children share a hue,
# parent dark and children light, so depth is legible without relying on row
# position alone.
#
# Validated as the ADJACENCY CHAIN in figure row order (grey, grey-tint, blue,
# blue-tint, orange, orange-tint) rather than as an all-pairs categorical set,
# because the tints never touch each other: light grey sits above dark blue and
# light blue above dark orange, so e.g. blue-tint vs grey-tint is not a
# comparison the figure ever asks a reader to make. In that chain the palette
# passes the lightness band and CVD separation. Three checks are knowingly
# accepted:
#   - chroma floor on the two greys: they are meant to read grey (neutral role)
#   - lightness ceiling grazed by the orange tint at 0.771 against a 0.77 band
#   - contrast < 3:1 on the tints: relieved by the visible count on every bar
#     and the full predicate on every row
# Lightening the tints raised the within-hue separation rather than lowering it:
# dark-vs-light orange went from dE 11.5 (failing the 15 floor) to 17.5.
DARK_GREY = "#5c6166"
LIGHT_GREY = "#a8aeb4"
LIGHT_BLUE = "#79aedd"
LIGHT_ORANGE = "#e8a36e"
INK = "#1a1c1e"
INK_SOFT = "#5c6166"
GRID = "#e3e5e8"
SURFACE = "#ffffff"

# Sequential ramp for the heatmap: one hue, light -> dark. Never a rainbow.
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fa", "#bcd4e8", "#5b9bcd", "#1f6fb2", "#14486f"])

REGIME_COLOR = {"harmful": HARM, "neutral": NEUTRAL, "supportive": SUPPORT}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", required=True)
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out-dir", default="docs/figures")
    p.add_argument("--preset", default="erm_resnet_3300")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--taus", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of {funnel,scatter,heatmap,sensitivity,"
                        "interventions}")
    return p.parse_args()


# ---------------------------------------------------------------------------
# P2-2  population funnel
# ---------------------------------------------------------------------------


def fig_funnel(s, buckets, tau_d, floor, out: Path):
    """
    The funnel as an indented two-measure chart rather than nested rectangles.

    A drawn-to-scale nested funnel cannot work here: the populations span 32 to
    114,688, so every box below the first level collapses to a hairline, and
    log-scaled *widths* would misstate proportion. Indentation carries the
    nesting instead, and the two panels carry the quantities.

    COUNTS ONLY, deliberately. A discrimination-mass panel was tried twice and
    cut both times, and the reason is not aesthetic: a panel should encode a
    comparison that is visually valid. A count is a count, so 113,146 against
    1,542 means what it looks like. Sum |D| across populations of different sizes
    does not -- |D| is a magnitude, so noise adds and never cancels, and the bar
    length tracks population size as much as importance. The paper measures this
    twice: the 111,530 inert pairs out-mass the 35-pair distributed-harm bucket
    and recover 40x less target error, and the 282 harmful pairs below the
    support floor out-mass all 138 above it and recover a sixth as much.

    A caption cannot repair that. If a panel needs "do not read these bars as
    importance" printed underneath it, the encoding is wrong. Mass appears in
    Table 3 instead, where the delta, errors-recovered, per-unit-mass and matched
    control columns sit beside it and make it interpretable.
    """
    absd = np.abs(s.D)
    sup = s.support(floor)
    total = s.H.size

    nonneutral = sup & (absd > tau_d)
    neutral = sup & ~(absd > tau_d)
    posn = nonneutral & (s.D > 0)
    negn = nonneutral & (s.D < 0)
    assert int((~sup).sum()) + int(sup.sum()) == total

    b = buckets

    def row(tier, label, sel_or_n, denom, color=NEUTRAL, rule=False):
        n = int(sel_or_n.sum()) if isinstance(sel_or_n, np.ndarray) else sel_or_n
        return dict(tier=tier, label=label, n=n, pct=n / denom * 100,
                    color=color, rule=rule)

    n_sup, n_nn = int(sup.sum()), int(nonneutral.sum())
    n_pos, n_neg = int(posn.sum()), int(negn.sum())
    n_sphi, n_shhi = b["S+_hi"].n_pairs, b["S-_hi"].n_pairs

    # Flush-left labels: indentation was harder to read than it was worth.
    # Hierarchy is carried by ink weight and by a hairline above each new block,
    # and the bucket rows name their own symbol, which already states membership.
    # Every level splits on exactly ONE criterion, and each criterion appears at
    # the level where it is meaningful: support, then magnitude, then sign, then
    # R. R deliberately does NOT appear above the non-neutral split -- an R = 0
    # vs R > 0 cut at the top reads as a population distinction when it is really
    # a statement about where a pair's effect sits, and it is not interpretable
    # at all until the magnitude gate has been applied (Section 3.6).
    #
    # Every row carries its defining predicate, so the figure is readable without
    # cross-referencing Table 1.
    rows = [
        row(1, "Below support floor    $n(k,c) < 30$", (~sup), total, DARK_GREY),
        row(1, "Above support floor    $n(k,c) \\geq 30$", sup, total, DARK_GREY, rule=True),
        row(2, "Neutral    $|D| \\leq \\tau_D$", neutral, n_sup, LIGHT_GREY),
        row(2, "Non-neutral    $|D| > \\tau_D$", nonneutral, n_sup, LIGHT_GREY),
        row(3, "Supportive    $D > \\tau_D$", posn, n_nn, SUPPORT, rule=True),
        row(4, "$S^+_{hi}$   Distributed support    $R \\geq \\tau_R$",
            n_sphi, n_pos, LIGHT_BLUE),
        row(5, "$S^+_{inv}$   Robust support    $R \\geq \\tau_R,\\; H \\geq \\tau_H$",
            b["keep_robust_support"].n_pairs, n_sphi, LIGHT_BLUE),
        row(4, "$S^+_{lo}$   Concentrated support    $R < \\tau_R$",
            b["S+_lo"].n_pairs, n_pos, LIGHT_BLUE),
        row(3, "Harmful    $D < -\\tau_D$", negn, n_nn, HARM, rule=True),
        row(4, "$S^-_{hi}$   Distributed harm    $R \\geq \\tau_R$",
            n_shhi, n_neg, LIGHT_ORANGE),
        row(5, "$S^-_{inv}$   Harmful invariant    $R \\geq \\tau_R,\\; H \\geq \\tau_H$",
            b["harmful_invariant"].n_pairs, n_shhi, LIGHT_ORANGE),
        row(4, "$S^-_{lo}$   Concentrated harm    $R < \\tau_R$",
            b["S-_lo"].n_pairs, n_neg, LIGHT_ORANGE),
    ]

    TIER_INK = {1: INK, 2: INK_SOFT, 3: INK, 4: INK_SOFT, 5: INK_SOFT}

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.5, 2.5), wspace=0.04,
                          left=0.015, right=0.975, top=0.855, bottom=0.135)
    axl = fig.add_subplot(gs[0, 0])
    axn = fig.add_subplot(gs[0, 1])

    y = np.arange(len(rows))[::-1]     # first row at the top

    axl.set_xlim(0, 1)
    axl.axis("off")
    for yi, r in zip(y, rows):
        axl.text(0.0, yi, r["label"], va="center", ha="left", fontsize=8.4,
                 color=TIER_INK[r["tier"]],
                 weight="bold" if r["tier"] in (1, 3) else "normal")

    # Counts on a log axis. The populations span 32 to 114,688, so log is the
    # only scale on which every row is visible at once.
    axn.barh(y, [r["n"] for r in rows], left=0.7, height=0.60,
             color=[r["color"] for r in rows], lw=0, zorder=3)
    axn.set_xscale("log")
    axn.set_xlim(0.7, total * 12)
    axn.set_xticks([1, 10, 100, 1000, 10000, 100000])
    for yi, r in zip(y, rows):
        axn.annotate(f"{r['n']:,}", xy=(0.7 + r["n"], yi), xytext=(4, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=7.6, color=INK)
    axn.set_xlabel("class–concept pairs   (log scale)", fontsize=8.5)
    axn.set_yticks([])
    axn.grid(axis="x", zorder=0)
    axn.tick_params(axis="x", labelsize=7.5)
    axn.spines["left"].set_visible(False)

    # hairline above each new block, spanning both panels
    for yi, r in zip(y, rows):
        if not r["rule"]:
            continue
        for ax in (axl, axn):
            ax.axhline(yi + 0.5, color=GRID, lw=0.8, zorder=1)

    for ax in (axl, axn):
        ax.set_ylim(-0.65, len(rows) - 0.35)

    fig.suptitle("Class–Concept Grid", fontsize=11.5, color=INK,
                 weight="bold", y=0.955)

    p = out / "fig2_funnel.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# P2-3  H vs D
# ---------------------------------------------------------------------------


def fig_h_vs_d(s, tau_d, tau_h, floor, out: Path):
    sup = s.support(floor)
    D, H = s.D[sup], s.H[sup]

    regime = np.where(D > tau_d, "supportive", np.where(D < -tau_d, "harmful", "neutral"))

    # Marginal histograms were tried and cut. Section 5.2 already states both
    # claims numerically -- 1269 of 1542 pairs neutral, 1173 of 1542 above tau_H
    # -- so the marginals restated in pictures what the text gives in numbers,
    # and cost a third of the figure's area to do it.
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    fig.subplots_adjust(left=0.095, right=0.935, top=0.80, bottom=0.145)

    # the neutral band is the claim "the neutral mass dominates", so shade it
    ax.axvspan(-tau_d, tau_d, color=NEUTRAL, alpha=0.13, lw=0, zorder=0)
    ax.axhline(tau_h, color=INK_SOFT, lw=0.8, ls=(0, (4, 3)), zorder=1)

    # Plot neutral FIRST so the two signed regimes are never buried under it,
    # but key them left-to-right in x-order (harmful at negative D, neutral at
    # the origin, supportive at positive) so each entry sits above the region it
    # names. Draw order and legend order are decoupled deliberately.
    handles = {}
    for name in ("neutral", "supportive", "harmful"):
        m = regime == name
        handles[name] = ax.scatter(
            D[m], H[m], s=9, c=REGIME_COLOR[name], alpha=0.55, linewidths=0,
            label=f"{name}  ({int(m.sum()):,})",
            zorder=2 if name == "neutral" else 3)
    key_order = ("harmful", "neutral", "supportive")

    ax.set_xscale("symlog", linthresh=tau_d, linscale=0.6)
    ax.set_xlabel("Discriminative Effect  $D(k,c)$      "
                  "(symlog, linear within $\\pm\\tau_D$)")
    ax.set_ylabel("Activation Invariance  $H(k,c)$")
    ax.set_ylim(-0.02, 1.06)
    ax.grid(axis="y", zorder=0)

    # Thresholds annotated clear of the tick labels: tau_H outside the right
    # spine on its own line, tau_D inside the shaded band at the foot of the
    # plot, where the point cloud is sparsest. Both previously sat on top of the
    # axis labels.
    ax.annotate(f"$\\tau_H$ = {tau_h}", xy=(1.0, tau_h),
                xycoords=("axes fraction", "data"), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7.8, color=INK_SOFT, annotation_clip=False)
    ax.annotate(f"$|D| \\leq \\tau_D$", xy=(0, 0.012),
                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                fontsize=7.4, color=INK_SOFT)

    # regime key across the top, above the plot area rather than inside it
    ax.legend([handles[n] for n in key_order], [handles[n].get_label() for n in key_order],
              loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), mode="expand",
              ncol=3, markerscale=1.9, handletextpad=0.35, borderaxespad=0,
              columnspacing=1.2)

    fig.suptitle("Activation Invariance vs Discriminative Effect",
                 fontsize=11.5, color=INK, weight="bold", y=0.965)

    p = out / "fig3_h_vs_d.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# P2-4  H x R heatmap + tau sensitivity companion
# ---------------------------------------------------------------------------


def _ramp(hue):
    """Light-to-hue sequential ramp. One hue, monotone in lightness."""
    return LinearSegmentedColormap.from_list("r", ["#f7f8f9", hue])


def fig_hr_heatmap(s, tau_d, taus, floor, out: Path):
    sup = s.support(floor)
    tau = taus[0]
    # Harmful and supportive are ADJACENT: the 23.2% vs 56.3% comparison is the
    # point of the figure, and the earlier ordering forced the eye across the
    # neutral panel to make it. Neutral is last, where it belongs -- it exists
    # only to motivate the magnitude gate.
    regimes = [
        ("Harmful", s.D < -tau_d, HARM),
        ("Supportive", s.D > tau_d, SUPPORT),
        ("Neutral", np.abs(s.D) <= tau_d, NEUTRAL),
    ]

    fig = plt.figure(figsize=(6.6, 2.85))
    gs = fig.add_gridspec(1, 3, wspace=0.22,
                          left=0.095, right=0.985, bottom=0.14, top=0.78)

    hiH, hiR = s.H >= tau, s.R >= tau
    for j, (name, rsel, hue) in enumerate(regimes):
        axh = fig.add_subplot(gs[0, j])
        sel = sup & rsel
        tot = int(sel.sum())
        M = np.zeros((2, 2))
        for a, hm in enumerate((hiH, ~hiH)):          # row 0 = high H (top)
            for b, rm in enumerate((~hiR, hiR)):      # col 1 = high R (right)
                M[a, b] = int((sel & hm & rm).sum())
        # Each panel is shaded in its OWN regime hue. A single blue ramp painted
        # the harmful panel in the colour that means "supportive" everywhere else
        # in the paper. Cells encode share-of-own-regime on a common 0-60% scale,
        # so shading is comparable across panels.
        axh.imshow(M / max(tot, 1), cmap=_ramp(hue), vmin=0, vmax=0.6)
        for a in range(2):
            for b in range(2):
                v = M[a, b] / max(tot, 1)
                axh.text(b, a, f"{int(M[a, b])}\n{v * 100:.1f}%", ha="center",
                         va="center", fontsize=8.2, linespacing=1.3,
                         color=SURFACE if v > 0.40 else INK)
        axh.set_xticks([0, 1], [f"R < {tau}", f"R ≥ {tau}"], fontsize=8)
        # y labels only on the leftmost panel; three copies was pure clutter
        if j == 0:
            axh.set_yticks([0, 1], [f"H ≥ {tau}", f"H < {tau}"], fontsize=8)
        else:
            axh.set_yticks([])
        # No outline or in-cell callout on the harmful high-H/high-R cell. It is
        # the paper's headline bucket, but the abstract, Section 5.4 and Table 3
        # all name it in words; annotating it here forced extra title padding and
        # bought nothing the text does not already say.
        axh.set_title(f"{name}  ({tot})", fontsize=9.5, color=hue,
                      weight="bold", pad=8)
        for sp in axh.spines.values():
            sp.set_visible(False)
        axh.tick_params(length=0)

    fig.suptitle("Invariance Against Consistency, by Discriminative Regime",
                 fontsize=11, color=INK, weight="bold", y=0.965)

    p = out / "fig7_hr_heatmap.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# threshold sensitivity, its own figure
# ---------------------------------------------------------------------------


def fig_tau_sensitivity(s, tau_d, tau_chosen, floor, out: Path,
                        lo=0.50, hi=0.95, step=0.025):
    """
    The asymmetry of Section 5.3 swept over tau, on its own rather than bolted to
    the heatmap as a fourth panel of a different chart type.

    Swept finely rather than at {0.7, 0.8, 0.9}. This costs nothing -- it is a
    read-only pass over the score file, no re-scoring and no ablation -- and it
    turns a three-point claim into a curve, which is what makes "the conclusion
    does not depend on the threshold" checkable rather than assertable.

    The log2/log3 = 0.63 bound is drawn, because it is the only principled
    landmark on this axis: above it, clearing tau guarantees nonzero activation
    (or effect) in all three source domains, since an even split over just two
    domains scores 0.63. Below it that guarantee is gone and the buckets change
    meaning, so the curve is drawn but greyed there.
    """
    sup = s.support(floor)
    taus = np.round(np.arange(lo, hi + 1e-9, step), 4)
    bound = np.log(2) / np.log(3)
    MIN_DENOM = 20          # below this the harmful fraction is a few pairs

    series = {"Supportive": [], "Harmful": []}
    counts = {"Supportive": [], "Harmful": []}
    for t in taus:
        for name, rsel in (("Supportive", s.D > tau_d), ("Harmful", s.D < -tau_d)):
            grp = sup & rsel & (s.H >= t)
            n = int(grp.sum())
            counts[name].append(n)
            series[name].append(int((grp & (s.R >= t)).sum()) / max(n, 1) * 100)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.145)

    # the region where the three-domain guarantee does not hold
    ax.axvspan(taus[0] - step, bound, color=NEUTRAL, alpha=0.11, lw=0, zorder=0)
    ax.axvline(bound, color=INK_SOFT, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.annotate("$\\log 2/\\log 3 = 0.63$\nbelow this, clearing $\\tau$ no longer\n"
                "implies all three domains active",
                xy=(bound, 3), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.0, color=INK_SOFT,
                linespacing=1.35)

    ax.axvline(tau_chosen, color=INK, lw=0.9, zorder=2)
    ax.annotate(f"$\\tau$ = {tau_chosen}\nchosen", xy=(tau_chosen, 96),
                xytext=(5, 0), textcoords="offset points", ha="left", va="top",
                fontsize=7.4, color=INK, weight="bold", linespacing=1.4)

    # Mark where the harmful denominator falls into single digits. Past that the
    # ratio is a handful of pairs and turns over -- at tau = 0.95 it drops from
    # 3.2 to 2.9 -- which would otherwise read as a real reversal of the trend
    # rather than as small-sample noise.
    thin = [i for i, c in enumerate(counts["Harmful"]) if c < MIN_DENOM]
    if thin:
        x_thin = taus[thin[0]]
        ax.axvspan(x_thin, taus[-1], color=NEUTRAL, alpha=0.11, lw=0, zorder=0)
        ax.annotate(f"fewer than {MIN_DENOM} harmful\npairs remain; ratio unstable",
                    xy=(x_thin, 62), xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.0, color=INK_SOFT,
                    linespacing=1.35)

    for name in ("Supportive", "Harmful"):
        ax.plot(taus, series[name], "-", color=REGIME_COLOR[name.lower()],
                lw=2.0, zorder=3)
        i = int(np.argmin(np.abs(taus - 0.86)))
        ax.annotate(name, xy=(taus[i], series[name][i]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=REGIME_COLOR[name.lower()], weight="bold")

    ax.set_xlim(taus[0], taus[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("$\\tau_H = \\tau_R$")
    ax.set_ylabel("Invariant pairs whose effect is also consistent (%)")
    ax.grid(axis="y", zorder=0)
    # Scoped deliberately: this sweeps tau only. The support floor is the other
    # free parameter and it does move the magnitude, so a title claiming "not a
    # threshold artifact" would over-claim. Section 4.4 carries the floor sweep.
    fig.suptitle("The Support–Harm Asymmetry Persists Across Invariance Thresholds",
                 fontsize=10.5, color=INK, weight="bold", y=0.955)

    p = out / "fig8_tau_sensitivity.png"
    fig.savefig(p)
    plt.close(fig)

    lohi = [(t, series["Supportive"][i], series["Harmful"][i],
             series["Supportive"][i] / max(series["Harmful"][i], 1e-9))
            for i, t in enumerate(taus)]
    print("    tau   supportive%   harmful%   ratio   n_supp  n_harm")
    for i, (t, sv, hv, r) in enumerate(lohi):
        if abs(t * 100 - round(t * 100)) < 1e-6 and round(t * 100) % 5 == 0:
            flag = "  <- thin" if counts["Harmful"][i] < MIN_DENOM else ""
            print(f"   {t:.3f}   {sv:9.1f}   {hv:8.1f}   {r:5.2f}   "
                  f"{counts['Supportive'][i]:6d}  {counts['Harmful'][i]:6d}{flag}")
    return p


# ---------------------------------------------------------------------------
# P2-5  interventions against matched controls
# ---------------------------------------------------------------------------


LABELS = {
    "all_harmful": "all harmful (138)",
    "S-_hi": "$S^-_{hi}$ distributed harm (35)",
    "harmful_invariant": "$S^-_{inv}$ harmful invariant (32)",
    "S-_lo": "$S^-_{lo}$ concentrated harm (103)",
    "inert_only": "inert only (111,530)",
    "S+_lo": "$S^+_{lo}$ concentrated support (54)",
    "S+_hi": "$S^+_{hi}$ distributed support (81)",
    "all_supportive": "all supportive (135)",
}


def fig_interventions(summary_path: str, out: Path):
    df = pd.read_csv(summary_path)
    df = df[~df["smoke"].astype(str).str.lower().isin(["true", "1"])]

    tgt, ctrl = {}, {}
    for _, r in df.iterrows():
        eid = str(r["experiment_id"])
        if eid.startswith("B2_"):
            tgt[eid[3:]] = r["sketch_delta_macro"]
        elif eid.startswith("B4_"):
            tgt[eid[3:]] = r["sketch_delta_macro"]
        elif eid.startswith("B3_ctrl_"):
            ctrl.setdefault(eid[len("B3_ctrl_"):].rsplit("_seed", 1)[0], []).append(
                r["sketch_delta_macro"])

    rows = [(k, float(tgt[k]), float(np.mean(ctrl[k])) if k in ctrl else None)
            for k in LABELS if k in tgt and pd.notna(tgt[k])]
    rows.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.subplots_adjust(top=0.82, left=0.30, right=0.975, bottom=0.125)
    h = 0.33

    lo = min([d for _, d, _ in rows] + [c for *_, c in rows if c is not None])
    hi = max([d for _, d, _ in rows] + [c for *_, c in rows if c is not None])
    pad = (hi - lo) * 0.13
    ax.set_xlim(lo - pad * 2.1, hi + pad * 1.5)

    for i, (k, d, c) in enumerate(rows):
        col = HARM if d > 0 else SUPPORT
        # alternating row band: makes it unambiguous which ghost belongs to
        # which target without adding a fourth colour
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f5f6", lw=0, zorder=0)
        ax.barh(i + h / 2 + 0.03, d, height=h, color=col, lw=0, zorder=3)
        ax.annotate(f"{d:+.2f}", xy=(d, i + h / 2 + 0.03),
                    xytext=(5 if d >= 0 else -5, 0), textcoords="offset points",
                    va="center", ha="left" if d >= 0 else "right",
                    fontsize=8, color=INK, weight="bold")
        if c is None:
            ax.annotate("no matched control", xy=(0, i - h / 2 - 0.03),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=6.9, color=INK_SOFT, style="italic")
            continue
        # control: same hue, lighter + hatched. A fourth categorical hue would
        # fail CVD separation against the blue, so texture carries it.
        ax.barh(i - h / 2 - 0.03, c, height=h, color=col, alpha=0.20, lw=0.8,
                edgecolor=col, hatch="////", zorder=3)
        ax.annotate(f"{c:+.2f}", xy=(c, i - h / 2 - 0.03),
                    xytext=(5 if c >= 0 else -5, 0), textcoords="offset points",
                    va="center", ha="left" if c >= 0 else "right",
                    fontsize=7.4, color=INK_SOFT)

    ax.axvline(0, color=INK_SOFT, lw=0.9, zorder=2)
    ax.set_yticks(np.arange(len(rows)), [LABELS[k] for k, _, _ in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δ sketch macro accuracy vs the SAE-reconstruction baseline (points)")
    ax.grid(axis="x", zorder=0)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.spines["left"].set_visible(False)

    # Legend ABOVE the axes. Inside, it sat on top of the two largest bars.
    ax.legend(handles=[Patch(fc=INK_SOFT, lw=0, label="Diagnostic bucket"),
                       Patch(fc=INK_SOFT, alpha=0.20, ec=INK_SOFT, hatch="////",
                             label="Size- and |D|-matched random control")],
              loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), mode="expand",
              ncol=2, handlelength=1.7, borderaxespad=0, columnspacing=6.0,
              handletextpad=0.7)

    fig.suptitle("Target-Domain Accuracy After Masking Each Diagnostic Bucket",
                 fontsize=11, color=INK, weight="bold", y=0.972)

    p = out / "fig10_interventions.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_style()

    want = set(args.only) if args.only else {
        "funnel", "scatter", "heatmap", "sensitivity", "interventions"}

    manifest_cfg = {
        "scores": args.scores, "summary": args.summary, "out_dir": str(out),
        "tau_d": args.tau_d, "taus": args.taus, "support_floor": args.support_floor,
        "only": sorted(want),
    }

    with RunRecorder("make_figures", config=manifest_cfg) as rec:
        tau = args.taus[0]
        s = buckets = None
        # Which figures need what. Keep these two sets beside the dispatch below;
        # they drifted apart once the sensitivity figure was added, and --only
        # sensitivity then ran with s = None.
        NEEDS_SCORES = {"funnel", "scatter", "heatmap", "sensitivity"}
        NEEDS_BUCKETS = {"funnel"}
        if want & NEEDS_SCORES:
            s = load_scores(args.scores, class_names=list(cfg.pacs.class_names))
            print(f"[scores] {s.n_classes} x {s.n_concepts}, "
                  f"{s.n_domains} source domains")
        if want & NEEDS_BUCKETS:
            buckets = build_buckets(s, tau_d=args.tau_d, tau_h=tau, tau_r=tau,
                                    support_floor=args.support_floor)

        made = []
        if "funnel" in want:
            with rec.step("fig2_funnel"):
                made.append(fig_funnel(s, buckets, args.tau_d, args.support_floor, out))
        if "scatter" in want:
            with rec.step("fig3_h_vs_d"):
                made.append(fig_h_vs_d(s, args.tau_d, tau, args.support_floor, out))
        if "heatmap" in want:
            with rec.step("fig7_hr_heatmap"):
                made.append(fig_hr_heatmap(s, args.tau_d, args.taus,
                                           args.support_floor, out))
        if "sensitivity" in want:
            with rec.step("fig8_tau_sensitivity"):
                made.append(fig_tau_sensitivity(s, args.tau_d, tau,
                                                args.support_floor, out))
        if "interventions" in want:
            with rec.step("fig10_interventions"):
                made.append(fig_interventions(args.summary, out))

        for p in made:
            rec.add_output(p)
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
