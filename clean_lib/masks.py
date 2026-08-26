"""
Score loading and mask construction.

MASK POLARITY, stated once: every function here returns a boolean
(classes, concepts) tensor where **True means "mask this (class, concept) pair
out"**. This is the polarity `MaskedAccuracyEvaluator` expects and the opposite
of `Analyzer.filter`, which expresses "keep".

SUPPORT FLOOR. Bucket definitions are restricted to pairs with
`H_counts >= support_floor`. This is not the label-leaking count filter warned
about in CLAUDE.md §3: that pathology was *masking* thin-support pairs, which
deletes competing-class evidence and buys ~2.7pp of pure leakage. Here the floor
does the opposite — thin-support pairs are *excluded from the mask*, i.e. left
intact — because R is an entropy over three per-domain estimates and a cell with
four images gives a meaningless D_d. Every bucket is reported with its pair count
so the restriction is visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


# ---------------------------------------------------------------------------
# score file
# ---------------------------------------------------------------------------


@dataclass
class Scores:
    """Score arrays extracted from a score JSON. Shapes (K, C) except R_acts."""

    H: np.ndarray            # (K, C)
    D: np.ndarray            # (K, C)
    R: np.ndarray            # (K, C)
    H_counts: np.ndarray     # (K, C)   active-image count for that class
    R_acts: np.ndarray       # (K, C, n_domains)  signed per-domain D_d
    R_counts: np.ndarray     # (K, C, n_domains)
    class_names: List[str]

    @property
    def n_classes(self) -> int:
        return self.H.shape[0]

    @property
    def n_concepts(self) -> int:
        return self.H.shape[1]

    @property
    def n_domains(self) -> int:
        return self.R_acts.shape[2]

    def support(self, floor: int = 30) -> np.ndarray:
        """(K, C) bool: pairs with enough active images to trust D_d and R."""
        return self.H_counts >= floor


def load_scores(path: str | Path, class_names: Optional[List[str]] = None) -> Scores:
    path = Path(path)
    # Bytes then decode: text-mode reads of these files have produced spurious
    # JSONDecodeErrors on this machine.
    with open(path, "rb") as fh:
        data = json.loads(fh.read().decode("utf-8"))

    class_keys = sorted(data.keys(), key=int)
    concept_keys = sorted(data[class_keys[0]].keys(), key=int)
    K, C = len(class_keys), len(concept_keys)

    probe = data[class_keys[0]][concept_keys[0]]
    n_dom = len(probe["R_acts"])

    H = np.zeros((K, C), dtype=np.float64)
    D = np.zeros((K, C), dtype=np.float64)
    R = np.zeros((K, C), dtype=np.float64)
    Hc = np.zeros((K, C), dtype=np.float64)
    Ra = np.zeros((K, C, n_dom), dtype=np.float64)
    Rc = np.zeros((K, C, n_dom), dtype=np.float64)

    for ki, ck in enumerate(class_keys):
        cls = data[ck]
        for ci, cc in enumerate(concept_keys):
            v = cls[cc]
            H[ki, ci] = v["H"]
            D[ki, ci] = v["D"]
            R[ki, ci] = v["R"]
            Hc[ki, ci] = v["H_counts"]
            Ra[ki, ci, :] = v["R_acts"]
            Rc[ki, ci, :] = v["R_counts"]

    return Scores(
        H=H, D=D, R=R, H_counts=Hc, R_acts=Ra, R_counts=Rc,
        class_names=list(class_names or CLASS_NAMES),
    )


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    """A named set of (class, concept) pairs plus the mask that ablates it."""

    name: str
    definition: str
    select: np.ndarray                # (K, C) bool: pairs IN the bucket
    mask: torch.Tensor                # (K, C) bool: True = mask out
    n_pairs: int                      # size of the bucket itself
    per_class_pairs: Dict[str, int]
    d_mass: float                     # sum |D| over the bucket
    per_class_d_mass: Dict[str, float]
    # What the mask ACTUALLY ablates. Identical to the above for ordinary
    # buckets, but for keep-only buckets the mask is the complement, so these
    # are the numbers the paper must report.
    n_masked: int = 0
    per_class_masked: Dict[str, int] = field(default_factory=dict)
    masked_d_mass: float = 0.0
    per_class_masked_d_mass: Dict[str, float] = field(default_factory=dict)
    keep_only: bool = False
    oracle: bool = True

    def summary(self) -> str:
        if self.keep_only:
            return (f"{self.name:<26} keep={self.n_pairs:<6} "
                    f"masked={self.n_masked:<7} "
                    f"sum|D|masked={self.masked_d_mass:.6g}   [{self.definition}]")
        return (f"{self.name:<26} pairs={self.n_pairs:<6} "
                f"sum|D|={self.d_mass:.6g}   [{self.definition}]")


def _bucket(
    name: str,
    definition: str,
    select: np.ndarray,
    scores: Scores,
    invert: bool = False,
) -> Bucket:
    """Wrap a selection into a Bucket. invert=True masks the complement (keep-only)."""
    mask_np = (~select) if invert else select
    mask = torch.from_numpy(mask_np.astype(bool))

    absd = np.abs(scores.D)
    per_class_pairs = {
        scores.class_names[k]: int(select[k].sum()) for k in range(scores.n_classes)
    }
    per_class_mass = {
        scores.class_names[k]: float(absd[k][select[k]].sum())
        for k in range(scores.n_classes)
    }
    return Bucket(
        name=name,
        definition=definition,
        select=select,
        mask=mask,
        n_pairs=int(select.sum()),
        per_class_pairs=per_class_pairs,
        d_mass=float(absd[select].sum()),
        per_class_d_mass=per_class_mass,
        n_masked=int(mask_np.sum()),
        per_class_masked={
            scores.class_names[k]: int(mask_np[k].sum())
            for k in range(scores.n_classes)
        },
        masked_d_mass=float(absd[mask_np].sum()),
        per_class_masked_d_mass={
            scores.class_names[k]: float(absd[k][mask_np[k]].sum())
            for k in range(scores.n_classes)
        },
        keep_only=invert,
    )


def build_buckets(
    scores: Scores,
    tau_d: float = 1e-4,
    tau_h: float = 0.7,
    tau_r: float = 0.7,
    support_floor: int = 30,
) -> Dict[str, Bucket]:
    """The paper's typology sets, plus the aggregate and keep-only variants."""
    sup = scores.support(support_floor)
    D, H, R = scores.D, scores.H, scores.R

    pos = sup & (D > tau_d)
    neg = sup & (D < -tau_d)
    hiR = R >= tau_r
    hiH = H >= tau_h

    out: Dict[str, Bucket] = {}

    def add(name, definition, select, invert=False):
        out[name] = _bucket(name, definition, select, scores, invert=invert)

    add("S+_lo", f"D > {tau_d:g} and R < {tau_r}", pos & ~hiR)
    add("S+_hi", f"D > {tau_d:g} and R >= {tau_r}", pos & hiR)
    add("S-_lo", f"D < -{tau_d:g} and R < {tau_r}", neg & ~hiR)
    add("S-_hi", f"D < -{tau_d:g} and R >= {tau_r}", neg & hiR)

    add("all_harmful", f"D < -{tau_d:g}", neg)
    add("all_supportive", f"D > {tau_d:g}", pos)

    # The paper's central category: invariant, consistent, and harmful.
    add("harmful_invariant", f"H >= {tau_h} and R >= {tau_r} and D < -{tau_d:g}",
        neg & hiH & hiR)

    # Keep-only variants. These are far leakier oracles than the mask-only ones:
    # retaining *only* concepts that support the true class deletes all evidence
    # for competing classes, which is close to injecting the label. They are
    # therefore reported as an upper bound, and only ever interpreted against
    # keep_only_random_control (which measures the injection component) and
    # against each other (which isolates the R axis, since the injection
    # component is common to all three).
    add("keep_robust_support",
        f"KEEP ONLY H >= {tau_h} and R >= {tau_r} and D > {tau_d:g}",
        pos & hiH & hiR, invert=True)
    add("keep_S+_lo",
        f"KEEP ONLY D > {tau_d:g} and R < {tau_r}", pos & ~hiR, invert=True)
    add("keep_all_supportive",
        f"KEEP ONLY D > {tau_d:g}", pos, invert=True)

    # Denoising control: pairs whose measured effect is exactly zero in every
    # source domain (or which never fire for that class). Not support-floored --
    # the point is to characterise the inert majority.
    add("inert_only", "R == 0 (no measured effect in any source domain)",
        scores.R == 0.0)

    # The pairs the support floor excludes that could plausibly matter: thin
    # support, but a gated effect and a nonzero R. 82% of all below-floor |D|
    # mass sits here, at a mean per-pair magnitude comparable to the above-floor
    # non-neutral population -- so they cannot be dismissed on magnitude.
    #
    # They are also a winner's-curse selection: choosing thin pairs BY their
    # |D_hat| clearing a threshold preferentially picks the ones noise pushed
    # up. Ablating them is the direct test of whether the floor at 30 is
    # discarding signal or discarding noise, and it settles empirically what no
    # argument about aggregate mass can. Not part of the typology; a diagnostic.
    below_floor = (~sup) & (scores.R != 0.0)
    add("below_floor_gated",
        f"n(k,c) < {support_floor} and R != 0 and |D| > {tau_d:g}",
        below_floor & (np.abs(D) > tau_d))

    # Split by sign, because the combined bucket masks supportive and harmful
    # pairs together and they move accuracy in opposite directions -- the net is
    # a lower bound on both. Only the sign-split rows are comparable to the
    # typology buckets in Table 3, which are all single-signed.
    add("below_floor_harmful",
        f"n(k,c) < {support_floor} and R != 0 and D < -{tau_d:g}",
        below_floor & (D < -tau_d))
    add("below_floor_supportive",
        f"n(k,c) < {support_floor} and R != 0 and D > {tau_d:g}",
        below_floor & (D > tau_d))

    return out


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def keep_only_random_control(
    target: Bucket,
    scores: Scores,
    seed: int,
    support_floor: int = 30,
) -> Bucket:
    """
    Keep the same NUMBER of pairs per class as a keep-only target, but chosen
    uniformly at random instead of because they support the true class.

    This isolates the label-injection component of a keep-only oracle. Retaining
    only concepts selected for supporting class y, on images whose true label is
    y, deletes every competing-class direction from the reconstruction — so a
    high accuracy could reflect the label used to build the mask rather than
    anything about the bucket. If keeping an arbitrary set of the same size
    reaches comparable accuracy, the keep-only row is circular and must be cut.

    Uniform, not |D|-stratified, on purpose: stratifying would preferentially
    retain high-|D| pairs, which are mostly supportive-for-some-class, and would
    reintroduce the very effect being controlled for.

    Candidates exclude the target's own pairs.
    """
    rng = np.random.default_rng(seed)
    sup = scores.support(support_floor)
    select = np.zeros_like(target.select, dtype=bool)

    for k in range(scores.n_classes):
        need = int(target.select[k].sum())
        if need == 0:
            continue
        cand = np.flatnonzero(sup[k] & ~target.select[k])
        if cand.size == 0:
            continue
        take = min(need, cand.size)
        select[k, rng.choice(cand, size=take, replace=False)] = True

    return _bucket(
        f"keep_random_matched_to_{target.name}",
        f"KEEP ONLY a uniform random per-class size-match to {target.name}",
        select,
        scores,
        invert=True,
    )


def stratified_random_control(
    target: Bucket,
    scores: Scores,
    seed: int,
    support_floor: int = 30,
    n_bins: int = 10,
) -> Bucket:
    """
    A size-matched, |D|-distribution-matched random bucket drawn from the pairs
    the target did NOT select.

    Why stratified: uniform random pairs skew neutral (1269 of 1542 support-floored
    pairs are neutral), so a uniform control is trivially easy to beat and proves
    nothing. Matching the target's |D| histogram per class is what makes the
    comparison informative.

    Candidates exclude the target's own pairs, so the control is a genuinely
    different set of the same size and effect-magnitude profile.
    """
    rng = np.random.default_rng(seed)
    sup = scores.support(support_floor)
    absd = np.abs(scores.D)

    select = np.zeros_like(target.select, dtype=bool)

    for k in range(scores.n_classes):
        tgt_idx = np.flatnonzero(target.select[k])
        if tgt_idx.size == 0:
            continue

        cand_idx = np.flatnonzero(sup[k] & ~target.select[k])
        if cand_idx.size == 0:
            continue

        want = min(tgt_idx.size, cand_idx.size)
        tgt_absd = absd[k, tgt_idx]
        cand_absd = absd[k, cand_idx]

        # Bin edges from the target's own |D| quantiles.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(tgt_absd, qs))
        if edges.size < 2:
            edges = np.array([tgt_absd.min(), tgt_absd.max() + 1e-30])

        tgt_bin = np.clip(np.digitize(tgt_absd, edges[1:-1]), 0, len(edges) - 2)
        cand_bin = np.clip(np.digitize(cand_absd, edges[1:-1]), 0, len(edges) - 2)

        chosen: List[int] = []
        used = np.zeros(cand_idx.size, dtype=bool)

        for b in range(len(edges) - 1):
            need = int((tgt_bin == b).sum())
            if need == 0:
                continue
            pool = np.flatnonzero((cand_bin == b) & ~used)
            take = min(need, pool.size)
            if take:
                picked = rng.choice(pool, size=take, replace=False)
                used[picked] = True
                chosen.extend(cand_idx[picked].tolist())

        # Top up any shortfall with the closest available |D|, so the size match
        # is exact even when a bin is underpopulated.
        shortfall = want - len(chosen)
        if shortfall > 0:
            remaining = np.flatnonzero(~used)
            if remaining.size:
                target_median = float(np.median(tgt_absd))
                order = remaining[np.argsort(np.abs(cand_absd[remaining] - target_median))]
                picked = order[:shortfall]
                used[picked] = True
                chosen.extend(cand_idx[picked].tolist())

        select[k, np.array(chosen, dtype=int)] = True

    return _bucket(
        name=f"random_ctrl[{target.name}]seed{seed}",
        definition=(f"size- and |D|-stratified random control matched to "
                    f"{target.name} (seed {seed})"),
        select=select,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# conflict, computed from D alone
# ---------------------------------------------------------------------------


def conflict_analysis(
    scores: Scores, tau_d: float = 1e-4, support_floor: int = 30
) -> dict:
    """
    A concept is class-conflicting if it supports at least one class and harms
    at least one other:

        max_k D(k,c) >  tau_d   and   min_k D(k,c) < -tau_d

    Computed entirely from the existing D matrix -- no extra forward passes.
    Co-activity comes for free: D(k,c) is only accumulated over images of class k
    on which c actually fires, so a conflict cannot arise from two disjoint
    populations.

    contrast(c) = max_k D - min_k D  ranks how conflicted a concept is.
    """
    sup = scores.support(support_floor)
    D = np.where(sup, scores.D, np.nan)  # ignore thin-support cells

    # Reduce only over concepts that have at least one class above the floor.
    # Calling nanmax on an all-NaN column is legal but emits a RuntimeWarning,
    # and ~15.8k of 16.4k concepts have no class above the floor.
    valid = sup.any(axis=0)
    dmax = np.full(scores.n_concepts, np.nan)
    dmin = np.full(scores.n_concepts, np.nan)
    if valid.any():
        dmax[valid] = np.nanmax(D[:, valid], axis=0)
        dmin[valid] = np.nanmin(D[:, valid], axis=0)
    conflicting = valid & (dmax > tau_d) & (dmin < -tau_d)

    supportive_any = valid & (dmax > tau_d)
    harmful_any = valid & (dmin < -tau_d)

    contrast = np.where(valid, dmax - dmin, np.nan)

    # Per class: how many conflicting concepts touch this class, and in which role.
    per_class = {}
    for k in range(scores.n_classes):
        row = np.where(sup[k], scores.D[k], np.nan)
        per_class[scores.class_names[k]] = {
            "supports_and_concept_conflicts": int(
                np.nansum((row > tau_d) & conflicting)
            ),
            "harmed_by_conflicting": int(np.nansum((row < -tau_d) & conflicting)),
            "conflict_mass": float(
                np.nansum(np.abs(row[conflicting])) if conflicting.any() else 0.0
            ),
        }

    order = np.argsort(np.where(np.isnan(contrast), -np.inf, contrast))[::-1]
    top = [
        {
            "concept": int(c),
            "contrast": float(contrast[c]),
            "argmax_class": scores.class_names[int(np.nanargmax(D[:, c]))],
            "argmin_class": scores.class_names[int(np.nanargmin(D[:, c]))],
        }
        for c in order[:25]
        if conflicting[c]
    ]

    return {
        "tau_d": tau_d,
        "support_floor": support_floor,
        "n_concepts_considered": int(valid.sum()),
        "n_supportive_any": int(supportive_any.sum()),
        "n_harmful_any": int(harmful_any.sum()),
        "n_conflicting": int(conflicting.sum()),
        "pct_conflicting_of_nonneutral": (
            float(conflicting.sum()) / float((supportive_any | harmful_any).sum()) * 100
            if (supportive_any | harmful_any).any() else 0.0
        ),
        "per_class": per_class,
        "top_conflicting": top,
    }


# ---------------------------------------------------------------------------
# domain attribution of concentrated effect
# ---------------------------------------------------------------------------


def domain_attribution(
    scores: Scores,
    domain_names: List[str],
    tau_d: float = 1e-4,
    tau_r: float = 0.7,
    support_floor: int = 30,
) -> dict:
    """
    For every low-R non-neutral pair, which source domain carries the effect
    (argmax_d |D_d|), cross-tabulated by sign of D and by class.

    Needs no masking and no GPU: R_acts already holds the per-domain effects.
    """
    sup = scores.support(support_floor)
    absd_dom = np.abs(scores.R_acts)          # (K, C, n_dom)
    peak = absd_dom.argmax(axis=2)            # (K, C)

    nonneutral = sup & (np.abs(scores.D) > tau_d)
    low_r = sup & (scores.R < tau_r)
    high_r = sup & (scores.R >= tau_r)
    pos = low_r & (scores.D > tau_d)
    neg = low_r & (scores.D < -tau_d)

    def tabulate(sel: np.ndarray) -> Dict[str, int]:
        return {
            domain_names[d]: int(((peak == d) & sel).sum())
            for d in range(scores.n_domains)
        }

    per_class = {}
    for k in range(scores.n_classes):
        m = np.zeros_like(pos)
        m[k] = True
        per_class[scores.class_names[k]] = {
            "supportive": tabulate(pos & m),
            "harmful": tabulate(neg & m),
        }

    # ---- confound controls -------------------------------------------------
    # A raw argmax table cannot distinguish "this concept's effect concentrates
    # in cartoon" from "every concept's effect is larger in cartoon". Three
    # references make that separable:
    #
    #  reference_high_r : argmax for HIGH-R pairs. These are by definition NOT
    #                     concentrated, so their argmax distribution is the null.
    #                     If low-R skews toward a domain more than high-R does,
    #                     the concentration claim survives.
    #  mean_abs_d_by_domain : is |D_d| systematically larger in some domain?
    #  exposure : total active-image count per domain, from R_counts. Controls
    #             for a concept simply firing more often in one domain.
    reference_high_r = {
        "supportive": tabulate(high_r & (scores.D > tau_d)),
        "harmful": tabulate(high_r & (scores.D < -tau_d)),
    }

    mean_abs_d = {}
    exposure = {}
    for d in range(scores.n_domains):
        vals = absd_dom[:, :, d][nonneutral]
        mean_abs_d[domain_names[d]] = float(vals.mean()) if vals.size else 0.0
        exposure[domain_names[d]] = float(scores.R_counts[:, :, d][nonneutral].sum())

    def share(tbl: Dict[str, int]) -> Dict[str, float]:
        tot = sum(tbl.values())
        return {k: (v / tot * 100 if tot else 0.0) for k, v in tbl.items()}

    return {
        "tau_d": tau_d,
        "tau_r": tau_r,
        "support_floor": support_floor,
        "domains": list(domain_names),
        "n_low_r_supportive": int(pos.sum()),
        "n_low_r_harmful": int(neg.sum()),
        "overall": {"supportive": tabulate(pos), "harmful": tabulate(neg)},
        "overall_pct": {"supportive": share(tabulate(pos)),
                        "harmful": share(tabulate(neg))},
        "reference_high_r": reference_high_r,
        "reference_high_r_pct": {k: share(v) for k, v in reference_high_r.items()},
        "mean_abs_D_by_domain": mean_abs_d,
        "active_image_exposure_by_domain": exposure,
        "per_class": per_class,
    }
