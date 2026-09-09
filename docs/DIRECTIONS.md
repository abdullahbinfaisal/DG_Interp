# Directions — claims the results point at but do not establish

Each entry: the **hypothesis**, why the existing results motivate it, the
**investigation** that would settle it, what each outcome would mean, and cost.

Ordered by value per unit of effort. D2–D5 are cheap and would each add a real
claim; D6–D8 are the expensive ones that would most strengthen the paper.

Companion: `docs/DISCUSSION.md` for what is already established.
`docs/RESULTS_LEDGER.md` for findings settled after the ERM-only draft (D1 has
moved there — see its note below).
Method specs for previously-defined experiments live in `EXPERIMENTS.md` §10.

---

**D1 — settled, moved to `docs/RESULTS_LEDGER.md` F1.** *Does an
invariance-trained model actually have fewer harmful-invariant concepts?*
Measured 2026-08-26/27 across two alignment objectives (MMD, DANN), extending
D1's original MMD-only scope. Neither reduces the harmful-invariant bucket;
MMD retains significantly more of it (p=0.048), DANN's lower raw count is not
statistically distinguishable from ERM's rate (p=0.372). The paper's central
argument is demonstrated, not merely asserted. See the ledger for the full
result, the mechanistic argument for why it generalizes, and the scope limits
(aggregate-level only, independently-trained SAEs not a tied USAE) — and
`docs/DIRECTIONS.md` D7 below, whose priority this result raises rather than
lowers.

---

## D2. Does non-robust material accumulate over training?

**Hypothesis.** The harmful-invariant bucket grows during training, as the model
fits increasingly source-specific structure. If so, the diagnostic measures
something like representational overfitting that target accuracy alone hides.

**Why now.** The USAE checkpoint `USAE_ERM_Multi_test_3300_2100.pt` was trained by
rotating the encoder across steps 2100 and 3300 and decoding each code with *every*
SAE, so **the two dictionaries are aligned by construction**. Latent *k* means the
same thing at both checkpoints. This is a genuine capability the draft never
mentions, and it makes a checkpoint comparison legitimate where cross-seed
comparisons would not be.

Step 2100 is the *oracle* selection (best mean accuracy including sketch) and 3300
the non-oracle one, so the pair is additionally interesting: 2100 is the checkpoint
target accuracy prefers.

**Investigation.** `build_scores.py` with `--preset` pointing at ckpt 2100, then
`analyze_scores.py` and stage B2. Compare bucket sizes, the asymmetry ratio, and
error recovery. Because dictionaries are aligned, also track *individual* pairs:
does a given harmful-invariant pair at 3300 already exist at 2100?

**Outcomes.** Bucket grows 2100 → 3300 → non-robust material accumulates, and the
diagnostic sees something accuracy does not. Bucket stable → it is a property of the
architecture/data rather than of training duration, which is also worth reporting.

**Cost.** ~10 min scoring + 5 min interventions. No training. Cheapest meaningful
extension.

---

## D3. Is generalization failure a collapse of redundancy?

**Hypothesis.** The model fails on sketch because it operates near its decision
boundary there, so individual concepts become individually decisive. Formally: mean
per-concept ablation effect should be much larger on sketch than on any source
domain, and should rank-order with domain difficulty.

**Why the current results motivate it.** `DISCUSSION.md` §4. Ablating 99.93% of the
grid leaves source accuracy unchanged while moving sketch 10 points — the same
intervention is a no-op in distribution and decisive out of it. And among source
domains, mean |D_d| is 1.15e-3 (photo, 99.78% acc), 1.92e-3 (cartoon, 99.19%),
2.13e-3 (art, 99.35%): the easiest domain is the least concept-sensitive. Three
points, one of them confounded by exposure — suggestive only.

**Investigation.** Compute per-domain D on **sketch** and compare mean |D_d| and its
distribution against the three source domains. This requires care: the protocol
forbids using sketch to *estimate scores used for masks or thresholds*. Here sketch
D would be a **reported diagnostic quantity only**, never feeding a mask, a
threshold or a bucket definition. Implement as a separate output (e.g.
`--diagnostic-domains 3`) so it cannot leak into `build_buckets`, and label it
unambiguously in any table.

A cleaner, leak-free alternative: measure **prediction margin** (top-1 minus top-2
logit) per domain on the unablated model, and correlate per-domain mean margin with
per-domain mean |D_d| over the three source domains. If margin explains
concept-sensitivity within sources, the extrapolation to sketch follows from
sketch's margin alone, which needs no sketch ablations at all.

**Outcomes.** Confirmed → a mechanistic account of DG failure as redundancy
collapse, measurable from source data via margin, and a genuinely new framing.
Refuted → §4 of the Discussion must be cut back to the bare observation.

**Cost.** Margin version: one evaluation pass, ~30 s. Sketch-D version: ~4 min.

---

## D4. How much of the oracle gain survives without labels?

**Hypothesis.** Because `ŷ = y` on ~80% of sketch images, a mask addressed by the
*predicted* label should recover a meaningful fraction of the oracle gain.

**Why now.** Every intervention in §5.4 is oracle. This is the paper's most obvious
reviewer objection and the honest answer is currently "the label-free variants we
tried were null". But those variants were structurally different: they collapsed the
mask across classes (intersection over all classes, or argmax-|D|), which throws
away the class-conditionality that `DISCUSSION.md` §6 argues is where the signal
lives. The predicted-label variant preserves class-conditionality and has never been
run.

**Investigation.** E7 as specified in `EXPERIMENTS.md` §10: forward pass unmasked,
record `ŷ_i`, rebuild the mask as `S(ŷ_i, ·)`, re-evaluate. Report overall accuracy
**and** accuracy split by whether `ŷ_i == y_i` on the first pass — the split is the
informative part. Run it for `all_harmful` and for `keep_robust_support`.

**Outcomes.** Recovers a substantial fraction → there is a deployable version, which
considerably raises the paper's ceiling. Gains little → the leakage account holds for
the keep-only rows, and we say so; the mask-only rows are unaffected because their
value is diagnostic. Either way this closes the objection rather than deferring it.

**Cost.** 2–4 configurations, ~2 min. Highest value among the cheap options.

---

## D5. Is the residual error irreducible?

**Hypothesis.** After keeping all supportive concepts (94.25% micro, 70.9% of errors
recovered), the remaining 5.75% of sketch images fail because **no** supportive
concept fires on them at all — the representation has nothing to say about them.

**Why it matters.** It converts "70.9% of errors recovered" into a statement about
the ceiling: whether the residual is a selection problem (fixable by better
read-out) or a coverage problem (fixable only by a better representation). Those
imply completely different remedies, and the distinction is exactly what §9 of the
Discussion claims the tool provides.

**Investigation.** No new training. For each sketch image still misclassified under
`keep_all_supportive`, record how many of its true class's supportive concepts were
active. Cross-tabulate against whether the image was already wrong at baseline.
Requires a small extension to `eval.py` to dump per-image active-concept counts
under a mask.

**Outcomes.** Residual errors have ~zero active supportive concepts → coverage
problem, and the paper can state a representational ceiling. They have supportive
concepts active but still lose → selection/competition problem, and the conflict
story of §6 extends to explain the residual.

**Cost.** ~1 h of code, one evaluation pass. Zero risk.

---

## D6. Does concentrated effect really track *style*, or just effect size?

**Hypothesis.** Domain-contingent effect lives in the stylised source domains
because the concepts carrying it are style-sensitive, and the target domain is a
*different* style — hence the failure to transfer.

**Why it is not yet established.** §5.5 found concentrated support enriched 2.34× in
cartoon and concentrated harm 1.66× in art_painting, with photo depleted 3–4× in
both. But photo also has the smallest mean effect (1.15e-3) and lowest exposure, so
part of the depletion is mechanical. The high-R null comparison controls for this
partially, not fully.

**Investigation.** Two steps. (a) Intervention: mask only the pairs whose effect
peaks in art_painting, then only those peaking in cartoon, then only photo, each
size- and mass-matched, and compare sketch effect. If the style hypothesis holds,
the art/cartoon-peaked masks should matter more for sketch than the photo-peaked
one, beyond what their sizes predict. (b) Recompute the attribution using a
*within-domain standardised* effect (|D_d| divided by that domain's mean |D|), which
removes the global per-domain scale difference outright.

**Outcomes.** Style hypothesis supported → §5.5 becomes a mechanism rather than a
correlation, and the paper gains its most concrete causal story. Not supported →
§5.5 stays a descriptive table, which is where it currently sits.

**Cost.** (b) is zero GPU, ~30 min. (a) is 6–9 configurations, ~5 min.

---

**D7 — settled, moved to `docs/RESULTS_LEDGER.md` F2.** *Does R carry
information beyond the sign and magnitude of D?* Measured 2026-08-27 via a
distribution-matched subset selector (not the sum-matched design originally
sketched below — see the ledger for why), across all three backbones. On the
harmful side: yes, consistently, 3/3 backbones — the true high-R harmful
bucket recovers 2.3–5.5× more accuracy than a magnitude-matched low-R subset
of the same size. On the supportive side: yes in ERM and MMD (matched subset
costs 8–10× more than the true low-R bucket), but DANN's comparison inverted
and was undercovered (30 of 74 pairs matched) — flagged as unresolved rather
than papered over. This directly contradicts the paper's standing §10 hedge
("we do not claim R carries information independent of D's sign and
magnitude") on the harmful side. See the ledger for full numbers and the
DANN caveat.

---

## D8. Do the aggregate conclusions survive a different sparse basis?

**Hypothesis.** The typology proportions, the support/harm asymmetry and the
error-attribution figures are properties of the model, not of one SAE.

**Why it matters.** §3.2 of the draft already commits to evaluating robustness "at
the level at which claims are made", and SAE dictionaries are known to be
non-identifiable across seeds. This is the standard objection to any SAE-based
result.

**Investigation.** Train a second SAE on the same ERM checkpoint with a different
seed (and ideally a different dictionary size), then compare **aggregate quantities
only**: the H×R contingency proportions, the asymmetry ratio at each τ, the fraction
of target error recovered by masking all harmful pairs, and the keep-only sufficiency
figure. Individual latent indices are not comparable and should not be matched.

**Outcomes.** Aggregates stable → the claims are about the model. Aggregates move →
they are partly about the dictionary, and the paper must say so prominently.

**Cost.** One SAE training run (~2–3 h) plus a full re-run of Group A and B1–B5
(~25 min). Mechanically trivial given the harness; just slow.

---

## Deprioritised

| Idea | Why not now |
|---|---|
| Other datasets (DomainNet, OfficeHome logs exist in `analysis/results/`) | `data.py` is PACS-only; each dataset needs a loader, a backbone and an SAE. Large scope increase for external validity we can instead scope honestly. |
| RSM / HIM / DCM aggregate profiles (draft §3.8) | `DISCUSSION.md` §7 shows per-class vulnerability is at least two-dimensional, so a one-dimensional profile score is likely to mislead. Cut rather than deferred. |
| `M.py` winner-take-all contrast | Conflict is fully computable from D, and M's normalisation (by total class images rather than active ones) answers a different question. Revisit only if a per-concept conflict *magnitude* becomes a headline quantity. |
| τ_D = 1e-3 sensitivity | One flag (`analyze_scores.py --tau-d 1e-3`), ~1 min. Do it on demand rather than pre-emptively; τ = 0.7/0.8/0.9 sensitivity is already reported and monotone. |
| `processors/B.py` | Still an unexplained stub. Nothing depends on it. |
