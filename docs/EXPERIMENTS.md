# EXPERIMENTS.md — what to run, why, and what we expect

Companion to `CLAUDE.md` (repo blueprint + conventions) and `docs/context.md`
(the original E0–E13 brief). **This file supersedes `context.md` §5 where they
disagree**, because `context.md` was written against the old signed-R score file
and a τ_R of 0.8, both of which the clean data has since revised.

Everything below is defined against **`processed/FINAL_ERM_ResNet_3300_T3.json`**
— source-domains-only, magnitude-R, full-domain deterministic scoring.

Legend: **[BLOCKING]** must finish before later work is meaningful ·
**[DECISIVE]** its outcome determines whether a paper claim survives ·
**[CHEAP]** no GPU, minutes · **[EXPERIMENTAL]** not committed to the paper.

---

## 0. Protocol invariants

These are not negotiable and every experiment inherits them (see `CLAUDE.md` §3):

1. H, D, R are estimated on **source domains only**. `RunConfig.validate()`
   hard-fails otherwise.
2. Report **micro and macro** for every configuration. Never a bare accuracy.
   Never an average over all four domains.
3. Report **pairs masked** and **Σ|D| of the masked set**, total and per class.
   A gain from a mask ten times the size of its comparison set means nothing.
4. Masks are **class-conditional**. Never collapse to a set of concept indices.
5. The support floor is a **statistics** inclusion criterion only. It must never
   touch a forward pass — as a class-conditional mask it leaks labels and is
   worth ~+2.7pp of pure leakage.
6. Emit the **predicted-label histogram** for anything near 14.29% (= 1/7), so
   erasure is distinguishable from degradation.
7. Every run writes `results/E*.json` per `context.md` §6 plus a row in
   `results/summary.csv`. Nothing counts if it only reached stdout.

**Cost note.** One masked evaluation over all four full domains is 158 batches,
about **25–30 s**. The entire suite below is a few hours of GPU at most. Compute
is not the constraint; the constraint is deciding what the numbers mean.

---

## 1. Set definitions and their actual populations

At τ_D = 1e-4, τ_H = τ_R = 0.7, support floor 30 (1542 pairs; 273 non-neutral):

```
S+_lo = { D > +τ_D , R <  τ_R }   concentrated support     54 pairs
S+_hi = { D > +τ_D , R >= τ_R }   distributed support      81 pairs
S-_lo = { D < -τ_D , R <  τ_R }   concentrated harm       103 pairs
S-_hi = { D < -τ_D , R >= τ_R }   distributed harm         35 pairs
```

Broken out by H (rows H<τ / H≥τ, columns R<τ / R≥τ):

| | supportive (135) | harmful (138) |
|---|---|---|
| H<τ | 25 / 5 | 46 / 3 |
| H≥τ | 29 / **76** | 57 / **32** |

The bold cells are the paper's two headline categories: **robust support** (76)
and **harmful invariant** (32).

**Two consequences to note before designing E3/E4.**

- `S-_lo` (103) is ~3× `S-_hi` (35), so mass-matching in E3 works in the natural
  direction: draw a subset of `S-_lo` up to `S-_hi`'s mass.
- `S+_lo` (54) is *smaller* than `S+_hi` (81). If `S+_lo`'s total Σ|D| is below
  `S+_hi`'s, E4 **cannot** be matched in that direction and must instead subset
  `S+_hi` down to `S+_lo`'s mass. Check the masses before building the mask;
  do not silently produce an unmatched comparison.

---

## 2. Phase A — foundations [BLOCKING]

Nothing downstream is interpretable until these three land.

### E0 — Baseline under the fixed protocol [BLOCKING]

**What.** Unmasked accuracy: original model and SAE reconstruction, per domain,
micro and macro, per class, on full domains with `drop_last=False`.

**Why.** Two reasons. First, the paper's Table 2 (80.29 sketch) and the scripts
(83.4x) have been silently reporting different averagings of the same run; that
has to be pinned to disk once, with both numbers side by side. Second, every Δ in
this document is measured against this baseline, and the old baselines carried
±0.08pp of jitter from `shuffle=True` + `drop_last=True` dropping a different
~7 sketch images per call.

**Expected.** Sketch micro ≈ 80.2–80.3, macro ≈ 83.4–83.5; sources 99.0–99.8.
Micro should reproduce Table 2 to within rounding — I verified this analytically
already (weighting the per-class sketch accuracies by PACS counts gives 80.30 vs
Table 2's 80.29). Reconstruction drop ≤ 0.1pp on every domain. Now exactly
reproducible run to run.

**Falsifier.** If micro sketch lands far from 80.3, the micro/macro explanation is
wrong and Table 2 came from a different population or checkpoint — which would
put the whole reconstruction-fidelity section in question.

**Note.** Full-domain evaluation now includes the 20% of source images previously
held out of scoring, so small shifts (±0.3pp) from the notebook numbers are
expected and fine. Anything larger needs explaining.

### E1 — Disambiguate the positive-D collapse [BLOCKING]

**What.** Two masks, run and reported separately:
- **E1a**: mask `{D > τ_D}` (135 pairs), ignoring R.
- **E1b**: mask `S+_lo` (54 pairs).
Per-class accuracy for all four domains, plus predicted-label histograms.

**Why.** `context.md` §4.3 cannot tell whether its collapse rows were built as
intersections with R or with R ignored — the captions and the prose disagree.
Every later claim about "removing supportive concepts destroys the model" depends
on which mask was actually used. This is bookkeeping, but it is load-bearing
bookkeeping.

**Expected.** E1a collapses hard (near or below 14.29%, six of seven classes at
0.00%, everything defaulting to `person`). E1b degrades substantially but should
*not* fully collapse, since all 81 `S+_hi` pairs including the 76 robust-support
pairs are retained. The interesting quantity is the gap between them.

**Falsifier / most informative outcome.** If E1b *also* collapses to chance, then
the robust-support bucket — the paper's positive headline category — does not by
itself contain enough evidence to classify. That is a genuinely important
negative result and it should be reported, not buried.

### E2 — Complete the harmful partition [BLOCKING]

**What.** Mask `S-_lo` (103), `S-_hi` (35), and their union (138), separately.
Check additivity of the three effects.

**Why.** The harmful side is where the paper's argument lives, and only one of
these three cells has ever been measured. Additivity tells us whether the two
harm categories are independent mechanisms or overlapping ones.

**Expected.** All three improve sketch, because masking negative-D pairs raises
`p(y)` *by construction* — see the risk register, §8.1. Ordering guess: union >
`S-_lo` > `S-_hi`, roughly tracking pair count and Σ|D|. Sub-additivity is more
likely than clean additivity.

**Do not report the direction as a finding.** Only the magnitude, the transfer to
sketch, and the comparison against mass-matched controls carry information.

---

## 3. Phase B — is R more than the sign of D? [DECISIVE]

This phase decides whether §5.3 of the paper stands as written, gets rewritten as
a correlation, or gets cut.

### E3 — Mass-matched harmful comparison [DECISIVE]

**What.** Compute `M_hi = Σ|D|` over `S-_hi` per class. Select
`S-_lo,matched ⊆ S-_lo` with the same per-class Σ|D|, by two rules: (a) descending
|D| until the target is met, (b) random to target, 5 seeds. Mask it. Compare
against masking all of `S-_hi`. Report the masses actually achieved — exact
matching will not be possible.

**Why.** This is the one experiment that separates "R carries information beyond
the sign and magnitude of D" from "low-R happens to be enriched in harmful
pairs". On the clean file the harmful population is 2.0× enriched in low-R
(36.0% vs 72.4% high-R rates), so the confound is real and quantified.

**Expected.** Honestly uncertain, which is why it is decisive. My prior is a
*small* advantage to the low-R subset — the enrichment is 2.0×, not 10×, so most
of the effect probably is sign-driven.

**Interpretation, committed to in advance:**
- Equal performance → R adds nothing beyond |D| and sign. §5.3 must be rewritten
  as a correlational observation and the "consistency matters" framing dropped.
- Low-R subset clearly wins → concentration carries independent information and
  the claim is earned.

Writing both branches down now is what stops this becoming a post-hoc
rationalisation.

### E4 — Mass-matched supportive comparison

**What.** As E3, on `S+_lo` vs `S+_hi`. **Check mass feasibility first** — see §1;
`S+_lo` is the smaller set and the match may have to run in the reverse
direction.

**Why.** Asks whether the model's dependence on domain-contingent evidence is
disproportionate rather than merely larger in aggregate. Given the collapse
behaviour in `context.md` §4.3 this is likely to be the more interesting of the
two mass-matched experiments.

**Expected.** If masking mass-matched `S+_lo` hurts sketch more than masking
`S+_hi`, the model's class evidence leans on domain-contingent concepts out of
proportion to their effect mass — a concept-level mechanism for the
discrimination–invariance tradeoff, and the strongest positive result available
in this paper.

### E6 — Random and stratified mask controls [DECISIVE]

**What.** Per class, sample random pair sets size-matched to each mask under test
(at minimum: all `R < τ_R`, `S-_lo`, `S+_lo`), 5 seeds, mean and range. Two
variants: naive uniform, and **stratified to match the target mask's |D|
histogram**.

**Why.** Currently the null has no number at all. Without it, a reviewer can say
that removing any large set of concepts helps a weak domain. The stratified
variant is the one that matters — uniform sampling skews neutral, which makes the
control artificially easy to beat.

**Expected.** Uniform random ≈ no change (most pairs are inert). Stratified
random will capture a real fraction of the effect; the honest claim is whatever
margin survives *that* comparison, not the margin over the uniform control.

### E6b — Inert-pair control [DECISIVE, CHEAP]

**What.** Mask exactly the pairs with `R == 0` — 111,530 of the 114,688 total.
Then mask `R < 0.1` minus the inert set. Report both.

**Why.** This is the confound `context.md` only gestures at, and I think it is the
biggest unexamined threat to the low-R story. The R-threshold sweep gains +3.35pp
by τ=0.1, but almost every pair removed at that point has R exactly 0, meaning
zero measured effect in all three domains or never active for that class. If
masking the inert set alone reproduces most of the early gain, then a large part
of the "low-R" effect is **dictionary denoising**, not a claim about concentrated
concepts. The paper draft already flags this as a TODO in §5.3.

**Expected.** Masking inert pairs should be near-neutral if they are truly inert.
If it is *not* near-neutral, the low-R narrative needs substantial requalification
and the sweep should only ever be reported gated on |D| > τ_D.

---

## 4. Phase C — typology and profiles

### E5 — Full typology factorial

**What.** Eight masks, one per cell of (H high/low) × (R high/low) × sign(D) at
τ_H = τ_R = 0.7, each masked alone. Plus two keep-only configurations:
- **keep only robust support**: mask everything except the 76 high-H/high-R/D>0
  pairs.
- **keep only high-R**: mask everything with `R < τ_R` regardless of H and sign.

**Why.** Gives every row of the paper's Table 1 typology a functional signature,
and fills the "test robust sufficiency" cell of Table 6, which is currently empty.

**Expected.** The keep-only-robust-support row is the one to watch. The archived
(contaminated, four-domain) run gave sketch 46.6% macro with elephant, guitar and
house at exactly 0.00% — indicative only, but it suggests robust support alone is
*not* sufficient to classify. Expect several of the eight single-cell masks to be
near-baseline simply because the cells are small (3 and 5 pairs in the low-H
high-R cells); report pair counts alongside so small-cell nulls are not read as
substantive.

### E9 — Where does concentrated effect live? [CHEAP]

**What.** For every low-R pair, record `argmax_d |D_d(k,c)|` over the three source
domains. Cross-tabulate by sign of D and by class.

**Why.** Potentially the most explanatory result in the paper, and it needs **no
masking and no GPU** — the per-domain effects are already in `R_acts`. Run it
early; it costs minutes.

**Expected.** If harmful concentrated concepts disproportionately peak in `photo`,
the story becomes concrete and mechanistic: the model learned photorealistic cues
that transfer to sketch as active interference. A flat distribution across domains
would be a much weaker result but still worth reporting.

### E10 — Do diagnostic profiles explain observed failure? [EXPERIMENTAL, CHEAP]

**What.** Per-class RSM / HIM / DCM / conflict mass, correlated against per-class
sketch accuracy. Spearman ρ with an explicit n = 7 caveat.

**Why.** Needs no labels at test time and no intervention, so it is the cleanest
possible support for "profiles explain failures".

**Expected.** Weak. `context.md` reports a preliminary HIM correlation around
ρ = 0.68 with `person` as an outlier, but the clean per-class counts already argue
against a simple story: `dog` has the most harmful pairs (50) *and* the worst
sketch accuracy (48.71%), which fits — but `person` has 32 harmful pairs and
94.62% accuracy, which does not. With n = 7 this cannot support a strong claim
regardless of what ρ comes out at.

**Status.** Per decision 7, RSM/HIM/DCM/conflict mass remain **experimental**.
Treat E10 as exploratory; do not build a headline claim on it. If it is reported,
it belongs in a clearly-hedged subsection.

---

## 5. Phase D — leakage and label-free variants

Every masking experiment above is an **oracle** diagnostic: it uses the true label
to decide what to mask. That is defensible for a controlled ablation and the paper
says so, but it has to be answered rather than dodged.

### E7 — Predicted-label mask

**What.** Forward pass with no mask, record `ŷ_i`. Rebuild the mask using
`R(ŷ_i, ·) < τ_R`. Re-evaluate. Report overall accuracy **and** accuracy split by
whether `ŷ_i == y_i` on the first pass.

**Why.** The honest deployable variant and a direct leakage probe.

**Expected.** Under the leakage account it gains little: correct predictions get
reinforced and wrong ones entrenched. Under the real-structure account it recovers
a meaningful fraction, since `ŷ = y` on ~80% of sketch images. The split-by-correct
breakdown is what distinguishes these, so it is not optional.

### E8 — Argmax-|D| label-free mask

**What.** For each concept `c`, let `k* = argmax_k |D(k,c)|`. If `R(k*,c) < τ_R`,
mask `c` globally for all images regardless of label. Report pairs masked for
comparison against the ~15k of the all-classes intersection rule and the ~114k of
the oracle rule.

**Why.** Less conservative than the all-classes intersection, which was null.

**Expected.** Something between the two, probably small. The known result here is
that **both** label-free variants tried so far are null while the oracle versions
give large gains — and the same is true for the unrelated count criterion. The
common factor is class-conditionality, not the criterion. That is itself the
finding: almost no concept is concentrated for *every* class at once, so concepts
are domain-contingent for particular classes rather than in general.

---

## 6. Phase E — sensitivity and robustness

### E11 — R-threshold sweep, refiltered

**What.** Sweep the masking threshold on R from 0.0 to 0.9 in steps of 0.1, in
three variants: ungated (`R < τ`), gated (`R < τ ∧ |D| > τ_D`), and harmful-only
(`R < τ ∧ D < -τ_D`). Per-domain micro and macro, plus pairs masked per threshold
per class.

**Why.** The *shape* matters as much as the peak. The old sweep rose to +7.61 at
τ=0.8 then fell to +6.39 at 0.9; if ablation were simply good for a weak domain
the curve would keep rising, so the turnover is evidence that the mask selects a
specific population.

**Expected.** The turnover should survive on clean R. The critical comparison is
ungated vs gated at low τ — that is E6b's confound viewed as a curve. Read this
experiment and E6b together.

### E12 — Threshold sensitivity

**What.** Regenerate the typology counts and the E2/E3 results for
τ_D ∈ {1e-4, 1e-3}, τ_H and τ_R ∈ {0.7, 0.8, 0.9}, and the support floor on/off.

**Why.** The paper commits to reporting this, and the main qualitative conclusion
must not depend on a single threshold.

**Expected.** Already partly known and encouraging: the support/harm asymmetry is
2.0× at 0.7, 2.4× at 0.8, 3.2× at 0.9 — monotone, so the qualitative claim is
threshold-robust even though the numbers move. τ_D = 1e-3 is the open question;
at that gate the non-neutral population shrinks sharply and several typology
cells will likely become too small to interpret, which is the argument for
committing to 1e-4 in the main text.

### E13 — Additional SAE seeds

**Status.** Out of scope per decision 4 (ERM checkpoint 3300 only). Dictionaries
are not comparable across independently trained SAEs, so only aggregate
quantities — heatmap proportions, typology masses, the E3 outcome — could be
compared.

**Action.** Name it as an explicit limitation. Note in §4.2 that the tied
multi-checkpoint USAE gives *cross-checkpoint* dictionary alignment (steps 2100
and 3300 share a latent space by construction), which is a partial substitute and
is currently undocumented in the draft.

---

## 7. Diagnostics with no GPU cost [CHEAP]

Run these first; they are minutes each and several change how later results are
read.

1. **Sign-flip audit.** 376 pairs (0.33%) have mixed-sign `R_acts`. How many fall
   inside the gated population, and does excluding them change the 72.4% / 36.0%
   asymmetry? In the old file these were forced to `R = -1` and swept into
   "low-R"; this quantifies what that did.
2. **Old-vs-new attribution.** Same mask rule, same eval protocol, run against
   both `NEW_*` and `FINAL_*`. Decomposes how much of the change came from the R
   definition versus the split change. Needed to explain to a co-author why
   `context.md` §4 numbers moved.
3. **E9** (above) — pure `R_acts` arithmetic.
4. **Typology counts and the Fig. 5 heatmap** regenerated against the clean file.
   The §4.6 tables in `context.md` are stale.
5. **Per-class Σ|D| masses** for all four S-sets, needed as inputs to E3/E4 and to
   confirm the E4 feasibility question in §1.

---

## 8. Risk register — what would sink the paper

**8.1 The sign tautology.** Masking negative-D pairs must raise `p(y)`; that is
how D is defined. Any table row showing "masking harmful concepts helps" is
uninformative on its own. Only three things carry information: the *magnitude*,
the *transfer to the held-out domain*, and the comparison against *mass-matched
and stratified-random* controls. E3 and E6 are therefore not optional extras —
they are what makes Phase A publishable.

**8.2 R as a proxy for sign.** Quantified at 2.0× enrichment on clean data. E3
decides this. If E3 comes out equal, §5.3 must be rewritten.

**8.3 Denoising rather than structure.** 111,530 of 114,688 pairs have R exactly
0. E6b decides how much of the low-R gain is dictionary cleanup.

**8.4 Oracle masks throughout.** Mitigated by honest labelling plus E7/E8, not by
avoidance. The current state of evidence is that label-free variants are null,
and the paper must say so plainly.

**8.5 Small cells.** Several typology cells hold 3–6 pairs at τ=0.7 and the
central harmful-invariant bucket holds 32. Any per-cell claim needs its n stated
inline. At τ=0.9 that bucket is 6 pairs and cannot support a headline.

**8.6 Stale numbers in the draft and brief.** Every masking figure in
`context.md` §4 and `CLAUDE.md` §8 came from signed-R on the 80% split. They are
superseded and must not be copied into the paper. Regenerate or delete.

**8.7 Single model, single checkpoint, single seed, single dataset.** Scope
decision, not a defect — but it caps the external-validity claim and the paper
should state the cap rather than let a reviewer find it.

---

## 9. Recommended order

```
CHEAP diagnostics (§7 items 1–5)        ~1 hour, no GPU, changes how we read everything
   │
E0  baseline  [BLOCKING]
   │
E1, E2        [BLOCKING]                establishes the partition behaves
   │
E6b inert control  ─┐
E6  random controls ─┤ these three decide whether Phase A means anything
E3  mass-matched   ─┘ [DECISIVE]
   │
   ├── if E3 favours low-R → E4, then E5 typology, then E9/E10 profiles
   └── if E3 is null       → rewrite §5.3 as correlational, pivot the paper's
                             weight onto E4 (disproportionate dependence on
                             domain-contingent evidence) and E9 (where effect
                             concentrates), both of which survive an E3 null
   │
E7, E8        leakage answers
E11, E12      sweeps and sensitivity
E13           limitation paragraph only
```

The branch after E3 is the important structural point: **the paper has a viable
story either way**, but they are different stories and the writing should not
commit until E3 is in. If R turns out to be a sign proxy, the surviving
contributions are the class-conditional framework itself, the demonstration that
invariance does not imply usefulness (§5.2, already solid), the
support/harm asymmetry as a *correlation*, and E4/E9.
