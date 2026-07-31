# Context: SAE concept diagnostics for domain generalization — experiment suite

## 0. How to use this document

This is a handoff brief. It describes a research codebase you should inspect,
the state of results so far, and a list of experiments to run.

Before running anything:

1. Read the existing code. Find where `H`, `D`, and `R` are computed, where the
   masking forward pass lives, and how accuracy is reported. Do not assume the
   structure described here matches the code exactly.
2. Confirm the conventions in Section 3 below against the code. Several open
   discrepancies are listed there. Resolve them first. Everything downstream
   depends on them.
3. Run experiments in the order given in Section 5. E0 through E3 are blocking.
   The rest can run in any order after that.
4. Write every result to disk in the schema in Section 6. Do not only print to
   stdout.

If a spec here conflicts with something you find in the code, stop and report
the conflict rather than guessing.

---

## 1. What the project is

A post-hoc interpretability framework for domain generalization models. The
goal is to understand what a trained model has learned internally, not to
propose a new training method or a test-time intervention.

Setup:

- Dataset: PACS. Four domains: art painting, cartoon, photo, sketch.
- Seven classes: dog, elephant, giraffe, guitar, horse, house, person.
- Sketch is the held-out target domain. The other three are source domains.
- Model: ERM-trained ResNet-50. Frozen throughout.
- A Top-K sparse autoencoder is trained on the final feature map before the
  classifier head. 7x7 spatial, 2048 channels. Dictionary size 16,384, top-16
  sparsity per spatial token.
- The SAE is trained on source domains only. Sketch is used only for
  evaluation.

For each class-concept pair `(k, c)` three scores are computed, all on source
domains only:

- **H(k,c)** — activation invariance. Normalized entropy of mean activation
  across domains. High H means the concept fires evenly everywhere. It says
  nothing about usefulness.
- **D(k,c)** — discriminative effect. Ablate concept `c` from the SAE code,
  decode, classify, and measure the drop in `p(y=k)`. Positive D means the
  concept supports the correct class. Negative D means it hurts.
- **R(k,c)** — discriminative consistency. Normalized entropy of `|D_d(k,c)|`
  across domains. High R means the effect is spread across domains. Low R
  means the effect concentrates in one or two.

All three are class-conditional. The same latent can have different scores for
different classes. This matters constantly and is easy to forget.

Because scores are estimated on three source domains, `log|D| = log 3`. An even
split across only two domains gives `log2/log3 = 0.63`, which is below the 0.7
threshold. So `H >= 0.7` implies nonzero activation in all three source
domains, and `R >= 0.7` implies nonzero effect in all three.

---

## 2. The central open question

The paper's claim is that H and R measure different things, and that
consistency matters beyond invariance.

The risk is that R turns out to be a proxy for the sign of D. Evidence for
that worry:

- Among high-H pairs that hurt the correct class, 70% are low-R (45/64).
- Among high-H pairs that support it, 28% are low-R (28/102).

So the low-R population is enriched in negative-D pairs by roughly 2.5x.
Masking low-R therefore preferentially removes harmful concepts. And masking
harmful concepts raises `p(y)` by construction, since that is how D is defined.

The experiments in Section 5 are designed to separate these two explanations.
E3 is the decisive one.

There is also a second, more interesting finding already emerging: low-R
concepts are not junk. Removing the harmful ones helps a lot. Removing the
supportive ones appears to destroy the model. If that holds, the model's class
evidence substantially consists of domain-contingent concepts. That is a
concept-level mechanism for the discrimination-invariance tradeoff.

---

## 3. Conventions — confirm these before running

### 3.1 Averaging: micro vs macro

There are two accuracy conventions in play and they have been mixed up.

The paper's Table 2 reports sketch at 80.29% (original) and 80.20% (SAE
reconstruction). The experiment scripts report sketch at ~83.44%.

These are almost certainly the same number under different averaging. Weighting
the per-class sketch accuracies by PACS sketch class counts gives roughly 80.3%,
matching Table 2. The scripts appear to report an unweighted macro-average over
classes, which reads higher because `dog` is both the worst class and one of
the largest.

**Action:** verify this by recomputing both from the same run. Then report
**both** micro and macro for every configuration, in every output file. Label
them explicitly. Never report a bare "accuracy".

Also: drop any "overall average over four domains" figure. It averages three
in-distribution domains with one held-out domain and hides the only number that
carries the argument. Report source domains and sketch separately.

### 3.2 The rare-concept count filter — now removed

There was a filter that masked class-concept pairs active on fewer than 30
images for that class. It gave roughly +2.7pp on sketch by itself.

**This filter is class-conditional and leaks labels.** A concept that fires
rarely for class `k` but often for class `k'` is a confusion signal. Masking it
for `k`-labelled images deletes evidence for the competing class using the true
label. The label-free version of the same filter (count < 30 for *all* classes)
masks only ~15k pairs and gives no gain, which confirms the entire effect came
from the class-conditional part.

**Action:**

- Remove the count filter from **all forward passes**. No masking of rare
  pairs during evaluation, ever.
- Keep a support floor as an **inclusion criterion for computing statistics**
  only — that is, when building the H/R heatmap and typology tables, exclude
  pairs with thin support, because `R` is an entropy over three per-domain
  effect estimates and a cell with four images gives a meaningless `D_d`. This
  changes no model behavior and leaks nothing.
- Make the two uses distinguishable in the code, e.g. `support_floor_stats=30`
  and no filter argument at all in the eval path.
- Report both filtered and unfiltered versions of any statistics table.

### 3.3 tau_D

The heatmap used `tau_D = 1e-4`. The paper's methods section promises `1e-3`.
Pick one and use it everywhere. Default to `1e-4` unless the code suggests
otherwise. Run E12 for sensitivity.

### 3.4 Masking protocol

This is the protocol used for every masking experiment below. Confirm the code
matches.

- A mask is a set `S` of `(k, c)` pairs.
- For an image `i` with true label `y_i`, the concepts masked are
  `{c : (y_i, c) in S}`. The mask is therefore class-conditional and uses the
  true label. This is an oracle diagnostic, by design.
- Masking is done in SAE code space: zero that concept's column across all
  spatial tokens, decode, denormalize, then classify. Identical to the D
  ablation protocol.
- A mask only affects an image if the concept actually fires on it.
- Scores used to build masks are always estimated on source domains only.
  Never compute H, D, or R using sketch data.

### 3.5 Oracle access is fine

This is an interpretability paper. Interventions are functional validation of
the diagnostic categories, not proposed methods. A controlled ablation is
allowed to use ground truth — that is what makes it controlled.

So do not avoid oracle masks. Just always label them as oracle in the output,
and always run the label-free variants alongside (E7, E8) so the leakage
question is answered rather than dodged.

---

## 4. Results so far

All sketch figures below are macro-average unless noted. They need
recomputation under a fixed convention (see 3.1).

### 4.1 Baselines (no count filter)

| Domain | Baseline |
|---|---|
| Art painting | 99.33 |
| Cartoon | 99.03 |
| Photo | 99.71 |
| Sketch | 83.44 |

Per-class sketch baseline: dog 48.71, elephant 95.98, giraffe 77.97,
guitar 96.86, horse 82.24, house 87.88, person 94.62.

With the count filter active, sketch reads 86.19. The filter is now removed.

### 4.2 Sign-of-D masking (from the previous section of the paper)

| Mask | Sketch | Overall |
|---|---|---|
| Baseline | 83.46 | 95.39 |
| Mask all `D < 0` | 96.66 (+13.20) | 99.15 |
| Mask all `D > 0` | 5.67 (-77.79) | 9.33 |

Under the all-positive mask, six of seven classes fall to exactly 0.00%. Only
`person` retains accuracy (65.29% averaged over domains). The model appears to
collapse onto a single default label.

### 4.3 Low-R masking

| Mask | Sketch change |
|---|---|
| All `R < 0.8` | +6.13 |
| `R < 0.8` and `D < 0` | +10.03 |
| `R < 0.8` and `D > 0` | collapse to 13.41% |

Note the ordering. Masking the harmful concentrated concepts gives +10.03.
Masking *all* low-R gives only +6.13. The rest of the low-R population is
costing about 4 points.

**Ambiguity to resolve:** it is unclear whether the two rows above were built
as intersections with `R < 0.8`, or whether R was ignored. The captions say one
thing and the prose says another. E1 resolves this.

Evidence they *were* intersections: the collapse figure here is 13.80% overall
with roughly even per-domain distribution, whereas the all-positive mask in 4.2
gave 9.33% with `person` at 65%. Different numbers imply different masks.

If they were intersections, the result is significant: every high-R positive-D
concept was retained — the entire "robust class-supporting" bucket — and the
model still went to chance. That would mean the robust-support bucket does not
contain enough evidence to classify.

### 4.4 Label-free variants — both null

| Variant | Pairs masked | Gain |
|---|---|---|
| `R < 0.7` for all classes | ~15k | none |
| Oracle `R < 0.8` (class-conditional) | ~114k | +6.13 |
| Count < 30 for all classes | ~15k | none |
| Oracle count < 30 | ~100k+ | +2.7 |

Two unrelated criteria, both giving large gains class-conditionally and nothing
uniformly. The common factor is class-conditionality, not the criterion.

This is a finding, not a failure. Almost no concept is concentrated for *every*
class at once. Concepts are domain-contingent for particular classes, not in
general. That is a substantive claim about how the representation is organized.

### 4.5 Gated sweep

Restricting the low-R mask to `|D| > 1e-4` preserves the accuracy gain. So the
effect is not merely from deleting near-inert latents. Noise removal is ruled
out as the explanation.

### 4.6 Heatmap counts (with support floor, tau_D = 1e-4)

Neutral panel, `|D| <= 1e-4`, n=1146:

| | R < 0.7 | R >= 0.7 |
|---|---|---|
| H < 0.7 | 182 (15.9%) | 62 (5.4%) |
| H >= 0.7 | 499 (43.5%) | 403 (35.2%) |

Negative panel, `D < -1e-4`, n=89:

| | R < 0.7 | R >= 0.7 |
|---|---|---|
| H < 0.7 | 25 (28.1%) | 0 (0.0%) |
| H >= 0.7 | 45 (50.6%) | 19 (21.3%) |

Positive panel, `D > 1e-4`, n=128:

| | R < 0.7 | R >= 0.7 |
|---|---|---|
| H < 0.7 | 23 (18.0%) | 3 (2.3%) |
| H >= 0.7 | 28 (21.9%) | 74 (57.8%) |

Derived facts:

- 1068 high-H pairs total. 496 also high-R, i.e. 46.4%. Knowing a concept is
  invariant tells you almost nothing about whether its effect is consistent.
- Only 3 of 217 non-neutral pairs are low-H and high-R. Consistency
  presupposes broad activation.
- Among high-H supportive pairs, 72.5% are high-R. Among high-H harmful pairs,
  29.7% are. Invariant support distributes; invariant harm concentrates.
- 35.2% of neutral pairs land in the high-H/high-R cell. Entropy is
  scale-insensitive, so tiny effects spread evenly score as consistent. This is
  why R must be read behind a magnitude gate.

---

## 5. Experiments to run

Notation. All sets are sets of `(k, c)` pairs, scores from source domains only:

```
S+_lo = { (k,c) : D >  tau_D  and  R <  0.8 }   concentrated support
S+_hi = { (k,c) : D >  tau_D  and  R >= 0.8 }   distributed support
S-_lo = { (k,c) : D < -tau_D  and  R <  0.8 }   concentrated harm
S-_hi = { (k,c) : D < -tau_D  and  R >= 0.8 }   distributed harm
```

For **every** experiment, report: micro and macro accuracy per domain,
per-class sketch accuracy, number of pairs masked (total and per class), and
total `sum |D|` of the masked set (total and per class).

### E0 — Fix the averaging convention (blocking)

Recompute the unmasked baseline reporting micro and macro side by side.
Confirm micro sketch lands near 80.2 and matches the paper's Table 2. Confirm
macro lands near 83.4. Document the PACS per-class image counts used for
weighting.

Deliverable: `results/E0_baseline.json`, plus a note confirming or refuting the
micro/macro explanation.

### E1 — Disambiguate the positive-D collapse (blocking)

Run two masks explicitly and separately:

- **E1a:** mask `{ D > tau_D }`, ignoring R entirely.
- **E1b:** mask `S+_lo`, i.e. `{ D > tau_D and R < 0.8 }`.

Report per-class accuracy for both, all four domains. The question is whether
these are the same experiment. Also record which label the model defaults to
under each collapse — compute the predicted-label histogram, not just accuracy.

### E2 — Complete the harmful partition (blocking)

Mask each of these separately:

- `S-_lo` (already have: +10.03 macro sketch — reproduce it under the fixed
  convention)
- `S-_hi`
- `S-_lo union S-_hi` (should reproduce roughly the +13.2 from 4.2)

Check whether the effects are additive.

### E3 — Mass-matched comparison, harmful population (decisive)

This isolates R from the sign and magnitude of D.

1. Compute `M_hi = sum |D|` over `S-_hi`, per class.
2. Select a subset `S-_lo,matched` of `S-_lo` whose total `|D|` equals `M_hi`,
   matched **per class**, since masks are class-conditional.
3. Use two selection rules: (a) descending `|D|` until the target is reached,
   (b) random selection to the target, 5 seeds.
4. Mask `S-_lo,matched`. Compare against masking all of `S-_hi`.

Interpretation:

- Equal performance → R carries no information beyond `|D|` and sign. The
  section's claim reduces to a correlation and must be rewritten.
- Low-R subset wins → concentration carries independent information. The claim
  is earned.

Report the matched masses achieved, since exact matching will not be possible.

### E4 — Mass-matched comparison, supportive population

Same construction as E3 but with `S+_lo` and `S+_hi`. Match total `|D|` per
class, mask each, compare.

This asks whether the model's dependence on domain-contingent evidence is
disproportionate, not merely larger in aggregate. Given the collapse in 4.3,
this is likely to be interesting.

### E5 — Full typology by sign factorial

Eight masks. Each cell of `(H high/low) x (R high/low) x (sign of D)` at
`tau_H = tau_R = 0.7`, masked alone. Plus two extra configurations:

- **Keep only robust support:** mask everything except
  `{ H >= 0.7 and R >= 0.7 and D > tau_D }`. This fills the "test robust
  sufficiency" row that is currently empty in the paper's intervention table.
- **Keep only high-R:** mask everything with `R < 0.7`, regardless of H and
  sign.

This gives every row of the paper's concept typology a functional signature.

### E6 — Class-conditional random mask control

The null that currently has no number.

Per class, sample a random set of pairs from the active population, size-matched
to the mask being tested. 5 seeds. Report mean and range.

Run two versions:

- naive uniform sampling
- sampling stratified to match the `|D|` histogram of the target mask

The stratified version matters. Uniform random pairs skew neutral, which makes
the control artificially easy to beat.

Size-match against at least: all `R < 0.8`, `S-_lo`, and `S+_lo`.

### E7 — Predicted-label mask

1. Forward pass with no mask, record predicted label `yhat_i`.
2. Build the mask using `R(yhat_i, .) < 0.8`.
3. Re-evaluate.

This is the honest deployable variant and a direct leakage probe. Under the
leakage account it should gain little: correct predictions get reinforced,
wrong ones entrenched. Under the real-structure account it should recover a
meaningful fraction, since `yhat = y` on ~80% of sketch images.

Report both overall accuracy and accuracy split by whether `yhat_i == y_i` at
step 1.

### E8 — Argmax-|D| label-free variant

Less conservative than the all-classes intersection.

For each concept `c`, let `k* = argmax_k |D(k,c)|`. If `R(k*, c) < 0.8`, mask
`c` globally for all images regardless of label.

Report pairs masked, for comparison against the ~15k of the intersection rule
and the ~114k of the oracle rule.

### E9 — Where does concentrated effect live? (no masking needed)

For every low-R pair, record `argmax_d |D_d(k,c)|` over the three source
domains. Cross-tabulate by sign of D and by class.

This is potentially the most explanatory result in the section and it requires
no intervention — the per-domain effects already exist. If harmful concentrated
concepts disproportionately peak in `photo`, the story becomes concrete: the
model learned photorealistic cues that carry over to sketch as active
interference.

Deliverable: a counts table, domains as rows, sign of D as columns, plus a
per-class version.

### E10 — Do diagnostic profiles explain observed failure? (no masking needed)

Compute per-class RSM, HIM, DCM, and conflict mass. Correlate each against
per-class sketch accuracy. Report Spearman rho with an explicit `n = 7` caveat.

A preliminary look suggests HIM correlates with sketch failure at around
rho = 0.68, with `person` as the clear outlier. Verify this properly after
refiltering.

This needs no labels at test time and no intervention, so it is the cleanest
support for the paper's claim that profiles explain failures.

### E11 — R threshold sweep, refiltered, with counts

Sweep the masking threshold on R from 0.0 to 0.9 in steps of 0.1. No count
filter. Report per-domain micro and macro accuracy, plus pairs masked per
threshold per class.

Run three variants:

- ungated: mask all pairs with `R < tau`
- gated: mask only pairs with `R < tau` and `|D| > tau_D`
- harmful only: mask only pairs with `R < tau` and `D < -tau_D`

The shape matters as much as the peak. The previous sweep rose to +7.6 at
tau = 0.8 then fell to +6.4 at 0.9. Confirm the turnover survives refiltering.

### E12 — Sensitivity

- `tau_D` in `{1e-4, 1e-3}`: regenerate the heatmap, the typology counts, and
  the E2/E3 results.
- Support floor on and off for statistics: regenerate the heatmap both ways.
- `tau_H`, `tau_R` in `{0.7, 0.8, 0.9}`: regenerate typology counts.

### E13 — If compute allows

Additional SAE seeds. Dictionaries are not comparable across seeds, so compare
only aggregate quantities: the heatmap proportions, the typology masses, and
the E3 outcome. The methods section commits to seed-level robustness of
aggregate conclusions, so at least two seeds would be valuable. If not
feasible, this becomes a named limitation.

---

## 6. Output schema

Write one JSON file per configuration to `results/`, named after the
experiment ID. Use this schema:

```json
{
  "experiment_id": "E2_S-_hi",
  "description": "mask distributed harmful concepts",
  "mask_definition": "D < -1e-4 and R >= 0.8",
  "oracle": true,
  "oracle_note": "class-conditional, uses true label",
  "tau_D": 1e-4,
  "tau_H": 0.7,
  "tau_R": 0.8,
  "support_floor_stats": 30,
  "count_filter_in_forward_pass": false,
  "pairs_masked": {"total": 0, "per_class": {}},
  "masked_D_mass": {"total": 0.0, "per_class": {}},
  "accuracy": {
    "art_painting": {"micro": 0.0, "macro": 0.0, "per_class": {}},
    "cartoon": {},
    "photo": {},
    "sketch": {}
  },
  "predicted_label_histogram": {"sketch": {}},
  "seed": null,
  "notes": ""
}
```

Also emit a single flat CSV, `results/summary.csv`, one row per configuration,
with columns for experiment ID, mask definition, pairs masked, masked D mass,
and micro and macro accuracy for each of the four domains. This is what gets
turned into paper tables.

---

## 7. Pitfalls

1. **Never compute H, D, or R on sketch data.** Source domains only. This is
   the core methodological commitment of the paper.
2. **The count filter goes nowhere near a forward pass.** Statistics only.
3. **Masks are class-conditional.** The same latent can be masked for one class
   and retained for another. Do not collapse a mask to a set of concept indices
   without keeping the class dimension.
4. **Report micro and macro every time.** Do not emit a bare accuracy number.
5. **Report pair counts and `|D|` mass every time.** A large gain from a mask
   ten times the size of its comparison set means nothing.
6. **Sign of D guarantees direction, not magnitude.** Masking negative-D pairs
   must raise `p(y)` by construction. Only the magnitude, and the transfer to
   sketch, carry information. Do not write up the direction as a finding.
7. **Watch for chance-level collapse.** 1/7 is 14.3%. A configuration landing
   near that has been erased, not degraded. Always emit the predicted-label
   histogram so the difference is visible.
8. **Do not average the four domains together.** Three are in distribution.
