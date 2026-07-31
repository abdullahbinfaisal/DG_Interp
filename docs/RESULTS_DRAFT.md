# Results — draft for `TMLR_Journal_Submissions__1_.md` §5

**Provenance.** All numbers from `processed/FINAL_ERM_ResNet_3300_T3.json` (ERM
ResNet-50, DomainBed checkpoint 3300 selected non-oracle on source envs; H, D, R
estimated on art_painting/cartoon/photo only; magnitude-R). Evaluation on full
domains, no split, no shuffle, no dropped batch. Thresholds τ_D = 1e-4,
τ_H = τ_R = 0.7, statistics support floor `H_counts ≥ 30`.

Run manifests: `results/runs/20260731T1{11215,12336,12619,13123,14039,20854}Z_*`.
Per-configuration records in `results/E_*.json`, flat table in
`results/summary.csv`, score-derived tables and figures in `results/analysis/`.

Prose below is paper-ready. Table and figure numbers are provisional; the patch
list at the end says what to do with the existing draft.

---

## 5.1 The sparse basis preserves classifier behaviour

Before using SAE latents as an analysis basis we verify that the reconstruction
preserves the classifier's behaviour, since a poor reconstruction would make every
downstream ablation an artifact of the autoencoder rather than a perturbation of
the model. Table 1 reports accuracy for the original model and for the same
classifier applied to SAE-reconstructed features, on every image of each domain.

We report micro-averaged (image-weighted) and macro-averaged (class-weighted)
accuracy separately throughout. The two differ substantially on the target domain
— sketch is 80.22% micro but 83.63% macro — because PACS sketch is heavily
imbalanced (772 dog images against 80 house images) and the largest class, dog, is
also the weakest. Reporting a single unqualified accuracy for this domain would
obscure a 3.4-point discrepancy, so we always give both.

**Table 1.** SAE reconstruction fidelity, ERM ResNet-50, checkpoint 3300. Sketch
is the held-out target domain.

| Domain | Original micro | Recon. micro | Δ micro | Original macro | Recon. macro |
|---|---|---|---|---|---|
| Art painting | 99.37 | 99.32 | 0.05 | 99.38 | 99.35 |
| Cartoon | 99.19 | 99.15 | 0.04 | 99.23 | 99.19 |
| Photo | 99.82 | 99.82 | 0.00 | 99.78 | 99.78 |
| Sketch (target) | 80.25 | 80.22 | 0.03 | 83.56 | 83.63 |

Reconstruction changes accuracy by at most 0.05 percentage points on any domain,
including the weakest. Concept ablations below are therefore interpretable as
perturbations of the model's own representation, subject to the caveat in §3.2
that they are ablations in reconstruction space rather than causal interventions
in the original network.

---

## 5.2 Activation invariance does not imply discriminative usefulness

Many domain-generalization objectives rest on the premise that a feature which
persists across source domains is more likely to be causal, and therefore more
likely to transfer. Our first result is that this premise is necessary but far
from sufficient: activation invariance and discriminative usefulness are close to
independent.

Of the 1542 class–concept pairs that clear the support floor, **1269 (82.3%) are
discriminatively neutral** (|D(k,c)| ≤ τ_D), 135 (8.8%) support the correct class,
and 138 (8.9%) actively hurt it. Restricting to the invariant population does not
change this picture: among the 1173 pairs with H ≥ 0.7, **83.5% are neutral**,
9.0% supportive and 7.6% harmful. Figure 1 shows the corresponding scatter — the
high-H region spans positive, near-zero and negative D rather than concentrating
at positive D.

The interpretation of the three regimes differs sharply. A neutral concept
activates uniformly across domains yet does not separate classes; a *legs* concept
that fires on elephants, dogs and horses alike contributes nothing to
discriminating among them. A negative-D concept is domain-invariant yet actively
harmful: §5.6 gives a concrete example of a chest-region concept that fires on
horses across every domain and pushes the model toward *dog*. Only positive-D
concepts are the canonical case that invariance-based reasoning implicitly assumes.

The practical consequence is that an objective which equalises activation
statistics across domains cannot distinguish these cases, because all three are
equally invariant by construction.

*[Figure 1: `results/analysis/fig_h_vs_d.png` — H(k,c) against D(k,c), coloured by
class, support floor applied.]*

---

## 5.3 Consistency of discriminative effect is a second, asymmetric axis

Activation invariance asks *where a concept appears*. It cannot ask where the
concept's *effect* appears, and these turn out to be different questions. R(k,c),
the normalised entropy of |D_d(k,c)| across source domains, measures the second.

**R is not a rival measure of invariance; it is a filter inside the invariant
population.** Both scores are entropies over three source domains, so a pair
clearing τ_R = 0.7 must have nonzero effect in all three: an even split over only
two domains gives log 2 / log 3 = 0.63, below threshold. The same bound applies to
H, so high consistency implies broad activation. The data confirm it — only **8 of
273** non-neutral pairs are low-H but high-R.

**But it is a filter that removes a great deal.** Of the 1173 high-H pairs, only
**595 (50.7%)** also clear τ_R. Knowing that a concept is activation-invariant
tells us almost nothing about whether its effect is consistent.

**The central observation of this section is an asymmetry.** Splitting the
invariant population by the sign of D:

**Table 2.** Among invariant (H ≥ τ) pairs with non-negligible effect, the fraction
whose effect is also consistent (R ≥ τ).

| τ_H = τ_R | invariant supportive also consistent | invariant harmful also consistent | ratio |
|---|---|---|---|
| 0.7 | 72.4% (76/105) | 36.0% (32/89) | 2.01× |
| 0.8 | 65.2% (60/92) | 27.3% (18/66) | 2.39× |
| 0.9 | 49.4% (39/79) | 15.4% (6/39) | 3.21× |

Invariant support is usually spread across domains; invariant harm is usually
concentrated in one or two. The effect strengthens monotonically as the threshold
tightens, so the qualitative conclusion does not depend on a particular τ. This
matters for how the harmful-invariant category should be read: most of those
concepts are not harmful everywhere. They fire in every source domain and do their
damage in a subset. An alignment objective that equalises activation statistics
cannot detect this, because their activations are already balanced — the asymmetry
that makes them harmful is invisible to activation-level criteria.

**R must be read behind a magnitude gate.** Entropy is scale-insensitive, so a
concept with negligible effect spread evenly across domains scores as highly
consistent. **487 of 1269 neutral pairs (38.4%)** fall in the high-H, high-R cell.
This is why we exclude |D| ≤ τ_D before interpreting R, and why the typology in
Table 3 is defined only over the 273 non-neutral pairs.

*[Figure 2: `results/analysis/fig_hr_heatmap.png` — H×R contingency at τ = 0.7,
panelled by D regime. The neutral panel is what motivates the magnitude gate.]*

**What this section does and does not claim.** The asymmetry is a structural
property of the representation, established from the scores alone. We do not claim
that R carries information independent of the sign and magnitude of D: the harmful
population is enriched roughly 2× in low-R pairs relative to the supportive
population, so the two are correlated in this model, and separating them would
require a mass-matched intervention we do not perform. §5.4 is therefore presented
as validation that the categories correspond to functional behaviour, not as
evidence that consistency is causally prior to sign.

---

## 5.4 The typology corresponds to functional prediction behaviour

We next test whether the diagnostic categories describe real model behaviour, by
ablating each bucket in SAE code space and re-evaluating. For an image with true
label y, the concepts masked are those paired with y in the bucket. These are
**oracle interventions**: they use the ground-truth label, and they are not
deployable test-time methods. Their purpose is the same as any controlled ablation
— if removing a category changes prediction behaviour in the direction the
diagnostic predicts, the category captures something functional.

Two facts must frame every row. First, **the direction of the sign effect is
tautological**: D is defined as the drop in p(y) under ablation, so masking
negative-D pairs must raise p(y). Only the magnitude, the source-to-target
asymmetry, and the margin over a matched control carry information. Second, we
therefore pair every intervention with a **size-matched, |D|-histogram-matched
random control** drawn from the support-floored pairs the target did not select
(three seeds). Without it, the objection that ablating any comparable set of
concepts helps a weak domain would be unanswerable.

**Table 3.** Concept interventions. Δ is against the SAE-reconstruction baseline
(sketch 83.63 macro, 80.22 micro). Control is the mean of three |D|-matched random
masks of identical size; margin is target Δ minus control Δ on sketch macro.

| Intervention | pairs masked | Σ\|D\| masked | sketch macro | Δ | sketch micro | Δ | control Δ | margin |
|---|---|---|---|---|---|---|---|---|
| Baseline (SAE recon.) | 0 | 0 | 83.63 | — | 80.22 | — | — | — |
| Mask all harmful | 138 | 0.250 | 90.87 | +7.24 | 89.23 | +9.01 | −20.98 | **+28.2** |
| Mask `S−_hi` (distributed harm) | 35 | 0.087 | 87.32 | +3.69 | 85.19 | +4.97 | −3.12 | +6.8 |
| Mask harmful-invariant | 32 | 0.083 | 86.96 | +3.33 | 84.70 | +4.48 | −2.10 | +5.4 |
| Mask `S−_lo` (concentrated harm) | 103 | 0.163 | 86.17 | +2.54 | 82.87 | +2.65 | −5.40 | +7.9 |
| Mask inert pairs only | 111 530 | 0.069 | 83.71 | +0.08 | 80.33 | +0.11 | — | — |
| Mask `S+_lo` (concentrated support) | 54 | 0.025 | 82.98 | −0.65 | 79.56 | −0.66 | −1.07 | +0.4 |
| Mask `S+_hi` (distributed support) | 81 | 0.237 | 44.55 | −39.08 | 39.17 | −41.05 | +1.06 | −40.1 |
| Mask all supportive | 135 | 0.262 | 23.63 | −60.00 | 18.53 | −61.69 | +2.50 | −62.5 |

Four results.

**(i) The categories are functional, and the controls are decisive.** Masking the
138 harmful pairs raises sketch accuracy by 7.24 points macro. Masking 138 random
pairs with a closely matched effect-mass profile (Σ|D| 0.245 against the target's
0.250) *lowers* it by 20.98. The two go in opposite directions, a 28-point margin.
Every harmful bucket beats its control by 5–8 points; every supportive bucket loses
to its control by 40–63. Ablation of a comparable set of concepts does not, in
general, help the weak domain.

**(ii) The gains are not dictionary denoising.** 111,530 of 114,688 pairs have zero
measured effect in every source domain. Masking all of them moves sketch by
**+0.08** macro — nothing — despite their aggregate |D| mass (0.069) exceeding that
of the distributed-harm bucket. Whatever the interventions are doing, they are not
removing inert latents.

**(iii) The source-to-target asymmetry.** Every intervention changes the three
source domains by well under one point (source accuracies stay in 99.3–100.0),
while sketch moves by up to 60. The diagnostic is estimated only on source
domains, yet the behaviour it predicts appears almost entirely in the domain it
never saw.

**(iv) Distributed support, not concentrated support, is what the model runs on.**
Masking `S+_hi` (81 pairs) costs 39 points; masking `S+_lo` (54 pairs) costs 0.65
and is statistically indistinguishable from its random control (−0.65 against
−1.07). This converges with §5.3's asymmetry from the functional side.

### Sufficiency of robust support

We also invert the mask, retaining only one bucket and ablating everything else.

**Table 4.** Keep-only interventions. All retain concepts paired with the true
label, so all inject label information; the random row measures how much of the
effect that injection accounts for.

| Keep only | pairs kept | kept Σ\|D\| | sketch macro | sketch micro |
|---|---|---|---|---|
| — (baseline) | — | — | 83.63 | 80.22 |
| 76 uniformly random pairs (3 seeds) | 76 | ≈0 | 14.29 / 14.25 / 14.82 | 4.07 / 4.05 / 4.84 |
| `S+_lo` (concentrated support) | 54 | 0.025 | 25.00 | 18.25 |
| Robust support (H,R ≥ τ, D > τ_D) | 76 | 0.236 | 94.02 | 93.56 |
| All supportive | 135 | 0.262 | 95.00 | 94.25 |

Retaining only the 76 robust-support pairs and ablating the other 114,612 yields
**94.02% macro / 93.56% micro on sketch**, above the unablated model, with source
domains at 99.85–100.0. Two comparisons make this interpretable rather than
circular.

First, keeping 76 *arbitrary* pairs, per-class size-matched, collapses the model to
**exactly chance** (14.29% macro; every prediction becomes *person*). So the
accuracy is a property of the retained bucket, not of the label used to build the
mask — a degenerate reconstruction does not recover the label.

Second, the three keep-only rows share the same label-injection component, so
differences between them are informative. Concentrated support alone (54 pairs)
collapses to 25.00%. Robust support alone reaches 94.02%. Adding the concentrated
pairs on top of robust support moves it only to 95.00%, a gain of 0.98 points. The
model's class evidence is carried almost entirely by invariant, distributed
supportive concepts; domain-contingent support is close to redundant.

**Limits of this comparison.** Robust support holds roughly 10× the effect mass of
`S+_lo` (0.236 against 0.025). The keep-only rows therefore conflate consistency
with effect magnitude, and we do not claim that distributed support is sufficient
*because* it is distributed rather than because it is larger. Separating those
would require mass-matched bucket comparisons, which we leave to future work. What
the rows do establish is that the categories track functional behaviour, and that a
small, source-estimated, invariant-and-consistent subset of the dictionary is
sufficient to classify the unseen domain.

---

## 5.5 Domain-contingent effects concentrate in the stylised source domains

Because R is an entropy over per-domain effects, for every low-R pair we can ask
*which* source domain carries the effect: `argmax_d |D_d(k,c)|`. This requires no
intervention.

A raw table of those counts is confounded, because a domain in which all effects
are systematically larger will win the argmax regardless of concentration. Mean
|D_d| over non-neutral pairs is indeed uneven — 2.13e-3 in art_painting, 1.92e-3 in
cartoon, 1.15e-3 in photo — and active-image exposure differs too (26.6k, 35.4k,
19.9k). We therefore report the low-R distribution against the corresponding
distribution for **high-R** pairs, which are by construction not concentrated and
so serve as the null.

**Table 5.** Domain carrying the largest per-domain effect. Low-R (concentrated)
pairs against the high-R null, as a share of each row.

| | art_painting | cartoon | photo |
|---|---|---|---|
| Concentrated supportive (n=54) | 48.1% | 46.3% | 5.6% |
| Distributed supportive, null (n=81) | 58.0% | 19.8% | 22.2% |
| *ratio* | 0.83× | **2.34×** | **0.25×** |
| Concentrated harmful (n=103) | 37.9% | 55.3% | 6.8% |
| Distributed harmful, null (n=35) | 22.9% | 54.3% | 22.9% |
| *ratio* | **1.66×** | 1.02× | **0.30×** |

Three readings survive the null comparison. Concentrated *support* is enriched 2.3×
in cartoon. Concentrated *harm* is enriched 1.7× in art_painting. And photo is
depleted 3–4× in both: effects that concentrate almost never concentrate in photo,
though part of that is mechanical, since photo has both the smallest mean effect
and the lowest exposure.

Cartoon's dominance of the harmful column (55.3%) is *not* a concentration effect —
it is 54.3% in the null as well. Cartoon supplies most harmful effect in this model
whether or not that effect is domain-contingent, and only the supportive column
shows genuine cartoon-specific concentration.

The pattern is that domain-contingent effect lives in the stylised source domains
rather than in photographs. Since the held-out target is itself a stylised, texture-
poor domain, the concepts carrying domain-contingent effect are the style-sensitive
ones — which is consistent with their effect failing to transfer to a *different*
style. We note this as a coherent reading rather than a demonstrated mechanism;
establishing it would require intervening per-domain, which we do not do.

*[Table also at `results/analysis/domain_attribution.csv`, per class as well as
aggregate.]*

---

## 5.6 Concepts are not class-isolated

Because D is class-conditional, the same latent can support one class and hurt
another. This is the failure mode that class-level scores cannot express, and it is
the reason all three of our scores are defined per (class, concept) pair rather
than per concept.

We call a concept **conflicting** if it has D > τ_D for at least one class and
D < −τ_D for at least one other, both above the support floor. Note that no
co-activity assumption is needed: D(k,c) is accumulated only over images of class k
on which c actually fires, so a conflict certifies that the concept fires on images
of both classes.

Of the 533 concepts with any above-floor class, 139 are non-neutral for at least
one class — 113 supportive for some class, 82 harmful for some class. **56 of those
139 (40.3%) are conflicting.** Two-fifths of the model's discriminatively active
concepts are shared in a way that interferes with class separation.

**Table 6.** Per-class diagnostic counts. Support is the number of pairs above the
floor; the last column counts concepts that harm this class while supporting
another.

| Class | Sketch acc. (micro) | Above floor | Supportive | Harmful | Neutral | Harmed by conflicting |
|---|---|---|---|---|---|---|
| dog | 48.06 | 286 | 33 | 50 | 203 | 34 |
| elephant | 95.54 | 208 | 23 | 16 | 169 | 15 |
| giraffe | 76.89 | 185 | 15 | 10 | 160 | 10 |
| guitar | 97.37 | 150 | 12 | 3 | 135 | 2 |
| horse | 83.21 | 243 | 24 | 24 | 195 | 18 |
| house | 88.75 | 144 | 9 | 3 | 132 | 3 |
| person | 95.62 | 326 | 19 | 32 | 275 | 26 |

The class-level picture is suggestive but not monotone. `dog` is the weakest class
on the target domain (48.06% micro) and carries both the most harmful pairs (50)
and the most conflicting concepts harming it (34); `guitar` and `house` are near
the top and carry three each. But `person` has 32 harmful pairs and 26 conflicting
concepts while scoring 95.62%, so harmful mass alone does not predict per-class
failure. With seven classes we do not attempt a correlation, and we report these as
descriptive profiles rather than a selection criterion.

Ranking conflicting concepts by contrast `max_k D(k,c) − min_k D(k,c)` gives the
clearest individual cases. Concept 4501 supports *person* and harms *dog*; concept
2979 supports *horse* and harms *dog*; concept 3128 supports *dog* and harms
*elephant*. Inspecting concept 2979's top-activating images across domains shows it
firing on the chest and forelimb region of both horses and dogs — a genuine
visual similarity that the representation has not separated, and which costs the
weaker class.

*[Figure 3: existing grid `analysis/ERM_ResNet_T3/HIPD/4501 - Pos Human Neg All/`
for the person/dog conflict; `analysis/ERM_ResNet_T3/HIND/1637 - Negative Dog Pos
Horse/` for the horse/dog chest concept. Scores to be re-quoted from the clean
file before use — the folder names come from the superseded score file.]*

---
---

# PATCH LIST for `TMLR_Journal_Submissions__1_.md`

## Structural changes

| Existing § | Action |
|---|---|
| 5.1 SAE reconstructions preserve classifier behavior | **Replace** with §5.1 above. Table 2 → new Table 1, add micro/macro columns. |
| 5.2 Activation-invariant concepts are not necessarily useful | **Replace** with §5.2 above. Keeps the same argument; all counts updated. |
| 5.3 Discriminative consistency matters beyond activation invariance | **Replace** with §5.3 above. Removes the R-threshold sweep, Fig. 4 and old Table 3 entirely; adds the explicit no-causal-claim paragraph. |
| 5.4 The concept typology reveals distinct class-level failure modes | **Merge into 5.6.** Its per-class content becomes new Table 6; the empty Table 4 is deleted. |
| 5.5 Harmful invariant concepts demonstrate why invariance is not enough | **Repurpose** as new §5.5 (domain attribution). Its qualitative-example role moves to §5.6. |
| 5.6 Class-conflict profiles reveal lack of class isolation | **Replace** with §5.6 above. |
| 5.7 Model or checkpoint profiles | **Cut.** Single checkpoint analysed. Add one sentence to Limitations. |
| 5.8 Oracle interventions validate the diagnostic categories | **Dissolve into new §5.4.** Table 6 (interventions) → new Table 3 + Table 4. |

Net: eight subsections → six. §5.4 and §5.8 were the same result told twice.

## Numbers in the draft that are now wrong

| Location | Currently says | Replace with |
|---|---|---|
| §5.1 / Table 2 | sketch 80.29 → 80.20, drop 0.10 | 80.25 → 80.22, drop 0.03 |
| §5.2 | "7.2% positive-D, 85.6% neutral, 7.2% negative-D" | 9.0% / 83.5% / 7.6% (of 1173 high-H pairs) |
| §5.3 | "only 3 of the 217 pairs" low-H high-R | 8 of 273 |
| §5.3 | "Of the 1068 high-H pairs, only 46.4%" | Of 1173 high-H pairs, 50.7% |
| §5.3 | "72.5% (74/102)" and "29.7% (19/64)" | 72.4% (76/105) and 36.0% (32/89) |
| §5.3 | "35.2% of near-zero-D pairs" high-H high-R | 38.4% (487/1269) |
| §5.3 / Fig. 4 / Table 3 | sweep peaking +7.67 at τ_R = 0.8; "+3.35 at τ_R = 0.1" | **Delete.** Sweep not rerun. The τ=0.1 gain was the 349 signed-R sentinels, not inert latents; B4 shows inert masking gives +0.08. |
| §5.3 / Table 3 | "+ rare-concept mask (n<30)" row at sketch 83.40 | **Delete the row.** That filter leaks labels (CLAUDE.md §3) and is no longer used anywhere. |
| §4.2 | SAE "trained and analyzed exclusively on the source domains" | Soften: the normalizer's two scalar statistics were estimated from a batch spanning all four domains. Everything else is source-only. |
| §4.3 | τ_H, τ_R "[TODO: 0.8 or final value]", τ_D "[TODO: 1e-3 or value]" | τ_H = τ_R = 0.7, τ_D = 1e-4. Note the log2/log3 = 0.63 bound holds only at 0.7. |
| §3.6 | optional sign-consistency score S | **Cut**, or mark explicitly unused. Not computed in the final pipeline. |
| §3.8 | RSM / HIM / DCM / conflict-mass definitions | Keep the conflict definition (used in §5.6). Mark RSM/HIM/DCM as future work — no results depend on them. |
| Abstract | "oracle concept interventions validate that the identified categories correspond to functional prediction behavior" | Accurate, keep. Optionally add that they are validated against size- and mass-matched random controls. |

## Claims to retire

1. **"Low-R concepts are not junk; the model's class evidence substantially
   consists of domain-contingent concepts."** (the deleted handoff brief §2, and
   `CLAUDE.md` §1 before it was corrected.)
   Refuted. Concentrated support is indistinguishable from random when masked
   (−0.65 vs −1.07) and collapses to 25.00% when kept alone, while distributed
   support alone reaches 94.02%. The replacement claim is the opposite and is
   stronger.
2. **"Masking harmful concentrated concepts (`R < 0.8 ∧ D < 0`) gives +10.03."**
   The *number* reproduces; the *label* is wrong. That row masked 114,609 pairs —
   it was `keep only {R ≥ 0.7 ∧ D > 1e-4}`, i.e. keep-only distributed support,
   mislabelled by the `filter`-keeps / `get_mask`-complements inversion. Clean
   equivalent: 94.02 (Table 4) against the old 93.47. Retire the label, keep the
   result, and move it from the mask table to the keep-only table.
   Separately: a genuine, gated, support-floored mask of concentrated harm
   (`S−_lo`, 103 pairs) gives +2.54, while `S−_hi` gives +3.69 from a third of the
   pairs and half the mass — so per unit of effect mass, distributed harm is the
   more damaging category. That comparison is new; nothing in the old tables
   addressed it.
3. **"Keep robust support only → sketch 46.6% macro with three classes at 0.00."**
   Archive artifact, computed over four domains including the target. Clean value
   is 94.02%.
4. **Harmful concentrated concepts peak in `photo` (photorealistic interference).**
   Never tested before; now tested and false. Photo is the *least* likely peak
   domain (0.25–0.30× the null).

## Still open, and deliberately so

- Mass-matched bucket comparisons (would separate R from |D|). Named as future
  work in §5.4.
- Label-free variants. Both previously tried were null; worth one sentence in
  Limitations rather than a rerun.
- τ_D = 1e-3 sensitivity. `analyze_scores.py --tau-d 1e-3` regenerates every
  §5.2/5.3 number in about a minute if a reviewer asks.
- Additional SAE seeds and additional checkpoints. Limitation paragraph. Worth
  noting in §4.2 that the tied multi-checkpoint USAE aligns dictionaries across
  steps 2100 and 3300 by construction, which is a partial substitute and is
  currently undocumented.
