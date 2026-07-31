# Discussion — what the diagnostic reveals

Built from the experiments already run. The argument here is *interpretive*: it
works from what individual concepts look like and what they do, using the
statistics as corroboration rather than as the case itself.

Figure slots are marked **[FIG-n]** and specified in §11. Concepts named below with
an id (e.g. concept 2979) are inspectable — `scripts/find_archetypes.py` locates and
renders them.

Companion: `docs/DIRECTIONS.md` for claims needing further investigation.

---

## 0. The punchline, in one paragraph

An SAE turns the model's final feature map into roughly 16,000 individual
detectors. For each detector and each class we ask three things: does it fire in
every domain, does switching it off help or hurt the right answer, and does that
help-or-hurt travel across domains. Doing this reveals that **the concepts most
damaging to out-of-distribution performance are not broken or spurious detectors.
They are good detectors wired to the wrong class** — a chest-and-forelimb detector
that is correct evidence for `dog` and, on a horse, becomes an argument for `dog`
anyway. 78% of harmful class–concept pairs are of exactly this kind. Such concepts
fire evenly across all domains, so every activation-alignment criterion in the DG
literature scores them as ideal. Thirty-five of them account for a quarter of this
model's target-domain errors.

---

## 1. What the three scores mean, one concept at a time

### H — does it fire everywhere?

`H(k,c)` is the entropy of concept `c`'s mean activation across the three source
domains, for images of class `k`. High H means the detector fires as readily on a
cartoon dog as on a photographed one.

The interpretable case is a **body-plan detector**: a concept that responds to the
four-legged animal silhouette — legs, torso, stance. Legs are legs whether painted,
drawn or photographed, so H is near 1. **[FIG-1]**

This is the score every invariance-based DG method is implicitly maximising. And on
its own it is close to worthless, for a reason that becomes obvious once you look at
the concept: the legs detector fires on dogs, horses *and* elephants. It is a
perfectly stable, perfectly reliable detector of something that does not distinguish
any of the classes we care about.

Stability is not usefulness. To see usefulness you have to intervene.

### D — does switching it off change the answer?

`D(k,c)` zeroes the concept in SAE code space, decodes, and measures the change in
the model's probability for the true class, averaged over images of class `k` where
the concept actually fires.

- `D > 0`: removing it **lowered** confidence in the right answer → it was helping.
- `D ≈ 0`: removing it changed nothing → it was not being used to discriminate.
- `D < 0`: removing it **raised** confidence in the right answer → it was arguing
  against the right answer.

Run this on the legs detector and you get `D ≈ 0` for dog, horse and elephant alike.
The model was not using it to tell them apart, because it *cannot*. This is the
single most common outcome in the whole dictionary: **83.5% of invariant concepts
are discriminatively inert.** Five-sixths of what an alignment objective is
stabilising does not separate classes at all.

### Negative D, made concrete

Now the case that matters. Consider a detector for the **chest and forelimb region**
— the front of a four-legged mammal seen side-on. **[FIG-2]**

On a dog, this is genuine evidence: the model uses it to say `dog`, and switching it
off makes the model less sure. `D(dog, c) > 0`.

On a horse, the same detector fires — a horse has the same region and it looks
similar, especially in the flatter, lower-texture renderings of cartoons and
paintings. But the model has learned to read this pattern as dog-evidence. So on a
horse it pushes probability *away* from `horse`. Switch it off and the model becomes
*more* confident the horse is a horse. `D(horse, c) < 0`.

**One detector. Two classes. Opposite signs.** Nothing about the detector is
defective: it detects exactly what it detects, reliably, in every domain. The defect
is in the mapping from that detector to a class decision.

This is why every score is defined per **(class, concept)** pair rather than per
concept. A single global score for this concept would average a positive number for
dog against a negative number for horse and report approximately zero — filing a
genuine failure mechanism into the inert 83%.

### R — does the usefulness travel?

`H` asks whether the concept *fires* everywhere. `R` asks whether it *matters*
everywhere: it is the entropy of |D| computed separately per domain.

These come apart, and the distinction is the crux of the framework:

```
                      fires in all 3 domains?
                          no            yes
                     +-------------+-------------+
  matters in all  no |  local cue  |  fires everywhere
  3 domains?         |             |  but only decides
                     |             |  in one domain
                     +-------------+-------------+
                 yes |  (nearly    |  fires everywhere
                     |   empty:    |  AND decides
                     |   8 of 273) |  everywhere
                     +-------------+-------------+
```

The interesting cell is top-right: a concept can be perfectly invariant in
activation while its *influence* is concentrated in a single domain. An outline or
line-weight detector might fire on every image — photographs have edges too — yet
only carry decision weight where outlines are the dominant cue, i.e. in cartoons.
H sees a textbook invariant concept; R sees a cartoon specialist. **[FIG-3]**

The bottom-left cell is nearly empty (8 of 273 non-neutral pairs), which is a
sanity check rather than a finding: with three domains, consistent influence
requires nonzero influence everywhere, which requires firing everywhere. R filters
*within* the invariant population; it does not compete with H.

**R needs a magnitude gate.** Entropy ignores scale, so a concept with negligible
influence spread evenly scores as perfectly consistent — 38.4% of *neutral* pairs
land in the high-H/high-R cell. Consistency of a non-effect is meaningless, which is
why R is only read for pairs with |D| > τ_D.

---

## 2. Four kinds of concept

The three scores partition the dictionary into recognisable characters.

| Archetype | Scores | Count | What it looks like | Consequence |
|---|---|---|---|---|
| **Reliable bystander** | H high, \|D\| ≈ 0 | 979 of 1173 invariant pairs (83.5%) | Body plan, legs, generic edges | None. Ablating all 111,530 inert pairs moves target accuracy **+0.08pp** |
| **Robust witness** | H, R high, D > 0 | 76 | Class-specific structure that survives restyling | Retaining *only* these gives **94.0%** on the unseen domain |
| **Consistent saboteur** | H, R high, D < 0 | 32 | A robust witness *for a different class* | 32 pairs cause **22.6% of all target errors** |
| **Local specialist** | R low | 54 support / 103 harm | Cue tied to one domain's rendering style | Support version is surplus (−0.65pp, = random). Harm version costs **13.4%** of errors |

Two of these are counterintuitive and both are load-bearing.

**The bystander is the majority.** The overwhelming outcome of asking "is this
concept invariant?" is *yes, and it doesn't matter*. That is the empirical content
of "invariance is not enough."

**The local specialist behaves asymmetrically by sign.** Domain-specific *support*
is surplus — masking it is indistinguishable from masking a random matched set, and
retaining it alone collapses the model to 25%. Domain-specific *harm* is not
surplus. The model can afford redundant style-specific evidence; it cannot afford
style-specific interference. Supportive concepts back each other up; each harmful
concept is an independent chance to flip a prediction the wrong way.

---

## 3. The unifying mechanism: harm is misdirected support

Why should a trained model contain harmful concepts at all? Every concept exists
because it was useful somewhere. The chest example answers this, and the data says
the example is the rule rather than an anecdote.

A concept is **conflicting** if it supports at least one class and harms at least
another. Measured on the clean scores:

- **56 of 139** discriminatively active concepts (40.3%) are conflicting.
- **108 of 138** harmful class–concept pairs (**78.3%**) involve a concept that
  supports some other class.

```
                     concept 2979  (one detector)
                            |
          +-----------------+------------------+
          |                                    |
     row: horse                            row: dog
     D > 0  supports                      D < 0  harms
     "this is a horse"                    "...but so does this dog"
          |                                    |
     robust witness                      consistent saboteur
```

So **negative D is, four times out of five, not a spurious feature — it is a real
feature attached to the wrong class.** The harm arises because the two classes share
visual structure that the representation never separated, and in-distribution some
*other* cue does the disambiguating. Under domain shift the disambiguating cue
weakens, the shared detector still fires, and it votes wrong.

The clean data's most conflicted concepts, ranked by `max_k D − min_k D`:

| Concept | Supports | Harms | Contrast |
|---|---|---|---|
| 52 | giraffe | person | 0.070 |
| 3128 | dog | elephant | 0.063 |
| 2979 | horse | dog | 0.053 |
| 4501 | person | dog | 0.031 |
| 3589 | elephant | dog | 0.021 |

**[FIG-4]** — concept 2979 or 4501 rendered across all seven classes, which shows
the same detector highlighting the same body region on two different animals.

This reframes what a fix would look like. "Remove the spurious feature" is the wrong
instruction, because the feature is not spurious. The problem is **class isolation**:
the same evidence needs to be read differently depending on what else is present.
That is a statement about the read-out, not about the features.

---

## 4. Why invariance criteria cannot see the saboteur

Put the two previous sections together and the paper's central claim becomes almost
mechanical.

The consistent saboteur, by construction:

- fires in **all three** source domains (H ≥ 0.7 forces this — an even split over
  only two domains gives log 2/log 3 = 0.63, below threshold);
- has nonzero discriminative influence in **all three** (R ≥ 0.7 forces the same);
- and is therefore **maximally well-behaved** under any criterion that compares
  feature distributions across domains.

A domain-adversarial or MMD-style penalty has *no gradient to apply* to such a
concept. Its activation statistics are already aligned; that is what makes it
qualify. Its pathology lives entirely in the sign of its effect on the class
posterior, which no activation-level statistic measures.

We can price what invariance does buy. Conditioning on H ≥ 0.7 moves the
probability that a concept is harmful from 13.3% (low-H) to 7.6% (high-H) — it
nearly halves harm *incidence*, which is a genuine if modest benefit. But it moves
the probability that a concept is *supportive* from 8.1% to 9.0%, i.e. not at all.

**Invariance is weakly informative about the absence of harm and completely
uninformative about the presence of usefulness.** And the harm it fails to remove is
disproportionately the harm that matters (§5).

---

## 5. What the saboteurs cost

Expressing each intervention as the share of the model's 19.78pp target-domain error
(micro) that it recovers turns effect sizes into something interpretable:

| Intervention | pairs | % of grid | errors recovered | per pair |
|---|---|---|---|---|
| Mask all harmful | 138 | 0.120% | 45.6% | 0.33% |
| Mask consistent saboteurs (`S−_hi`) | 35 | 0.031% | **25.1%** | **0.72%** |
| Mask harmful-invariant (H, R ≥ 0.7) | 32 | 0.028% | 22.6% | 0.71% |
| Mask local harmful specialists (`S−_lo`) | 103 | 0.090% | 13.4% | 0.13% |
| Mask all 111,530 inert pairs | 111,530 | 97.2% | 0.6% | ~0 |

**Nearly half of this model's residual error on an unseen domain is attributable to
138 identifiable class–concept pairs**, and a quarter of it to 35 — the 35 that
score best on invariance.

Per pair, a consistent saboteur is **5.5× more damaging** than a local harmful
specialist (0.72% vs 0.13% of errors each). Concepts that hurt everywhere are far
more expensive than concepts that hurt in one style.

The inert row is the control that makes the rest meaningful: 111,530 pairs, 97% of
the grid, aggregate |D| mass (0.069) comparable to the 35 saboteurs' (0.087), and
removing all of them recovers 0.6% of errors. **Aggregate effect mass is not
importance** — influence spread thinly over a hundred thousand pairs cannot move an
argmax; the same mass concentrated in 35 pairs flips predictions. This is also why
the right control is one matching the *distribution* of per-pair magnitudes, not the
total.

---

## 6. The model already knows enough

The standard reading of a DG failure is a deficit — the model never learned features
that transfer. The keep-only experiment says that reading is incomplete.

Retain only the 76 robust witnesses and ablate the other 114,612 pairs — **99.93% of
the grid**:

| | art_painting | cartoon | photo | sketch |
|---|---|---|---|---|
| baseline (macro) | 99.35 | 99.19 | 99.78 | 83.63 |
| keep robust witnesses only | 99.85 | 99.88 | 100.00 | **94.02** |

Every domain improves. Sketch gains 10.4 macro / 13.3 micro, recovering 67% of its
errors. **The frozen model already contains a concept subset sufficient for 94% on a
domain it never saw**, and that subset is identifiable from source-domain statistics
alone. It scores 83.6% because its read-out also incorporates everything else.

The failure is one of **selectivity, not capacity.**

Two caveats keep this honest. It is an **oracle bound**: the *set* was found from
source data, but addressing the mask requires each image's true label, so this
measures what a perfect per-image selector could extract — analogous to oracle model
selection in DomainBed, and not a method. And it is non-vacuous only because of the
control: retaining 76 *arbitrary* pairs collapses the model to exactly chance
(14.29%, every prediction `person`). The accuracy reflects which concepts were kept,
not the label used to look them up.

---

## 7. In-distribution redundancy hides out-of-distribution fragility

Across every intervention, the three source domains move by under one point while
sketch moves by up to 60. Ablating 99.93% of the grid leaves source accuracy
*unchanged or better*.

In distribution the model has so much redundant evidence that deleting almost all of
it changes no argmax. On the target domain the same deletion is worth ten points.
This points at a failure mode about **margin** rather than features: in distribution
the model sits far from its decision boundary and every concept is individually
dispensable; under shift it sits near the boundary and the same concepts become
individually decisive.

Corroboration from the score distributions: mean per-domain effect magnitude is
1.15e-3 in photo (99.78% accuracy), 1.92e-3 in cartoon (99.19%), 2.13e-3 in
art_painting (99.35%). The easiest domain is the one where ablating a single concept
perturbs the posterior least — the most redundant. Three points, one confounded by
exposure, so this is suggestive; `DIRECTIONS.md` §D3 gives a leak-free measurement.

---

## 8. Per-class vulnerability has two ingredients

| Class | Sketch acc. | Harmful pairs | Harmed by conflicting concepts |
|---|---|---|---|
| dog | 48.06 | 50 | 34 |
| giraffe | 76.89 | 10 | 10 |
| horse | 83.21 | 24 | 18 |
| house | 88.75 | 3 | 3 |
| elephant | 95.54 | 16 | 15 |
| person | 95.62 | 32 | 26 |
| guitar | 97.37 | 3 | 2 |

`dog` is the weakest class and carries the heaviest conflict burden; `guitar` is the
strongest and carries almost none. But `person` carries 26 conflicting concepts
against it and still scores 95.6%, so burden alone does not predict failure.

The resolution is visible in the collapse behaviour: under *every* degenerate mask we
ran, surviving predictions pile onto `person`, up to 100%. **`person` is the model's
default class — it wins ties.** It also has the most pairs above the support floor
(326). So it carries a heavy conflict burden and is insulated from it by a prior
advantage. `dog` carries the heaviest burden with no insulation, and loses.

Class robustness therefore depends on at least two things: **how much conflicting
evidence points at the class, and whether the class wins or loses the resulting
ties.** This is a caution against the one-dimensional per-class profile scores the
draft's §3.8 proposes — they cannot express it. `giraffe` (10 conflicts, 76.9%)
remains unexplained by either ingredient.

---

## 9. What a practitioner gets

From a frozen model and source-domain data only:

1. **An error attribution.** "45.6% of the residual error on the unseen domain is
   attributable to 138 identifiable pairs, 25.1% to 35 of them."
2. **A capacity verdict.** "This representation contains a subset sufficient for
   93.6% on the unseen domain" — telling you the problem is the read-out, not the
   features, which implies a different remedy.
3. **Named, inspectable failure sites.** Concept 2979 harming `dog` through shared
   chest geometry is actionable in a way "sketch accuracy is 80%" is not.
4. **A warning about the metric being optimised.** The most damaging bucket is the
   one that scores best on activation invariance.

None of this is visible from target-domain accuracy, and the scores need no
target-domain labels.

---

## 10. What we deliberately do not claim

- **Not that R carries information independent of D's sign and magnitude.** Harmful
  pairs are ~2× enriched in low-R, and the keep-only comparison conflates
  consistency with mass (robust witnesses hold ~10× the local specialists' mass).
  §5.3 is descriptive for this reason.
- **Not a deployable intervention.** Every mask uses the true label to select its
  row; the label-free variants tried so far are null.
- **Not causal identification.** Ablations act in SAE reconstruction space, not the
  original network.
- **Not that latents are semantic units.** They are candidate concepts. Visual
  evidence is a qualitative aid; the quantitative claims rest on reconstruction
  fidelity and ablation behaviour.
- **Not external validity.** One dataset, one algorithm, one checkpoint, one SAE
  seed. The mechanism may be general; we have shown it in one place.

---

## 11. Figure manifest

The argument above depends on visual evidence at four points. The SAE is unchanged
(`USAE_ERM_Multi_test_3300_2100.pt`), so **existing renders remain valid as
images** — a latent detects what it detects regardless of which score file we read.
Only the H/D/R values quoted in captions must be re-taken from
`FINAL_ERM_ResNet_3300_T3.json`.

| Slot | Purpose | Status |
|---|---|---|
| **FIG-1** | Reliable bystander: invariant, fires across several animal classes, D ≈ 0. Establishes "stable but useless". | Draft's existing Fig. 3 ("near zero discrimination on the 4 classes above") is this. Candidate: 1893 (labelled *Lines*), 1705, 15483. **Re-quote scores; confirm identity.** |
| **FIG-2** | The chest/forelimb conflict: same concept, positive for one animal, negative for another, shown side by side. **The most important figure in the paper.** | Grids exist for 1637 (`analysis/ERM_ResNet_T3/HIND/1637 - Negative Dog Pos Horse/`, all 7 classes) and 4409. Clean-file equivalents: 2979 (+horse/−dog), 4501 (+person/−dog). **Pick whichever has the cleanest visual and verify its clean scores.** |
| **FIG-3** | High H, low R: fires in every domain but only decides in one. Motivates R as separate from H. | **Does not exist.** Needs rendering. `find_archetypes.py` locates candidates. |
| **FIG-4** | The conflict panel: one concept across all seven classes, showing which body region it locks onto. | Partially exists (1637, 4501 grids are per-class already). Assemble into one panel. |

Also worth fixing while doing this: the draft's §5.2 text refers to "a legs concept
(Fig. 2)" and "a dog-chest concept (Fig. 3)", but the captions are the other way
round — Fig. 2 is captioned as the positive-dog/negative-horse concept and Fig. 3 as
the near-zero one. **The figure references are swapped.**

`scripts/find_archetypes.py` prints, for each archetype, the top candidate concepts
with their full per-class H/D/R profiles, and with `--render` writes the grids via
`visualize_concept_on_class`. That is the missing step: the framework's whole
advantage is that these categories are lookable-at, and so far we have only counted
them.
