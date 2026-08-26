# Critique and revision brief — `docs/TMLR_Journal_Submissions__1_.md`

**Reviewer pass date:** 2026-08-01
**Scope:** storyline, structure, logical flow, redundancy, clarity, figures/tables,
internal consistency. **No new experiments, ablations or data collection are
proposed.** Everything below is achievable with the numbers, figures and score
files that already exist.

**Source of truth:** the markdown draft. LaTeX will be built from it later, so all
edits happen in `docs/TMLR_Journal_Submissions__1_.md`.

**Figure assets live in two places:** `docs/figures/` and `results/analysis/`.

**How to use this document.** Work top-down. P0 items are correctness problems and
must be resolved before the draft circulates. P1 is the structural rewrite. P2 is
figures and tables. P3 is polish. Each item has a checkbox; tick as you go and note
the resolution inline.

---

# STATUS as of 2026-08-01 — first revision pass applied

`docs/TMLR_Journal_Submissions__1_.md` has been rewritten against this brief.
Section and figure numbering changed; see the map at the bottom of this block.

| Item | State | Note |
|---|---|---|
| P0-1 | **done** | 32 / 0.028% / 22.6% in abstract, §5.4 Finding 1, §6.1. The 35 is kept as $S^-_{\text{hi}}$, correctly labelled, with the nesting stated. |
| P0-2 | **done** | "strictly preserve … exclusively" clause deleted; §4.2 ¶2 and ¶3's first sentence deleted. |
| P0-3 | **done** | §3.2 ¶3 deleted entirely; the seed question now lives only in §7 item 3. |
| P0-4 | **done** | `per unit Σ\|D\|` column added to Table 3; §5.4 Finding 2 presents it, explicitly as a normalisation. Duplicate concessions in §5.3 and §5.7 removed; §7 item 4 rewritten. **Numbers differ from the brief's**: computed on micro (3.5× harm, 6.5× support) with the macro values (2.7×, 6.3×) also given, rather than the brief's single 3.5/6.3. |
| P0-5 | **done** | Table 3 now reads macro / Δ macro / micro / Δ micro / errors rec. / per-unit-mass / control Δ. Caption states errors-recovered is micro. |
| P0-6 | **done — and this brief was wrong** | Lookup Q1: concept 1637 has `D(horse) = +2.34e-3` and `D(dog) = -9.50e-3`. It **supports horse and harms dog**, the opposite of what the manuscript said and of the author confirmation recorded below. The `analysis/HIND/1637 - Negative Dog Pos Horse` folder name was the *correct* artifact; the manuscript and this brief were the stale ones. Concept 2979 has the same polarity at 3× the magnitude, so `RESULTS_DRAFT` §5.6 was also right. Archetype fixed as **1637** (it is the one with a figure); §5.2, §5.5 and the Figure 4 caption all corrected. Both dog pairs are members of the 32-pair $S^-_{\text{inv}}$ bucket, which strengthens §5.5 — the archetype is an instance of the paper's central category, and it harms the model's weakest sketch class. |
| P0-7 | **done — the manuscript was right** | Lookup Q2 top three by contrast: **52** (giraffe/person), **3128** (dog/elephant), **2979** (horse/dog). That is exactly the class list the manuscript gave; `RESULTS_DRAFT`'s list (4501/2979/3128) was the wrong one, and this brief backed the wrong horse. Only the missing word and the concept ids needed fixing. |
| P0-8 | **done** | "and possibly target-domain statistics" deleted. |
| P0-9 | **done** | Resolved from `results/analysis/typology_counts.csv`, no run needed: gated figure is **55.7% (108/194)**, matching the brief's read-off exactly. Ungated 50.7% retained as a parenthetical with the caveat. |
| P0-10 | **done — and it does not close in one sentence** | Lookup Q3: **1,629** pairs (not 1,616 — the brief's arithmetic missed that 13 pairs are both inert and above the floor, so the three populations do not partition as `114,688 − 111,530 − 1,542`). Their Σ\|D\| is **0.390 = 42.5% of all non-inert effect mass**, with 327 above τ_D. That is not negligible. §4.3 now states the size plainly and gives two bounding considerations — the estimates are the high-variance ones and \|D\| is a magnitude so noise inflates it, and §5.4 shows aggregate mass is a poor proxy for consequence — while explicitly declining to claim the population is unimportant. Added to §7 item 3 as a scoping limitation. The funnel now partitions correctly and labels this box with its mass share. |
| P0-11 | **done** | Resolved from `clean_lib/masks.py`: `S^-_hi` is `D < -τ_D ∧ R ≥ τ_R` (R alone). All six buckets now defined with predicates in §3.7, plus the nesting statement and an indent in Table 3. |
| P1-1 | **done** | Reconciling paragraph added at the end of §5.3. |
| P1-2 | **done** | New order: 5.5 mechanism, 5.6 domain attribution, 5.7 selectivity. Per-class table moved to Appendix A. |
| P1-3 | **done** | Chest example told once, in §5.5. §5.2 carries a forward reference; §6.3 deleted. |
| P1-4 | **done** | Four contributions → three, as proposed. |
| P1-5 | **done** | Bucket vocabulary defined once in §3.7 with symbols. **Deviation:** Table 1 keeps all eight rows with a count column rather than collapsing to four — the brief's "bottom half is ~3% of the data" is not right. The low-H rows hold 79 of 273 pairs (28.9%); only the two low-H/**high-R** rows are small, at 8 pairs. A count column makes this self-evident and is more informative than a footnote. |
| P1-6 | **done** | §6.3–6.5 deleted, §6.1 ¶1 deleted, §6 is now two subsections. The redundancy line is promoted to the Conclusion. Tautology-of-D framing appears once, in §5.4. |
| P1-7 | **done** | §3.8 deleted; the conflict definition moved to the end of §3.7. §3.2 trimmed three paragraphs → one. |
| P1-8 | **done** | §2 is now three subsections; old §2.6's class-conditionality point moved into §1. |
| P1-9 | **done** | Six limitations → four, merged as proposed, with the §6.4 and §6.5 content absorbed into item 2. |
| P2-1 | **author** | Pipeline schematic. Caption written (Figure 1) specifying both panels. |
| P2-2 | **code ready** | `scripts/make_figures.py` → `fig2_funnel.png`. |
| P2-3 | **code ready** | → `fig3_h_vs_d.png`. symlog, by regime, τ_D band shaded, τ_H labelled, both marginals, no in-figure title. |
| P2-4 | **code ready** | → `fig5_hr_heatmap.png`, with the harmful-invariant cell named and the τ-sensitivity slope panel. Discharges §4.4's obligation. |
| P2-5 | **code ready** | → `fig7_interventions.png`, diverging bars with ghost controls. |
| P2-6 | **table route taken, scatter cut** | The brief offered "turn Table 5 into a scatter, **or** fix the table". Both were done, then the scatter was cut by author decision: seven points supporting a *negative* claim, when §5.5 already states the two decisive data points in prose and Appendix A carries the full table. Table fixed as specified — rate column added, sorted by accuracy, neutral column dropped, moved to Appendix A. |
| P2-7 | **done** | Domain-attribution table added as Table 4 in the new §5.6. |
| P2-8 | **author, all exemplars now fixed** | Four plates, captions final with concept ids and scores. The exemplar hunt took three passes and the lesson is worth keeping: the brief's candidate **9839 was unusable** (Q1: `H = 0.198`, low-H, illustrates the opposite of §5.3), and every replacement `Q7` ranked highly turned out to be **a concept whose home is a different class** — 52 is a giraffe latent, 15421 an elephant latent, both shown on dog. That is not bad luck but §5.5's own finding: 78.3% of harmful pairs involve a concept supporting some other class, so conditioning on "harmful" nearly guarantees an unownable exemplar, and a single-class grid of one is uninterpretable. `Q10` was added to make **ownership** an explicit criterion — `share(k,c) = n(k,c) / Σ_k' n(k',c)` plus argmax class — and to list supportive candidates first, since `D > 0` guarantees the concept fires on its own class. Settled on **11005 / elephant** (trunk): `H = 0.926`, `R = 0.626`, 848 of 914 elephant images active. |
| P3 | **done** | Abstract leads with the finding, two paragraphs. Table 1 markdown repaired. §3 equations restored from `clean_lib` source. Table 3 precedes its prose; validity split from findings. All eight `[URL 🔗]` artifacts stripped. Seed decimals explained in Table 5's caption. |

**Numbering map.** Sections: old 5.5 → new 5.6; old 5.6 → new 5.5; keep-only extracted from old 5.4 → new 5.7. Tables: 1 typology, 2 SAE fidelity, 3 interventions, 4 domain attribution, 5 keep-only, 6 per-class (Appendix A).

Figures, final: **1** pipeline *(author)*, **2** class–concept population, **3** H-vs-D, **4** three-regime plate *(author)*, **5** H×R contingency, **6** τ sensitivity, **7** high-H/low-R exemplar *(author, serves §5.3 and §5.6)*, **8** interventions, **9** conflict plate *(author)*. Script outputs are `fig2`, `fig3`, `fig5`, `fig6`, `fig8`; the gaps are the author-built plates. The τ sensitivity was split out of Figure 5 into its own figure and swept finely (0.50–0.95 in 0.025 steps) rather than at three points — free, since it is a read-only pass over the score file. The per-class scatter was cut, so there is no Figure 10.

## Figure asset manifest — `docs/figures/`, renamed 2026-08-06 to match numbering

| Fig | File | Source |
|---|---|---|
| 1 | `fig1_pipeline.png` | author *(was `InterpFlowChartV3.png`)* |
| 2 | `fig2_funnel.png` | `make_figures.py` |
| 3 | `fig3_h_vs_d.png` | `make_figures.py` |
| 4a | `fig4a_neutral_feet.png` | author *(was `concept_feet.png`)* |
| 4b | `fig4b_supportive_chin.png` | author *(was `Fig4_Human_Chin.png`)* |
| 4c | `fig4c_harmful_chest.png` | author, concept 1637 *(was `concept_dogchest.png`)* |
| 5 | `fig5_hr_heatmap.png` | `make_figures.py` |
| 6 | `fig6_tau_sensitivity.png` | `make_figures.py` |
| 7 | `fig7_lowR_trunk.png` | author, concept 11005 *(was `Fig7_Elephant_Trunk.png`)* |
| 8 | `fig8_interventions.png` | `make_figures.py` |
| 9 | `fig9_conflict_52.png` | author, concept 52 *(was `Fig9_Concept_52.png`)* |

Figure 4 is **three separate assets**, to be assembled as subfigures (a)/(b)/(c) at
LaTeX build time; §5.2 and the caption reference them individually.

**Unreferenced legacy assets**, left in place but cited nowhere in the manuscript:
`DvR.png`, `HvD.jpeg`, `R-acc.png`, `heatmapR.png`. These predate the clean score
file and must not be reused without re-quoting scores — the same provenance trap
that made the 1637 folder name look wrong when it was right.

---

**Both commands were run on 2026-08-01** (`results/runs/20260801T0849{36,57}Z_*`) and
every `[PENDING LOOKUP …]` marker is now resolved out of the manuscript. Two reruns
are outstanding, both seconds:

```
python scripts\answer_critique.py --scores processed\FINAL_ERM_ResNet_3300_T3.json   # for the new Q7 block
python scripts\make_figures.py    --scores processed\FINAL_ERM_ResNet_3300_T3.json   # funnel partition fix
```

**Three findings from the lookups contradicted this brief** — see P0-6, P0-7 and
P0-10 above. Two of them (the 1637 polarity and the conflicted-concepts list) mean
the manuscript and the `analysis/` folder names were right and the brief's
reconstruction was wrong. The third means P0-10 is a real limitation rather than a
formality.

---

## Verdict in one paragraph

The science is solid and the central argument is genuinely interesting. The problem
is that **the paper is organised as a framework paper but its best material is a set
of findings.** Those findings are buried under formalism nothing uses (§3.8), a
Related Work that lists rather than argues (six subsections), and a Discussion that
is roughly 60% verbatim restatement of Results. Separately, there is one place where
the paper concedes a limitation that its own numbers already partly answer — see
**P0-4**, which is the single highest-value edit available.

**If only three things get done:** (1) cut §3.8 and §6.3–6.5; (2) add a pipeline
figure and a population-funnel figure; (3) fix the 35-vs-32 mislabelling that
propagates into the abstract.

---

# P0 — Correctness. Fix before circulating.

## P0-1 The 35/32 mislabelling propagates into the abstract

- [ ] **Fix in abstract, §5.4 and §6.1.**

Three places describe the same bucket inconsistently:

| Location | Current text |
|---|---|
| Abstract | "35 pairs that are both invariant and consistent in their harm, 0.03% of the class–concept grid, account for a quarter of all target-domain errors" |
| §5.4 (third finding) | "The 35 pairs that are both invariant and consistent in their harm … account for 25.1% … and the 32-pair harmful invariant bucket accounts for 22.6%" |
| §6.1 | "35 class–concept pairs, 0.03% of the grid, account for a quarter of all target-domain errors" |

From Figure 4's own harmful panel: **32** pairs sit in (H ≥ 0.7, R ≥ 0.7) and **3**
sit in (H < 0.7, R ≥ 0.7). Therefore:

- **35 = all high-R harmful = "distributed harm".** Three of them are *not* invariant.
- **32 = high-H ∧ high-R harmful = "harmful invariant".**

So "the 35 pairs that are both invariant and consistent" is false for 3 of them, and
it makes §5.4 self-contradictory across consecutive clauses — it names 35 as the
invariant-and-consistent set, then names 32 as the harmful-invariant set. The
abstract and §6.1 inherit the error. `0.03%` rounds identically for both, so the
percentage does not disambiguate.

**Action.** The bucket the argument is actually about — the one alignment cannot see
— is the 32. Change the abstract and §6.1 to:

> **32 class–concept pairs, 0.028% of the grid, account for 22.6% of all
> target-domain errors.**

Then keep §5.4's distributed-vs-concentrated comparison as a separate, correctly
labelled point about the 35.

---

## P0-2 §4.2 contradicts itself within three paragraphs

- [ ] **Delete the stale clause in §4.2 paragraph 1.**

Paragraph 1 still reads: *"To strictly preserve the domain generalization setting,
the SAE is trained and analyzed exclusively on the source domains."*

Paragraph 3 then discloses that the normalizer's two scalar statistics came from a
batch spanning all four domains.

The disclosure patch was applied to paragraph 3 but the offending clause in
paragraph 1 was never removed, so the section asserts something and retracts it two
paragraphs later. Delete "exclusively" (and the "strictly preserve" framing) from
paragraph 1 and let paragraph 3 carry the qualification.

While in §4.2, also delete paragraph 2 ("To verify training convergence…") — it
previews §5.1, which does the job properly — and the first sentence of paragraph 3
("Unless otherwise stated, the SAE is trained on source-domain features only"),
which repeats paragraph 1.

---

## P0-3 §3.2 promises a robustness analysis §7 admits was not run

- [ ] **Rewrite the last paragraph of §3.2.**

§3.2 says: *"we report sensitivity to threshold choices and, where computationally
feasible, to SAE hyperparameters or independently trained SAE seeds."*

§7 (fourth limitation) says: *"…a single sparse autoencoder … we would compare only
aggregate quantities across seeds … and we have not done so."*

A reviewer reading §3.2 will go looking for the seed analysis and find a retraction.
Rewrite §3.2 to promise **threshold sensitivity only**, and let §7 own the seed
question.

---

## P0-4 The paper concedes a limitation its own numbers partly answer

- [ ] **Add one paragraph to §5.4. Downgrade §7's sixth limitation.**

**This is the highest-value edit in the document.**

The draft concedes *four separate times* (§5.3 closing, §5.4 closing, §6-adjacent,
§7 sixth) that R may be a proxy for the sign and magnitude of D, because robust
support carries ~10× the |D| mass of concentrated support.

But Table 3 already permits a **mass-normalised** comparison using numbers that are
already in the paper:

| Bucket | pairs | Σ\|D\| | errors recovered | **errors recovered per unit \|D\| mass** |
|---|---|---|---|---|
| Distributed harm | 35 | 0.087 | 25.1% | **289** |
| Concentrated harm | 103 | 0.163 | 13.4% | **82** |

Distributed harm is **3.5× more damaging per unit of effect mass** than concentrated
harm. The supportive side agrees: distributed support costs 39.08 points on 0.237
mass (165/unit) against concentrated support's 0.65 on 0.025 mass (26/unit) — a
factor of **6.3**.

**This is not a mass-matched intervention and must not be claimed as one.** But it is
a mass-*normalised* comparison drawn entirely from existing table entries, and it
shows the R effect survives a first-order magnitude adjustment by a factor of 3.5–6.

**Action.**
1. Add a short paragraph to §5.4 presenting the per-unit-mass column, explicitly
   labelled as a normalisation rather than a matched control.
2. Optionally add a `per unit Σ|D|` column to Table 3.
3. Rewrite §7's sixth limitation from *"we do not establish that consistency carries
   information independent of…"* to *"we do not fully separate consistency from
   magnitude; a mass-normalised comparison indicates the effect is not purely one of
   magnitude, but a mass-matched intervention would be required to settle it."*
4. Delete the duplicate concessions in §5.3 and §5.4 — state it once, in §7.

As written the paper understates its own result.

---

## P0-5 Table 3's Δ column is macro; its "Errors recovered" column is micro

- [ ] **Restore the micro Δ column.**

`+7.24 / 19.78 = 36.6%`, but the table prints 45.6%. That 45.6% comes from the
**micro** Δ of +9.01, which exists in `docs/RESULTS_DRAFT.md` but was dropped from
the manuscript. Every row behaves this way. As printed, no reader can reproduce any
value in the "Errors recovered" column from the columns beside it.

This is doubly bad because §4.3 makes a principled point of always reporting both
averages — and then a headline percentage is silently computed from the one that was
dropped.

**Action.** Restore the columns from `RESULTS_DRAFT.md` Table 3 so the table reads:
sketch macro | Δ macro | sketch micro | Δ micro | errors recovered | control Δ. State
in the caption that errors-recovered is computed on micro.

---

## P0-6 Reconcile the Figure 2 concept's identity and polarity

- [ ] **Settle from the clean score file, then use one description everywhere.**

> **SUPERSEDED 2026-08-01 by lookup Q1.** The author confirmation recorded below —
> that concept 1637 supports dog and harms horse — is **false**. The score file gives
> `D(horse, 1637) = +2.34e-3` and `D(dog, 1637) = -9.50e-3`: it supports **horse** and
> harms **dog**. The `analysis/` folder name was right and the manuscript was wrong,
> which is the reverse of the assumption below. Everything downstream is corrected;
> the original text is kept only so the reasoning is auditable.

~~**Confirmed by the author: the concept supports dog (D > 0) and harms horse
(D < 0).** The manuscript is correct.~~ Two on-disk artifacts disagree and must be
reconciled so nobody re-introduces the error:

| Artifact | Says |
|---|---|
| `docs/figures/concept_dogchest.png` | "Concept 1637 – Dog Chest", panels for Class 4 Horse and Class 0 Dog |
| `analysis/ERM_ResNet_T3/HIND/1637 - Negative Dog Pos Horse/` | folder name asserts the **opposite** polarity |
| `docs/RESULTS_DRAFT.md` §5.6 | narrates the chest story for **concept 2979**, "supports horse and harms dog" |

Per `CLAUDE.md` §9.5, the `analysis/` folder names encode scores from the
**superseded** score file, so the folder name is the likely stale artifact — but this
must be verified, not assumed.

**Action.**
1. Look up H, D, R for **(dog, 1637)** and **(horse, 1637)** in
   `processed/FINAL_ERM_ResNet_3300_T3.json`.
2. Decide whether the paper's archetype is **1637** or **2979** and use that concept
   ID consistently in §5.2, §5.6 and the figure caption.
3. Re-quote the H/D/R values into the caption from the clean file.
4. Rename or annotate the stale `analysis/` folder so it stops contradicting the paper.

---

## P0-7 §5.6's most-conflicted-concepts sentence is broken and possibly wrong

- [ ] **Rewrite and verify against the clean file.**

Current text:

> "The most conflicted concepts by contrast max_k D − min_k D support giraffe while
> harming person, support dog while harming elephant, and support horse while harming
> dog."

Two problems: (a) a word is missing after "by contrast" — it should read "ranked by
the contrast max_k D − min_k D"; (b) the giraffe/person pair does not appear in
`RESULTS_DRAFT.md`, which lists **4501** (person/dog), **2979** (horse/dog) and
**3128** (dog/elephant). Determine which list is correct and quote concept IDs.

---

## P0-8 §7's third limitation contradicts the paper's own protocol

- [ ] **Delete the clause "and possibly target-domain statistics".**

§7 currently says oracle interventions use the true class *"and possibly
target-domain statistics, to define masks."* This is not true — H, D and R are
source-only by construction, and §5.4 makes a point of it. The clause hands a
reviewer a leakage quote from the paper's own Limitations section. Remove it.

---

## P0-9 §5.3's "50.7%" is computed over a population where R is meaningless

- [ ] **Recompute or explicitly qualify.**

§5.3 says: *"Of the 1173 high-H pairs, only 50.7% also clear τ_R. Knowing that a
concept is activation-invariant tells us almost nothing about whether its effect is
consistent."*

83.5% of those 1,173 pairs are **neutral** — and three paragraphs later the same
section argues at length that R must not be interpreted without a magnitude gate. The
headline statistic for "H tells you little about R" is therefore drawn from a
population the paper declares R-meaningless.

**Action.** Recompute on the gated invariant population. Reading Figure 4:
`(76 + 32) / 194 = 55.7%` — **verify this against the score file** rather than
trusting the read-off. Report the gated figure as the headline and, if useful, the
ungated one as a parenthetical with the caveat stated.

---

## P0-10 Reconcile the population arithmetic

- [ ] **Resolve from the score JSON, then state it in the paper.**

`114,688 − 111,530 (inert) − 1,542 (above floor) = 1,616` pairs that have **nonzero
measured effect but fall below the support floor.** They are excluded from every
intervention and are never mentioned in the paper. A reviewer will ask.

**Action.** Confirm the count and their aggregate Σ|D| from
`processed/FINAL_ERM_ResNet_3300_T3.json`. If the mass is negligible, one sentence in
§4.3 closes it. If it is not, say so plainly. Either way this becomes a labelled box
in the funnel figure (**P2-2**), which is the natural place for it.

---

## P0-11 Nail down the definition of the "distributed harm" bucket

- [ ] **State the predicate explicitly in §3.7.**

Is the 35-pair bucket defined as `D < −τ_D ∧ R ≥ τ_R` (R alone), or
`D < −τ_D ∧ R ≥ τ_R ∧ H ≥ τ_H`? Figure 4 implies the former (32 + 3 = 35). This must
be confirmed, because **P0-1** depends on it and because no reader can currently
determine it from the text.

Do the same for every named bucket. See **P1-5**.

---

# P1 — Structure and storyline

## P1-1 Close the gap between the paper's two halves

- [ ] **Add the reconciling sentence at the end of §5.3 or the start of §5.4.**

The headline is "the concepts that damage held-out accuracy most are those an
activation-alignment criterion scores best." That criterion is **H**. But §5.2 shows
invariance is *protective* — it halves the incidence of harm (13.3% → 7.6%) — and the
per-pair damage comparison in §5.4 is a **distributed-vs-concentrated** contrast,
which is an **R** contrast, not an H contrast.

So the punchline attributes to H what the data attributes to H ∧ R, and §5.2 and §6.1
read as though they contradict each other. The paper never reconciles them.

The reconciliation is one sentence and it strengthens the argument:

> Invariance reduces the *rate* at which concepts are harmful, but the harm that
> survives the invariance filter is precisely the harm alignment cannot see, and it
> is several times more damaging per pair than the harm invariance screens out.

Support it with the per-pair figures already in §5.4 (0.72% vs 0.13% of errors per
pair) and the per-unit-mass figures from **P0-4**.

---

## P1-2 Reorder §5 so it does not end on its weakest material

- [ ] **Apply the reordering below.**

Current order ends with an admitted non-result ("Giraffe … is not explained by either
quantity") and places the paper's most-hedged section (§5.5 domain attribution)
between its two strongest.

**§5.5 is being kept** per author decision — but it should move, and it needs its
table (**P0/P2**, see **P2-7**) and a qualitative exemplar (**P2-8**).

| New § | Content | Rationale |
|---|---|---|
| 5.1 | SAE fidelity | unchanged |
| 5.2 | Invariance ≠ usefulness | unchanged |
| 5.3 | Consistency, and the support/harm asymmetry | unchanged |
| 5.4 | Interventions: the categories are functional | drop the keep-only material (moves to 5.7) |
| 5.5 | **Harm is largely misdirected support** (was 5.6, minus per-class table) | This is the *mechanism*; it explains why 5.2–5.4 look as they do. Belongs after the phenomenon, not last. |
| 5.6 | **Domain-contingent effects concentrate in the stylised source domains** (was 5.5) | Kept, but placed after the mechanism it elaborates, not before it. |
| 5.7 | **Selectivity: retaining 0.07% beats the model** (extracted from 5.4) | The most striking single number in the paper. Deserves its own subsection and is a natural closer. |

Extracting the keep-only result also relieves a length problem: §5.4 currently carries
a 9-row table, a 5-row table, four numbered findings and two caveat paragraphs. It is
doing the work of two sections.

Move the **per-class table (Table 5)** out of the narrative flow — either to the end
of the new §5.5 as a short descriptive paragraph, or to an appendix. Its own text says
the pattern is non-monotone and partly unexplained; it should not be the last thing
the Results section says.

---

## P1-3 Tell the chest-and-forelimb example once

- [ ] **Consolidate three tellings into one.**

It appears at paragraph length in §5.2, again in §5.6 ("the archetype"), and again in
§6.3. Tell it **once**, in the new §5.5 where it functions as the mechanism. In §5.2
reduce it to a forward reference: *"one such concept is examined in §5.5."*

This also fixes a sequencing problem: in §5.2 the example is introduced to illustrate
negative D, but the reader has no framework yet for what makes it interesting (class
conflict), so it lands flat.

---

## P1-4 Rewrite the contributions list to match what is delivered

- [ ] **Four contributions → three.**

Contribution 4 claims "class-level and model-level diagnostic profiles." There are no
model-level results (single checkpoint), and §6.4 explicitly retracts the class-level
profiles as non-predictive. A reviewer who reads the contributions and reaches §6.4
will feel something was walked back.

Proposed replacement:

1. Sparse concept diagnostics: three class-conditional scores (H, D, R) over SAE
   candidate concepts, estimated on source domains only.
2. A measurement showing activation invariance is weakly informative about the
   absence of harm and uninformative about the presence of usefulness, and that
   invariant support distributes across domains while invariant harm concentrates.
3. Controlled interventions, validated against size- and magnitude-matched random
   controls, establishing that the categories are functional — and that a
   source-identified 0.07% of the class–concept grid suffices to exceed the model's
   own held-out accuracy.

---

## P1-5 Fix the bucket vocabulary — currently the largest clarity failure

- [ ] **Define every bucket name exactly once, in §3.7, and use it verbatim thereafter.**

A reader cannot map Table 3's rows onto Table 1's typology. Table 1 uses names like
"Domain-contingent harmful concept" and "Local or unstable harmful cue." Table 3 uses
"distributed harm (35)", "harmful invariant (32)", "concentrated harm (103)". These
are different vocabularies for overlapping sets — and "distributed harm (35)" is the
**union of two Table 1 rows**, which is why it is 35 and not 32.

Actions, in order:

1. **Name the buckets once, in §3.7**, alongside the typology, each with its defining
   predicate. Introduce the short symbols (`S+_hi`, `S+_lo`, `S−_hi`, `S−_lo`) there
   and use them as table row labels — the long phrases are eating line width in every
   table.
2. **Collapse Table 1.** Its eight rows are over-specified: only 8 of 273 non-neutral
   pairs are low-H, so the entire bottom half of the truth table describes ~3% of the
   data. Keep the four populated rows in the table; footnote the rest.
3. **Make the set relations visible.** "Harmful invariant (32) ⊂ distributed harm
   (35)" is invisible in Table 3 because rows are sorted by Δ. Add a predicate column
   (e.g. `D < −τ_D ∧ R ≥ τ_R ∧ H ≥ τ_H`) or an indent/brace showing nesting. A reader
   currently cannot tell that two rows overlap by 91%, and may read 25.1% + 22.6% as
   additive.

---

## P1-6 Cut the Discussion to two subsections

- [ ] **Delete §6.3, §6.4, §6.5. Trim §6.1.**

| Subsection | Duplicates | Action |
|---|---|---|
| §6.1 para 1 | §5.2 (83.5%, halving, unchanged usefulness) | **Delete.** Number recitation. |
| §6.1 para 2 ("the sharper point") | — | **Keep.** Genuinely new framing; this is the strongest paragraph in §6. |
| §6.2 | — | **Keep as is.** Best-written passage in the paper. |
| §6.3 | §5.6, near sentence-for-sentence, including the remediation paragraph | **Cut entirely.** §5.6 already draws the conclusion. |
| §6.4 | §5.6's last paragraph (person, ties, non-monotone) | **Cut.** One sentence in Limitations. |
| §6.5 | §5.4's opening paragraph (tautology, controls, chance-collapse) | **Cut.** One sentence in Limitations. |

Resulting §6: two subsections — *What alignment cannot reach* and *Failures of
generalization can be failures of selectivity*.

Note also: **the tautology-of-D framing appears three times** (§5.4 opening, §5.4
finding (i), §6.5). Keep it once, in §5.4's opening.

Promote §6.2's closing line — *"Redundancy, rather than invariance, may be what
degrades under shift"* — it is the best sentence in the Discussion and is currently
the last clause of the last paragraph of a subsection. Move it to the subsection
opening or the Conclusion.

---

## P1-7 Cut §3.8 and trim §3.2

- [ ] **Delete §3.8; keep only the conflict definition.**

§3.8 defines total discriminative mass, RSM, HIM, DCM and CM. **Nothing downstream
uses RSM, HIM or DCM.** Four defined quantities with zero results is the clearest
"why is this here?" in the paper.

Keep only the **conflicting concept** definition and move it to §3.7, or inline it in
the new §5.5 where it is used. Delete the rest.

- [ ] **Trim §3.2's "Scope of the SAE representation" from three paragraphs to one.**

Paragraph 1 (non-identifiability of SAE dictionaries) is necessary and well-argued —
keep it, roughly four sentences. Paragraphs 2 and 3 duplicate §7 limitations 1, 4 and
6, and paragraph 3 additionally makes the promise flagged in **P0-3**. Delete both.

---

## P1-8 Collapse Related Work from six subsections to three

- [ ] **Restructure §2.**

Six subsections for three ideas. It reads as a survey rather than an argument, and
§2.6 exists only because §2.1–2.5 never made a point.

| New | Merges | Note |
|---|---|---|
| 2.1 Domain generalization and the invariance hypothesis | old 2.1 + 2.2 | The only subsection that sets up the contribution. End it with the gap stated as your research question — the closing paragraph of old §2.2 already does this well; promote it. |
| 2.2 Concept-based interpretability | old 2.3 | Trim. |
| 2.3 Sparse autoencoders | old 2.4 + 2.5 | Both currently end on the same "SAE quality must be validated by intervention" point. Merge and state it once. End with the two design choices inherited. |
| — | old 2.6 | **Delete.** Recaps 2.1–2.5 then restates §1. Its useful content (class-conditionality as the distinguishing move) belongs in §1 or at the end of new §2.1. |

---

## P1-9 Merge and trim Limitations

- [ ] **Six limitations → four.**

- Items **2 and 3** say the same thing (post-hoc / uses labels; oracle / uses true
  class). Merge into one, and apply **P0-8** while there.
- Item **6** is stated a third time here after §5.3 and §5.4. Keep it *only* here, and
  rewrite per **P0-4**.
- Item **1** and item **4** are fine.
- Add one sentence absorbing the deleted §6.4 (class-level profiles are descriptive,
  not validated as a selection criterion) and one absorbing the deleted §6.5 (oracle
  interventions are not deployment methods).

---

# P2 — Figures and tables

## P2-1 Add a method/pipeline figure — this should be Figure 1

- [ ] **New figure.**

There is currently **no picture of the pipeline**: image → feature map (7×7×2048) →
49 spatial tokens → SAE code (16,384-dim, top-16) → **zero column c** → decode →
denormalise → pool → classify → Δp(y). The entire method has that shape, and the
reader must assemble it from stripped equations across §3.2–3.5.

Add a **right-hand panel on the same figure** showing the three scores for a single
(k, c) pair:

- three bars = per-domain mean activation → **H**
- one bar = Δp(y) under ablation → **D**
- three bars = per-domain |D_d| → **R**

That panel replaces two paragraphs of §5.3 prose about where a concept *fires* versus
where it *matters*. The H/R distinction is the hardest idea in the paper and is
currently prose-only.

---

## P2-2 Add a population funnel — second priority

- [ ] **New figure.**

The paper asks the reader to hold twelve populations, introduced across four
sections, several of which are subsets of one another: 114,688 → 111,530 inert →
1,542 above floor → 1,269 / 135 / 138 → 76 / 35 / 32 / 103 / 54 / 81.

A single funnel or Sankey — grid → above floor → non-neutral → signed → H×R buckets,
annotated with counts and Σ|D| — prevents every population confusion in the paper, and
forces **P0-10**'s 1,616 unexplained pairs to become a labelled box rather than a
silent omission.

---

## P2-3 Rebuild Figure 1 (H vs D scatter)

- [ ] **Regenerate `results/analysis/fig_h_vs_d.png`.**

As rendered it is an unresolvable vertical stripe at D ≈ 0 plus a scatter of outliers.

| Problem | Fix |
|---|---|
| D spans ~1e-5 to 1.4e-2 on a **linear** axis, so 82% of points (the neutral majority) collapse to a line | symlog x-axis with linear threshold at τ_D, or plot `sign(D)·log₁₀\|D\|` |
| Coloured by **class** (7 categories, heavily overplotted) — but no claim in §5.2 is about class | colour by **regime** (harmful / neutral / supportive); the figure then states its own caption |
| τ_D band invisible; the H = 0.7 line is an unlabelled dash | shade the \|D\| ≤ τ_D band; label the τ_H line |
| No marginals | add marginal histograms on both axes — the H marginal is what makes "invariance is common and cheap" visible; the D marginal is what makes "the neutral mass dominates" visible. Both are currently prose-only claims. |
| In-figure title duplicates the caption | drop the title (TMLR prefers caption-only) |

---

## P2-4 Improve Figure 4 (H×R heatmap) and give it a sensitivity panel

- [ ] **Annotate, and add a companion panel.**

The heatmap is good and carries the central asymmetry, but the key comparison —
**56.3% (supportive, high-H high-R) vs 23.2% (harmful, high-H high-R)** — requires the
eye to jump between panel 3 and panel 2.

1. Label the harmful high-H/high-R cell **"harmful invariant"** by name, so it visibly
   connects to Table 3 and to the abstract.
2. Add a small companion panel: a slope/dumbbell chart with τ ∈ {0.7, 0.8, 0.9} on x
   and two lines — invariant-supportive-also-consistent (72.4 → 65.2 → 49.4) and
   invariant-harmful-also-consistent (36.0 → 27.3 → 15.4).

That second panel makes the monotone strengthening visual rather than a prose triple,
**and it is the threshold-sensitivity analysis that §4.4 promises** and §5.3 currently
delivers as three numbers in one sentence. One small panel discharges both obligations.

---

## P2-5 Give Table 3 a chart

- [ ] **New figure.**

"Every harmful bucket beats its matched random control, and the control for the full
harmful set moves in the *opposite direction*" is the most persuasive result in the
paper, and it is currently nine rows of numbers.

Horizontal diverging bar chart: sketch Δ per intervention, with the control Δ as a
paired ghost bar, sorted by Δ. The −20.98 control against the +7.24 target is a
striking picture and an unmemorable table cell.

---

## P2-6 Turn Table 5 (per-class) into a scatter, or fix the table

- [ ] **Pick one.**

The claim about Table 5 is *negative*: conflict burden does not predict per-class
accuracy, with `person` as the outlier. That is one scatter — sketch accuracy (y)
against harmful-pair rate (x), seven labelled points, `person` and `dog` annotated.
The reader sees the non-relationship instantly instead of parsing a paragraph.

If the table stays:
- drop the **Neutral** column (fully determined by the other three);
- sort rows by sketch accuracy, not alphabetically;
- add a **rate** column — raw counts are not comparable across classes whose floor
  populations range from 144 to 326.

---

## P2-7 §5.5 quotes a table that is not in the paper

- [ ] **Add the domain-attribution table.**

§5.5 gives eight numbers (46.3%, 19.8%, 37.9%, 22.9%, 5.6%, 6.8%, 55.3%, 54.3%) from
a 2×3 contingency with a null row — **entirely in prose, with no table.** It is the
densest prose in the paper and the least supported by a display.

The table exists as Table 5 in `docs/RESULTS_DRAFT.md` and as
`results/analysis/domain_attribution.csv`. Since §5.5 is being kept, bring the table
in.

---

## P2-8 Qualitative examples — add four, no more

- [ ] **Extract and place as below.**

Each addition must carry an argument the quantitative results already establish. The
paper cites Han et al. (2025) warning that top-activation grids can mislead, and
commits to treating them as aids only — so every extra grid increases that exposure
without adding argument. Four, and stop.

| # | Section | What to show | Why | Candidate asset |
|---|---|---|---|---|
| 1 | **§5.3** | A real high-H, **low-R**, non-neutral pair: grid across the three source domains, **with two bar insets** — per-domain mean activation (flat → high H) and per-domain \|D_d\| (spiked → low R) | §5.3 currently illustrates itself with a **hypothetical** ("consider a concept responding to outline and line weight… may be concentrated in cartoons"). Inventing an example when 103 real ones exist is a weakness a reviewer will notice. Flat bars beside spiked bars *is* the H-vs-R distinction, in one glance. **Highest-value qualitative addition.** | `analysis/ERM_ResNet_T3/LIPD/9839 - High for Elephant Cartoon` |
| 2 | **§5.2** | A **three-row comparative plate**: neutral / supportive / harmful, one concept each, three source domains per row | §5.2 shows neutral (Fig 3) and harmful (Fig 2), then says the positive case is "the same chest concept evaluated on dog" — i.e. robust support, the bucket that reaches 94% alone, has **no exemplar**. A single plate also lets the reader see all three D regimes in the same visual frame that Figure 1 plots quantitatively. Replaces Figs 2 + 3 with one figure. | `HIPD/4501`, `HILD/1893 - Lines` or `concept_feet.png`, `concept_dogchest.png` |
| 3 | **§5.6 (domain attribution)** | A **cartoon-concentrated supportive** concept, grid plus per-domain \|D_d\| bars spiking in cartoon | This section's claim is the paper's only mechanism claim not backed by an intervention, and it currently has zero visual support. **May be the same figure as #1** if the exemplar is chosen to serve both. | `LIPD/9839` |
| 4 | **§5.5 (class conflict)** | ~~A 3-row conflict plate~~ → **one concept, two rows** (supported class / harmed class), top four activations each, unselected | **Revised 2026-08-01.** The brief wanted three rows so the claim reads as a pattern rather than an anecdote. In practice only concept **52** is legible — it detects giraffe-neck patterning and fires on ties and checked regions of people, which shows the mechanism outright — while 3128 and 2979 are not visually revealing. Weak rows dilute strong ones: a reviewer who cannot see the conflict in rows 2 and 3 begins doubting row 1. The pattern claim is carried by the 56 conflicting concepts and 78.3% in the text, and the mechanism still appears in two distinct concepts across two figures, since **1637 remains §5.5's prose archetype** (its dog pair is in the 32-pair $S^-_{\text{inv}}$ bucket, which 52's person pair may not be). | 52 (giraffe/person) |

**Caution on the existing figure.** `docs/figures/concept_dogchest.png` includes a
**Sketch** row. Sketch is the held-out target and contributes nothing to H, D or R.
The caption **must** say the sketch row is illustrative only — otherwise a reviewer
sees target-domain images in a figure that also quotes D values and suspects leakage.
Also reorder the rows so the three source domains group together and sketch sits last,
visually separated. Current order is Art / Sketch / Photo / Cartoon.

**All figure captions must re-quote H, D and R from
`processed/FINAL_ERM_ResNet_3300_T3.json`.** Per `CLAUDE.md` §9.5, the folder names
under `analysis/` encode scores from the superseded file and must not be trusted.

---

# P3 — Polish

- [ ] **Abstract.** ~290 words, one paragraph, seven quantitative claims. It opens with
  the framework and does not reach a finding until sentence four. **Lead with the
  finding.** Consider breaking after "…no measurable discriminative effect" and
  opening the second half with the mechanism (misdirected support) rather than another
  statistic. Apply **P0-1** here.
- [ ] **Table 1 is broken in the markdown** — all eight rows have collapsed into the
  first column. Fix now rather than at LaTeX-build time; see also **P1-5** item 2.
- [ ] **§3.1 is unreadable** ("Let denote a set of domains and / di / fcls."). §3.3–3.6
  are similarly damaged. The whole of §3 needs the equations restored and reflowed
  before the LaTeX build.
- [ ] **§5.4 puts four numbered findings before the table they refer to.** Move Table 3
  above the prose.
- [ ] **§5.4's four findings mix validity checks with findings.** (i) controls and (ii)
  denoising are validity; (iii) and (iv) are results. Split into one short "validity"
  paragraph and two named findings.
- [ ] **Figures 2 and 3 are cited in §5.2 but placed after §5.3.** Watch float placement
  at build time.
- [ ] **Decimals and seeds.** Table 4 gives "14.29 / 14.25 / 14.82" for three seeds while
  seeds are averaged elsewhere. Add a clause to the caption saying why (presumably: to
  show two land exactly at chance).
- [ ] **Strip the `[URL 🔗](#page-0)` artifacts** littered through the markdown from the
  PDF extraction.

---

# Consolidated section-by-section action table

| Section | Action |
|---|---|
| Abstract | Fix 35→32 (**P0-1**); lead with the finding (**P3**) |
| §1 Contributions | Four → three, matched to delivered results (**P1-4**) |
| §2.1–2.6 | Six subsections → three; delete §2.6 (**P1-8**) |
| §3.1, §3.3–3.6 | Restore equations; unreadable in markdown (**P3**) |
| §3.2 | Three paragraphs → one; remove the seed-sensitivity promise (**P0-3**, **P1-7**) |
| §3.7 / Table 1 | Define all bucket names + predicates here, once; collapse the 8-row truth table (**P1-5**, **P0-11**) |
| §3.8 | **Delete.** Keep only the conflict definition, moved to §3.7 (**P1-7**) |
| §4.2 | Delete the "exclusively source domains" clause and two redundant paragraphs (**P0-2**) |
| §4.4 | Sensitivity obligation discharged by the new Fig-4 companion panel (**P2-4**) |
| §5.2 | Reduce the chest example to a forward reference (**P1-3**); add the three-regime plate (**P2-8** #2) |
| §5.3 | Fix the 50.7% population (**P0-9**); add the real low-R exemplar (**P2-8** #1); delete the duplicated R-vs-magnitude concession (**P0-4**) |
| §5.4 | Restore micro Δ (**P0-5**); add per-unit-mass paragraph (**P0-4**); fix 35/32 (**P0-1**); table before prose, split validity from findings (**P3**); extract keep-only into new §5.7 (**P1-2**) |
| §5.5 → new §5.6 | Keep, but move after the mechanism; **add its missing table** (**P2-7**); add the cartoon exemplar (**P2-8** #3) |
| §5.6 → new §5.5 | Move earlier; absorb the single full telling of the chest example (**P1-3**); fix the broken conflicted-concepts sentence (**P0-7**); add the conflict plate (**P2-8** #4); move Table 5 out of the flow (**P1-2**, **P2-6**) |
| new §5.7 | Keep-only / selectivity result, extracted from §5.4 (**P1-2**) |
| §6.1 | Delete paragraph 1; keep paragraph 2 (**P1-6**) |
| §6.2 | Keep as is; promote its closing line (**P1-6**) |
| §6.3, §6.4, §6.5 | **Delete** (**P1-6**) |
| §7 | Six → four; merge 2+3; delete the target-statistics clause; rewrite item 6 (**P0-8**, **P0-4**, **P1-9**) |
| §8 Conclusion | Verify it does not restate the 35-pair claim incorrectly after **P0-1** |

---

# Data lookups required (from `processed/FINAL_ERM_ResNet_3300_T3.json`)

These are the only items that need a run. All are read-only queries against the
existing clean score file — no re-scoring, no new experiments. Per `CLAUDE.md` §6, the
**author runs these**, one at a time, in an Anaconda Prompt with `interpretability`
activated.

| # | Query | Resolves |
|---|---|---|
| 1 | H, D, R for (dog, 1637) and (horse, 1637); same for 2979 | **P0-6** figure polarity and concept identity |
| 2 | Top concepts by `max_k D − min_k D`, with the supported/harmed class pair | **P0-7** the broken sentence |
| 3 | Count and Σ\|D\| of pairs with nonzero effect below the support floor | **P0-10** the missing 1,616 |
| 4 | Fraction of the **gated** invariant population (n = 194) clearing τ_R | **P0-9** the 50.7% |
| 5 | Exact predicate and membership of the 35-pair and 32-pair harmful buckets | **P0-1**, **P0-11** |
| 6 | H/D/R for every concept used in a figure caption | **P2-8** caption re-quoting |

---

# Estimated effect on length

Cuts (§2 merge, §3.2 trim, §3.8, §6.3–6.5, §7 merge, three duplicate tellings) recover
roughly **2–2.5 pages**. Additions (pipeline figure, funnel, Fig-4 companion panel,
Table-3 chart, four qualitative plates, domain-attribution table) consume roughly
**1.5–2 pages**. Net roughly neutral, with substantially more of the space spent on
argument and evidence rather than restatement.

TMLR has no strict page budget, but the cuts are worth making regardless of venue —
they are redundancy, not compression.
