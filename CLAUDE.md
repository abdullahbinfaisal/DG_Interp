# CLAUDE.md — Sparse Concept Diagnostics for Domain Generalization

Session primer for this repo. Read this first, then `docs/context.md` (experiment
spec) and `docs/TMLR_Journal_Submissions__1_.md` (paper draft).

---

## 1. What this project is

A **post-hoc interpretability paper** for domain generalization (DG), targeting TMLR.
Title: *When Invariance Is Not Enough: Sparse Concept Diagnostics for Domain
Generalization*.

We do **not** propose a training method or a test-time intervention. We take a
frozen, already-trained DG model, decompose its final feature map with a sparse
autoencoder, and score each **(class, concept)** pair on three axes. The argument
is that activation invariance — the thing most DG objectives optimise — is
necessary but not sufficient, because an invariant concept can still be neutral,
harmful, or useful in only some domains.

**Setup.** PACS, 4 domains (art_painting, cartoon, photo, sketch), 7 classes
(dog, elephant, giraffe, guitar, horse, house, person). **Sketch is the held-out
target**; art/cartoon/photo are sources. Backbone: ERM ResNet-50 from DomainBed,
frozen. SAE: Top-K, 2048-dim input (7×7×2048 feature map → 49 spatial tokens per
image), dictionary 16,384, top-16 per token.

### The three scores (all class-conditional, all source-domains-only)

| Score | Meaning | Range | Computed in |
|---|---|---|---|
| `H(k,c)` | **Activation invariance.** Normalised entropy of mean activation across domains. Says *where a concept fires*, nothing about usefulness. | [0,1] | `processors/H.py` |
| `D(k,c)` | **Discriminative effect.** Zero concept `c`'s column in SAE code, decode, denormalise, classify; mean drop in `p(y=k)` over images of class `k` where `c` is active. `D>0` = supports class `k`; `D<0` = hurts it. | signed, tiny (1e-5..1e-2) | `processors/D.py` |
| `R(k,c)` | **Discriminative consistency.** Normalised entropy of `\|D_d(k,c)\|` across domains. High = effect spread across domains; low = concentrated in one or two. | [0,1] | `processors/R.py` |

**Class-conditionality is the whole point.** The same latent has different H/D/R
for different classes. Never collapse a mask to a set of concept indices without
keeping the class dimension.

**Entropy bound that carries an argument.** With 3 source domains, `log|D| = log 3`.
An even split over only 2 domains gives `log2/log3 = 0.63 < 0.7`. So at τ=0.7,
`H ≥ 0.7` implies nonzero activation in **all three** source domains, and `R ≥ 0.7`
implies nonzero effect in all three. This bound only works at 0.7 — restate it if
thresholds move.

### The central open risk

`R` may be a proxy for the **sign** of `D`. Low-R pairs are enriched ~2.5× in
negative-D pairs, and masking negative-D pairs raises `p(y)` *by construction*
(that is how D is defined). The mass-matched experiments (E3/E4 in
`docs/context.md`) exist to separate "R carries independent information" from
"R correlates with harm". **E3 is decisive. Do not write up the direction of a
sign-based effect as a finding.**

A second, more interesting result is already emerging: low-R concepts are not
junk. Masking the harmful ones helps a lot (+10.03pp sketch); masking the
supportive ones collapses the model to chance. If that holds, the model's class
evidence substantially *consists of* domain-contingent concepts — a
concept-level mechanism for the discrimination–invariance tradeoff.

---

## 2. Repo blueprint

```
clean_lib/                  ← the live library (all new code goes here)
  config.py                 frozen dataclasses + PRESETS; RunConfig.validate()
                            hard-fails if the target domain leaks into scoring
  provenance.py             RunRecorder (manifest + stdout tee + step timing)
  data.py                   PACSDataset, Load_PACS (80% split), Load_PACS_full
                            (deterministic, every image); pacs_domains, pacs_envs
  checkpoints.py            CheckpointManager: DomainBed ckpt loading + selection
  sae.py                    Normalizer, SparseAEs (multi-checkpoint "USAE" trainer)
  utils.py                  extract_features, make_uniform_concept_mask
  analyzer.py               Analyzer: file-backed filter chain → boolean mask
  visualize.py              visualize_concept_on_class (top-8 grid per domain)
  processors/
    processor.py            Processor base: template, .loader(), .dump(),
                            .from_processor() (inherited by all subclasses)
    H.py                    H, H_mean_acts, H_counts
    D.py                    D, D_counts
    R.py                    R (magnitude entropy), R_acts, R_counts
    Hc.py                   EXPERIMENTAL: entropy across classes, not domains
    M.py                    EXPERIMENTAL: signal / interference / contrast
    B.py                    stub ("## TO DO")

scripts/
  build_scores.py           run H/D/R into a score JSON (--only, --limit-batches,
                            --force, --split)
  inspect_scores.py         integrity checks + H×R population tables per D regime

refactor.ipynb              your working notebook (97 cells). GITIGNORED, and it
                            cannot be executed here — nbconvert is not installed.
                            Left untouched; new exploration goes in a thin notebook
                            that imports from clean_lib.
generate_tmlr_results.py    1711-line CLI harness: typology, masks, sensitivity,
                            CSV tables. See §5 — its outputs are contaminated.
docs/EXPERIMENTS.md         WHAT TO RUN: every experiment with rationale, expected
                            outcome, falsifier, and the risk register. Supersedes
                            context.md §5 where they disagree.
docs/context.md             original experiment brief E0–E13 + output schema
docs/RUNLOG.md              append-only log of every command an agent ran
docs/TMLR_Journal_Submissions__1_.md    paper draft (many TODOs)

results/runs/<stamp>_<script>/          manifest.json + stdout.log per run
processed/                  score JSONs (gitignored)
  FINAL_ERM_ResNet_3300_T3.json   CURRENT: source-only + magnitude R
  NEW_ERM_ResNet_3300_T3.json     superseded: source-only, legacy signed R — §5
  eval_ERM_ResNet_3300_T3.json    empty template; exists only to build a
                                  Processor with process_domains=[0,1,2,3]
PACS_ResNet_Sketch_Test_Only/   DomainBed runs: ERM_ResNet_T3, MMD_ResNet_T3
SAEs/normalization_testing/      USAE_ERM_Multi_test_3300_2100.pt
archive/                    old lib/ + five paper_results* runs (contaminated)
analysis/                   concept visualisation grids by typology bucket
```

`.gitignore` excludes `*.ipynb`, `SAEs/`, `processed/`, `PACS_ResNet_*/`,
`domainbed/`, `invariances/`, `logs/`. **The notebook is not version-controlled.**

### Pipeline

```
CheckpointManager(dir)          parses "ERM_ResNet_T3" → algorithm/arch/split
  .get_top_k_checkpoints(envs)  ranks out.txt rows by mean env{e}_out_acc
  .load_checkpoints([...])      → {step: DomainBed algorithm object}
        │                       (ResNet global_pool replaced with Identity so
        │                        featurizer emits the 7×7×2048 map)
        ▼
SparseAEs(feature_dim=2048, topk=16, nb_concepts=16384,
          rearrange_string="n c w h -> (n w h) c", train_envs=[0,1,2], w=7)
  .load_checkpoint(path)        → {step: TopKSAE with .normalizer attached}
        │
        ▼
Processor(sae_manager, ckpt, process_domains, file_path)
  H / D / R .from_processor(p).process()   → dumps into the score JSON
        │
        ▼
Analyzer(score_json).filter(...).get_mask()   → bool (7, 16384)
        │
        ▼
MaskedAccuracyEvaluator.calculate_masked_accuracy(mask)  → per-domain/per-class
```

**Checkpoint selection.** `get_top_k_checkpoints(envs=[0,1,2])` → **step 3300**
(non-oracle, source-domains-only selection — this is what the paper uses).
`envs=[0,1,2,3]` → step 2100 (oracle; do not use for headline numbers).

**The USAE is a tied multi-checkpoint SAE.** `SparseAEs.train` rotates the
encoder across checkpoints and decodes each code with *every* SAE, summing L1
losses. So the dictionaries for steps 2100 and 3300 are **aligned by
construction** — cross-checkpoint latent comparison is legitimate here, unlike
across independently trained SAEs. The paper draft never mentions this; it should,
because it is what would justify a checkpoint-level Table 5.

---

## 3. Conventions and gotchas (read before touching masks)

**Mask polarity is inverted, twice.** `Analyzer.filter(...)` **keeps** matching
pairs. `Analyzer.get_mask()` then returns `True` = **mask this pair out**,
computed as the complement of the surviving set against the full grid reloaded
from the original file. So a filter chain reads as "keep X" and the resulting
mask ablates *everything except* X. `analysis.filter("R", [(0.7, None)])` keeps
R ≥ 0.7 and therefore **masks R < 0.7**.

**Masking protocol.** For image `i` with true label `y_i`, the masked set is
`{c : (y_i, c) ∈ S}`. A 2-D mask is applied as `concept_mask[y]` then
`repeat_interleave(h*w)` over spatial tokens, zeroing those columns in SAE code
space before decode → denormalise → classify. This is an **oracle diagnostic**
by design (it uses the true label). That is fine and defensible for a controlled
ablation — just always label it as oracle in output, and always run the
label-free variants (E7, E8) alongside.

**Intervals are inclusive on both ends;** `None` means open. `(None, None)` keeps
everything. Adjacent buckets like `(0,1e-3)` and `(1e-3,None)` therefore overlap
at exactly 1e-3.

**`make_uniform_concept_mask(mask, reduce)`** — `"any"` masks a concept
everywhere if *any* class masks it (extremely aggressive; collapses the model to
14.29% = 1/7). `"all"` masks only if *all* classes do (~15.9k pairs; the null
result in context.md §4.4). In the notebook these appear directly as
`mask.any(dim=0)` / `mask.all(dim=0)`.

**The count filter must never touch a forward pass.** `H_counts < 30` as a
*class-conditional* mask leaks labels: a concept firing rarely for class `k` but
often for `k'` is a confusion signal, and masking it for `k`-labelled images
deletes evidence for the competing class using the true label. It is worth
+2.7pp on sketch, entirely from that leak. Keep a support floor **only** as an
inclusion criterion when computing statistics (R is an entropy over three
per-domain estimates; a cell with 4 images gives a meaningless `D_d`). Name the
two uses distinguishably, e.g. `support_floor_stats=30` and no filter argument at
all in the eval path.

**Never compute H, D or R using sketch.** `process_domains=[0,1,2]` for all
scoring. `[0,1,2,3]` only when constructing the evaluator's Processor, so it
knows to *report* four domains.

**Report micro and macro every time.** Never emit a bare "accuracy". Never
average the four domains together — three are in-distribution and one is the
held-out domain that carries the entire argument.

**Watch for chance-level collapse.** 1/7 = 14.29%. A configuration landing near
that has been *erased*, not degraded. Emit the predicted-label histogram so the
difference is visible. Under the mask-all-positive-D condition, six of seven
classes go to exactly 0.00% and only `person` survives — the model defaults to a
single label.

**`rearrange_string` is `"n c w h -> (n w h) c"` in `SparseAEs`** but `D.py` and
`R.py` hardcode `"n c h w -> (n h w) c"`. Equivalent only because h == w == 7.
Latent bug for non-square feature maps.

**PACS root is hardcoded** at `C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS`
in both `data.py` and `visualize.py`.

**Class indices** follow alphabetical directory order:
`0 dog, 1 elephant, 2 giraffe, 3 guitar, 4 horse, 5 house, 6 person`.

**`Processor.dump` must stay vectorised.** It originally recursed per element —
~114k tensor index + `.item()` calls per score — which *corrupted memory* on this
machine: a tensor index returned `None`, and the full run died with an access
violation (`0xC0000005`). It now uses a single `.tolist()`. If you ever see
inexplicable `NoneType`/`list_iterator` errors or silent segfaults, suspect a
per-element loop over a large tensor, not the data.

**Read big score JSONs as bytes**, i.e. `json.loads(open(p,"rb").read().decode())`.
Text-mode reads of the 50 MB+ files have produced spurious `JSONDecodeError`s here.

### PACS image counts (needed for micro-averaging)

| Domain | dog | elephant | giraffe | guitar | horse | house | person | total |
|---|---|---|---|---|---|---|---|---|
| art_painting | 379 | 255 | 285 | 184 | 201 | 295 | 449 | 2048 |
| cartoon | 389 | 457 | 346 | 135 | 324 | 288 | 405 | 2344 |
| photo | 189 | 202 | 182 | 186 | 199 | 280 | 432 | 1670 |
| sketch | 772 | 740 | 753 | 608 | 816 | 80 | 160 | 3929 |

---

## 4. Verified facts (established by inspection — do not re-derive)

**The micro/macro discrepancy is resolved.** `_build_report` computes an
*unweighted mean over classes* → that is the 83.4x figure. Weighting the sketch
per-class accuracies (48.71, 95.98, 77.97, 96.86, 82.24, 87.88, 94.62) by the
counts above gives **80.30%**, matching the paper's Table 2 value of 80.29. The
gap is driven by `house` (80 images) and `person` (160) being tiny in sketch
while `dog` (772) is both the largest and the worst class. E0 is confirmed
analytically; it still needs emitting to disk under the new eval protocol.

**Score-file population sizes** (`processed/NEW_ERM_ResNet_3300_T3.json`,
support floor `H_counts ≥ 30`): 1363 pairs pass the floor; 217 are non-neutral at
τ_D = 1e-4 (128 positive, 89 negative); 1146 neutral. These match
`docs/context.md` §4.6 exactly, so that file is the provenance of every §4 number.

**Evaluation currently runs on each domain's 80% *train* split with
`drop_last=True`.** `Load_PACS` returns `(train_loader, test_loader)` and every
call site takes `dataloader, _ = ...`. For sketch that is 49×64 = 3136 of 3929
images, and because `shuffle=True` a *different* ~7 images are dropped per call.
This is exactly the ±0.08pp baseline jitter seen across notebook cells
(83.40–83.48). Also, the 80% subset of `Load_PACS([one_domain])` is **not** nested
inside the one from `Load_PACS([three_domains])` used for SAE training, so
per-domain scoring splits partially overlap SAE training images.

**`Normalizer` was built with `domains=None`**, so μ/σ came from one 1024-image
batch spanning all four domains, sketch included. Numerically negligible (two
scalars). But §4.2 of the draft claims the SAE is trained and analysed
"exclusively on the source domains", which is then literally false — **soften
that clause**, or pass `domains=[0,1,2]` next time a Normalizer is constructed.
No retraining needed.

---

## 5. Two contamination problems — the current blocker

**(a) The source-only score file has legacy *signed* R.**
`processed/NEW_ERM_ResNet_3300_T3.json` contains exactly **349 pairs with
R = −1.0**, and exactly **349 pairs with mixed-sign `R_acts`** — a 1:1 match with
the old `R.py` sentinel (`entropy = -1` whenever the signed-normalised
distribution went negative). The file also lacks `R_mag` / `R_sign_consistency`,
which the current `R.py` writes.

Consequence: every low-R result in `docs/context.md` §4 folds those 349
sign-flipping pairs into "low-R". Inside the gated population (support ≥ 30,
|D| > 1e-4, n = 217), **15 pairs are sentinels** (5 positive-D, 10 negative-D) —
6.9%. They are *sign-inconsistent*, not concentrated. The §4.6 tables reproduce
exactly off this file with sentinels counted in the `R < 0.7` column.

**(b) Every `archive/paper_results*` run computed H, D and R over all four
domains.** `R_acts` has length 4 in all five archived score files, and `dump.txt`
confirms `--process_domains 0 1 2 3` with an `R scores [sketch]` pass. Those
tables violate the paper's central methodological commitment. Treat them as
**indicative only, never as paper evidence** — including the otherwise-tempting
`keep_robust_support_only` row (sketch 46.6% macro, with elephant/guitar/house at
0.00) that would fill the draft's empty "test robust sufficiency" cell.

**RESOLVED 2026-07-30.** `processed/FINAL_ERM_ResNet_3300_T3.json` is the first
score file that is both source-only and magnitude-R: `R_acts` length 3, zero
`R == -1`, 1542 pairs above a support floor of 30. It supersedes both problem
files. Use it for everything; treat `NEW_*` and `archive/paper_results*` as
history only. Verify any new score file with `scripts/inspect_scores.py`.

---

## 6. Code-running policy

**Interpreter is pinned. Never call bare `python`.**

```powershell
$PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
```

Bare `python` on PATH resolves to `C:\Program Files\Python311\python.exe`, which
cannot even `import torch` (missing `typing_extensions`) and produced three
*different* spurious failures on the same 56 MB JSON. **PowerShell only** — the
Bash tool also misbehaved on this box. Note PowerShell variables are
case-insensitive: never use `$py` and `$PYEXE` in one script, they are the same
variable.

**Scripts do the work; the notebook never writes to disk.** Anything producing
`processed/` or `results/` lives in `scripts/*.py` and imports from `clean_lib`.
`nbconvert` is not installed, so notebook cells cannot be executed here at all —
this is a hard constraint, not a preference.

**Every run is recorded twice.**
- `results/runs/<UTC-stamp>_<script>/manifest.json` — argv, resolved config, git
  SHA + dirty flag, interpreter, torch/CUDA versions, seed, step timings, and the
  sha256 + size of every output. Written by `RunRecorder` even on failure.
- `docs/RUNLOG.md` — append-only prose entry: command, purpose, outcome, verdict.

`processed/` and `results/` are gitignored, so manifests are the only durable link
from a number to the code that produced it.

**Smoke before real.** `--limit-batches 2` exercises the whole path in ~30 s. Any
run using it is explicitly not a valid result and says so in its own output.

**No silent clobbering.** Scripts refuse to overwrite an existing `--out` without
`--force`. `Processor.dump` writes to `.tmp` then `os.replace`.

**Long runs go background**, polled via the run's own `stdout.log`. Do not pipe a
background command through `Select-Object`/`Where-Object` — that buffers until the
process exits and you lose interim progress.

**I never stage and never commit.** All changes stay in the working tree for you
to review. No `git add`, no `git commit`, ever.

---

## 7. Decisions taken (2026-07-30)

1. **Recompute magnitude-R, source-only.** Rerun the current `R.py` with
   `process_domains=[0,1,2]` into a fresh score file. All §4 numbers get
   recomputed against it. ~3 min GPU.
2. **Do not report sign-consistency `S`.** Keep the framework at three scores
   exactly as the draft describes. (`R.py` currently also dumps `R_mag` and
   `R_sign_consistency`; harmless, but they must not appear in any table. Trimming
   those dumps would also shrink the score JSON.)
3. **Evaluate on full domains with `drop_last=False`.** No split, no dropped
   batch — deterministic, and matches Table 2's population. Needs a new loader
   path in `data.py`. Batch counts become 32/37/27/62 at batch_size 64.
4. **Scope: ERM checkpoint 3300 only.** Table 5 becomes single-row or is cut.
   MMD_ResNet_T3 and step 2100 exist on disk but are out of scope for now.
5. **Promote the harness into `clean_lib`.** Move `MaskedAccuracyEvaluator` out
   of the notebook into `clean_lib/eval.py`, add mask-builder helpers, and write
   `run_experiments.py` emitting `results/E*.json` + `results/summary.csv` per
   `docs/context.md` §6. The notebook stays for exploration only.
6. **Thresholds: decide from data after the recompute.** Do not hardcode τ_D,
   τ_H, τ_R yet. Candidates: τ_D ∈ {1e-4, 1e-3}, τ_H = τ_R ∈ {0.7, 0.8}. Pick
   whatever keeps bucket populations viable, then report full sensitivity (E12).
   Note the `log2/log3 = 0.63` argument only holds at 0.7.
7. **`M.py`, `Hc.py`, and RSM/HIM/DCM/conflict mass stay experimental.** Leave
   as-is; not blocking. Paper §§5.4–5.7 and Tables 4–5 remain open.
8. **Normalizer leak: disclose by softening one sentence in §4.2.** No retrain.
   §4.2 currently claims the SAE is trained and analysed "exclusively on the
   source domains"; that clause is what needs softening. Numerically the leak is
   two scalars and does not matter.
9. **Config lives in `clean_lib/config.py`** as frozen dataclasses, overridable
   from the CLI. Not YAML.
10. **Never commit or stage.** Everything stays in the working tree for review.
11. **`refactor.ipynb` is left alone.** New exploration goes in a thin notebook
    importing from `clean_lib`. Notebooks stay gitignored.
12. **Scoring uses full domains too**, not just evaluation — same
    `Load_PACS_full` path, `split="full"`. Consequence: `H_counts` is now over
    every image in each source domain rather than 80%, so support-floor
    populations shift up roughly 25% versus the old file. The support floor of 30
    may want revisiting against the new distribution.

---

## 8. Current state of results

### Score distributions on the clean file (FINAL, τ_D = 1e-4, support ≥ 30)

1542 pairs pass the support floor; 273 are non-neutral (135 supportive, 138
harmful); 1269 neutral.

| τ_H = τ_R | high-H supportive also high-R | high-H harmful also high-R | ratio | harmful-invariant bucket |
|---|---|---|---|---|
| 0.7 | 72.4% (76/105) | 36.0% (32/89) | 2.0× | 32 pairs |
| 0.8 | 65.2% (60/92) | 27.3% (18/66) | 2.4× | 18 pairs |
| 0.9 | 49.4% (39/79) | 15.4% (6/39) | 3.2× | 6 pairs |

**The §5.3 asymmetry survived the recompute** — invariant support distributes,
invariant harm concentrates — and it strengthens monotonically with τ, which is a
better robustness story than any single threshold. The supportive figure barely
moved from the old signed-R file (72.5 → 72.4); the harmful figure rose
(29.7 → 36.0), so the asymmetry is real but weaker than §4 of context.md claims.

Only 8 of 273 non-neutral pairs are low-H and high-R, so "consistency
presupposes broad activation" still holds. 38.4% of *neutral* pairs sit in the
high-H/high-R cell, so R still must be read behind a magnitude gate.

**τ recommendation: τ_H = τ_R = 0.7.** At 0.9 the central harmful-invariant
category has 6 pairs, which cannot carry a headline claim; and 0.7 is the only
value where the log2/log3 = 0.63 bound argument holds. Not yet written into
`config.py` pending the τ_D = 1e-3 sensitivity check.

Per-class non-neutral counts (support ≥ 30, |D| > 1e-4) — note `dog` has both the
most harmful pairs and the worst sketch accuracy, while `person` has 32 harmful
pairs and 94.62% accuracy, so the relationship is not monotone (that is E10's job):

| class | support | D>0 | D<0 |
|---|---|---|---|
| dog | 286 | 33 | 50 |
| elephant | 208 | 23 | 16 |
| giraffe | 185 | 15 | 10 |
| guitar | 150 | 12 | 3 |
| horse | 243 | 24 | 24 |
| house | 144 | 9 | 3 |
| person | 326 | 19 | 32 |

### Masking results — still stale

All figures below are **macro** sketch accuracy from the signed-R source-only
file on the 80% split. They need recomputation against FINAL under decision 3.

| Configuration | Sketch (macro) | Δ |
|---|---|---|
| Baseline (no mask) | 83.44–83.48 | — |
| Mask all `D < 0` | 96.66 | +13.20 |
| Mask all `D > 0` | 5.67 | −77.79 (collapse) |
| Mask `R < 0.8` | 91.08 | +7.61 |
| Mask `R < 0.8 ∧ D < 0` | 93.47 | +10.03 |
| Mask `R < 0.8 ∧ D > 0` | 13.41 | collapse |
| Label-free `R < 0.7` for all classes (~15k pairs) | 83.35 | −0.12 (null) |

Per-class sketch baseline: dog 48.71, elephant 95.98, giraffe 77.97, guitar
96.86, horse 82.24, house 87.88, person 94.62.

Note the ordering: masking harmful concentrated concepts gives +10.03, masking
*all* low-R gives only +7.61. The rest of the low-R population is costing ~2.4
points — i.e. it contains real class evidence.

The R-threshold sweep rises to +7.61 at τ=0.8 then **falls** to +6.39 at 0.9.
The turnover matters as much as the peak: if ablation were simply good for a weak
domain the curve would keep rising. But most of the effect arrives by τ=0.1
(+3.35), where the pairs being removed are largely inert (111,729 pairs have
R exactly 0) — so part of that early gain may be dictionary denoising rather than
a claim about concentrated concepts. The gated sweep (|D| > 1e-4) preserves the
gain, which argues against pure noise removal, but this needs redoing on clean R.

**Two unrelated criteria (low-R, low-count) both give large gains
class-conditionally and nothing uniformly.** The common factor is
class-conditionality, not the criterion. That is a finding, not a failure: almost
no concept is concentrated for *every* class at once. Concepts are
domain-contingent for particular classes, not in general.

---

## 9. Immediate next steps

1. ~~Full-domain loader~~ — done (`Load_PACS_full`).
2. ~~Regenerate the score JSON~~ — done and verified:
   `processed/FINAL_ERM_ResNet_3300_T3.json`.
3. Finish threshold selection: run `inspect_scores.py --tau-d 1e-3` and compare
   bucket viability against 1e-4, then write the chosen τ_D, τ_H, τ_R into
   `config.py`. τ_H = τ_R = 0.7 is already indicated (see §8).
4. Promote `MaskedAccuracyEvaluator` → `clean_lib/eval.py`; add micro **and**
   macro to `_build_report`, plus a predicted-label histogram.
5. Run E0 → E1 → E2 → E3 (blocking, in that order) per `docs/context.md` §5,
   writing the §6 JSON schema plus `summary.csv`.
6. Regenerate the typology heatmap and the §4.6 counts against the clean file.
7. Then E4–E12 in any order. E9 and E10 need no masking at all — the per-domain
   effects already exist in `R_acts`.

---

## 10. Open questions

- **`processors/B.py`** is a one-line stub. What is B meant to compute?
- **Paper source of truth.** Is `docs/TMLR_Journal_Submissions__1_.md` the working
  draft, or is there a LaTeX source elsewhere? Several figures referenced in the
  text (Fig. 1–5) exist only as PNGs in `analysis/viz/` and `archive/paper_results/figures/`.
- **Submission timeline**, which sets how much of E4–E13 is realistic.
- **`generate_tmlr_results.py`**: fix its `process_domains` default and reuse its
  typology/sensitivity/CSV machinery, or leave it archived and build fresh in
  `clean_lib`? Decision 5 implies fresh, but it contains a lot of working code
  (mask rules, sensitivity sweeps, representative-concept selection) worth porting.
- **`analysis/results/`** holds DomainNet and OfficeHome DomainBed logs, but
  `data.py` is PACS-only. Are those in scope at all?
