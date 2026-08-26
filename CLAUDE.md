# CLAUDE.md — Sparse Concept Diagnostics for Domain Generalization

Session primer for this repo. Read this first, then `docs/EXPERIMENTS.md` (what to
run and why), `docs/RESULTS_DRAFT.md` (written-up results + patch list for the
paper) and `docs/TMLR_Journal_Submissions__1_.md` (paper draft).

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
The *maximum* normalised entropy attainable while only 2 domains are active is an
even split over those two, `log2/log3 = 0.63`. So **any** τ > 0.63 implies nonzero
activation (for `H`) or nonzero effect (for `R`) in **all three** source domains.

0.7 is chosen as the **lowest** round value above that bound — the most permissive
threshold that still carries the guarantee, hence the largest bucket populations.
Stricter thresholds *keep* the guarantee (they are strictly stronger) but shrink
the populations: at 0.9 the harmful-invariant bucket holds 6 pairs. Only τ ≤ 0.63
loses the argument. **Corrected 2026-08-01** — this was previously written as "0.7
is the largest value at which the guarantee holds" and "higher thresholds lose that
guarantee", both of which are backwards.

### The central open risk

`R` may be a proxy for the **sign** of `D`. Low-R pairs are enriched ~2.5× in
negative-D pairs, and masking negative-D pairs raises `p(y)` *by construction*
(that is how D is defined). The mass-matched experiments (E3/E4 in
`docs/EXPERIMENTS.md` §10) exist to separate "R carries independent information" from
"R correlates with harm". **E3 is decisive. Do not write up the direction of a
sign-based effect as a finding.**

**RESOLVED 2026-07-31, and the answer is the opposite of the old hypothesis.**
The project previously believed low-R concepts were load-bearing — that "the
model's class evidence substantially consists of domain-contingent concepts".
Clean data refutes this. Concentrated support (`S+_lo`, 54 pairs) is
indistinguishable from a matched random mask when ablated (−0.65 vs −1.07), and
retaining it alone collapses the model to 25.0%. Distributed support alone reaches
94.0%, and adding the concentrated pairs on top moves that only to 95.0%.

The replacement claim is cleaner and aligns with the §5.3 asymmetry: **the model
runs on invariant, distributed support; domain-contingent support is close to
redundant.** See `docs/RESULTS_DRAFT.md` §5.4.

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
                            the deleted context.md. §10 holds the deferred
                            experiment specs, §11 the record of that brief.
docs/RESULTS_DRAFT.md       written-up §5.1–5.6 + patch list for the paper
docs/DISCUSSION.md          the "so what": insights from existing results only
docs/DIRECTIONS.md          claims the results point at + how to settle them
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
label-free null result). In the notebook these appear directly as
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

**The micro/macro discrepancy is resolved and measured.** `_build_report` computed
an *unweighted mean over classes* → that is the 83.4x figure. Confirmed on disk by
B1: original model sketch **80.25 micro / 83.56 macro**, SAE reconstruction
**80.22 / 83.63**. The paper's Table 2 value of 80.29 was micro. The gap is driven
by `house` (80 images) and `person` (160) being tiny in sketch while `dog` (772) is
both the largest and the worst class. Always report both.

**Score-file population sizes** (`processed/FINAL_ERM_ResNet_3300_T3.json`,
support floor `H_counts ≥ 30`): **1542** pairs pass the floor; **273** are
non-neutral at τ_D = 1e-4 (135 supportive, 138 harmful); 1269 neutral. The
superseded `NEW_*` file gave 1363 / 217 (128 / 89) / 1146 — the difference is
scoring on full domains rather than 80% splits, which raises `H_counts` and so
admits more pairs.

**Evaluation and scoring both use full domains** (`Load_PACS_full`): every image,
no split, `shuffle=False`, `drop_last=False`. Deterministic. The legacy path
(`Load_PACS`, still present and used by `Hc.py`/`M.py`) returns an 80% shuffled
train split with `drop_last=True`, which dropped a *different* ~7 sketch images per
call — the ±0.08pp jitter visible across notebook cells (83.40–83.48). Note also
that the 80% subset of `Load_PACS([one_domain])` is **not** nested inside the one
from `Load_PACS([three_domains])` used for SAE training, so any legacy per-domain
scoring partially overlapped SAE training images. Full-domain scoring makes that
moot but means scoring now includes the 20% the SAE never trained on.

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

Consequence: every low-R result in the deleted brief folds those 349
sign-flipping pairs into "low-R". Inside the gated population (support ≥ 30,
|D| > 1e-4, n = 217), **15 pairs are sentinels** (5 positive-D, 10 negative-D) —
6.9%. They are *sign-inconsistent*, not concentrated. The brief's typology tables
reproduce exactly off this file with sentinels counted in the `R < 0.7` column,
which is what pins that file as their provenance. On the clean file the same
population has 376 sign-flipping pairs overall (0.33%) and they carry real R values
rather than a sentinel.

**(b) Every `archive/paper_results*` run computed H, D and R over all four
domains.** `R_acts` has length 4 in all five archived score files, and `dump.txt`
confirms `--process_domains 0 1 2 3` with an `R scores [sketch]` pass. Those
tables violate the paper's central methodological commitment. Treat them as
**indicative only, never as paper evidence** — including the tempting
`keep_robust_support_only` row (sketch 46.6% macro, with elephant/guitar/house at
0.00). That row is not merely contaminated, it is **wrong**: the clean measurement
of the same intervention is **94.02%** macro (§8). Do not reuse it.

**RESOLVED 2026-07-30.** `processed/FINAL_ERM_ResNet_3300_T3.json` is the first
score file that is both source-only and magnitude-R: `R_acts` length 3, zero
`R == -1`, 1542 pairs above a support floor of 30. It supersedes both problem
files. Use it for everything; treat `NEW_*` and `archive/paper_results*` as
history only. Verify any new score file with `scripts/inspect_scores.py`.

---

## 6. Code-running policy

**The user runs all scripts.** Do not execute them from an agent session — they
crash unpredictably when launched that way (see §6.1). Hand over exact commands
instead, one stage at a time, and wait for the output.

**Preferred shell: Anaconda Prompt with the `interpretability` env activated.**
There, bare `python` is correct:

```
cd /d c:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp
python scripts\build_scores.py --out processed\FINAL_ERM_ResNet_3300_T3.json
```

Only in a *non-activated* shell must the interpreter be spelled out, because bare
`python` then resolves to `C:\Program Files\Python311\python.exe`, which cannot
even `import torch` (missing `typing_extensions`) and produced three *different*
spurious failures on the same 56 MB JSON:

```powershell
$PYEXE = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
```

Avoid the Bash tool on this box. In PowerShell, note that variables are
case-insensitive: never use `$py` and `$PYEXE` in one script, they are the same
variable.

**Scripts do the work; the notebook never writes to disk.** Anything producing
`processed/` or `results/` lives in `scripts/*.py` and imports from `clean_lib`.
`nbconvert` is not installed, so notebook cells cannot be executed here at all —
this is a hard constraint, not a preference.

### 6.1 This machine is unstable under load

Five distinct type-confusion failures were observed in one session — `'function'
> int`, `list_iterator`, a tensor index returning `None`, a `str` chunk arriving
as `bool`, and `WinError 1114` on `import torch` — all non-deterministic, all
under heavy allocation churn, plus a `0xC0000409` fail-fast. Two consequences:

- **Keep every stage independently resumable** (`--only`, `--stage`) so a crash
  costs one stage, never a whole run.
- **Minimise allocation churn.** Prefer one vectorised operation over a loop of
  small ones; see the `dump()` note in §3.

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
   `run_experiments.py` emitting `results/E_*.json` + `results/summary.csv`
   (schema in `docs/EXPERIMENTS.md` §10). The notebook stays for exploration only.
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
(29.7 → 36.0), so the asymmetry is real but weaker than the old brief claimed.

Only 8 of 273 non-neutral pairs are low-H and high-R, so "consistency
presupposes broad activation" still holds. 38.4% of *neutral* pairs sit in the
high-H/high-R cell, so R still must be read behind a magnitude gate.

**τ recommendation: τ_H = τ_R = 0.7.** At 0.9 the central harmful-invariant
category has 6 pairs, which cannot carry a headline claim; 0.7 is the lowest round
value clearing the log2/log3 = 0.63 bound, so it keeps the three-domain guarantee
with the largest possible populations. (Every τ > 0.63 carries the guarantee — see
§3. The earlier claim that 0.7 was "the only value where the bound holds" was
wrong.) Not yet written into `config.py` pending the τ_D = 1e-3 sensitivity check.

Per-class non-neutral counts (support ≥ 30, |D| > 1e-4) — note `dog` has both the
most harmful pairs and the worst sketch accuracy, while `person` has 32 harmful
pairs and 95.62% accuracy, so the relationship is not monotone (that is E10's job):

| class | support | D>0 | D<0 |
|---|---|---|---|
| dog | 286 | 33 | 50 |
| elephant | 208 | 23 | 16 |
| giraffe | 185 | 15 | 10 |
| guitar | 150 | 12 | 3 |
| horse | 243 | 24 | 24 |
| house | 144 | 9 | 3 |
| person | 326 | 19 | 32 |

### Intervention results (2026-07-31, clean file, full-domain eval)

Baseline = SAE reconstruction, nothing masked: **sketch 83.63 macro / 80.22 micro**;
sources 99.19–99.78 macro. Original model sketch 83.56 / 80.25 — reconstruction
drop ≤ 0.05pp on every domain, so §5.1 is settled.

All masks are gated (`|D| > 1e-4`) and support-floored (`H_counts ≥ 30`), which is
why they are small. Control = mean of 3 size- and |D|-histogram-matched random
masks; margin = target Δ − control Δ on sketch macro.

| Mask | pairs | Σ\|D\| | sketch macro | Δ | control Δ | margin |
|---|---|---|---|---|---|---|
| all harmful | 138 | 0.250 | 90.87 | +7.24 | −20.98 | **+28.2** |
| `S-_hi` distributed harm | 35 | 0.087 | 87.32 | +3.69 | −3.12 | +6.8 |
| harmful-invariant | 32 | 0.083 | 86.96 | +3.33 | −2.10 | +5.4 |
| `S-_lo` concentrated harm | 103 | 0.163 | 86.17 | +2.54 | −5.40 | +7.9 |
| inert only (`R == 0`) | 111530 | 0.069 | 83.71 | +0.08 | — | — |
| `S+_lo` concentrated support | 54 | 0.025 | 82.98 | −0.65 | −1.07 | +0.4 |
| `S+_hi` distributed support | 81 | 0.237 | 44.55 | −39.08 | +1.06 | −40.1 |
| all supportive | 135 | 0.262 | 23.63 | −60.00 | +2.50 | −62.5 |

Keep-only (ablate everything except the named set):

| Keep only | kept | sketch macro | sketch micro |
|---|---|---|---|
| 76 uniformly random pairs (3 seeds) | 76 | 14.29 / 14.25 / 14.82 | 4.07 / 4.05 / 4.84 |
| `S+_lo` | 54 | 25.00 | 18.25 |
| robust support (H,R ≥ 0.7, D > τ_D) | 76 | 94.02 | 93.56 |
| all supportive | 135 | 95.00 | 94.25 |

**What these establish.**
1. The categories are functional and the controls are decisive — every harmful
   bucket beats its matched random control by 5–28 points, and the control for
   "all harmful" moves in the *opposite* direction (−20.98 vs +7.24).
2. **No denoising confound.** Masking all 111,530 inert pairs moves sketch +0.08,
   despite their aggregate |D| mass exceeding `S-_hi`'s.
3. **No label injection in the keep-only rows.** Keeping 76 *random* pairs gives
   exactly chance (14.29%, everything predicted `person`), so the 94.02 is a
   property of the bucket.
4. Every intervention leaves the three source domains within one point while moving
   sketch by up to 60 — the diagnostic is source-estimated but its consequences
   appear in the unseen domain.

**Known limit:** robust support carries ~10× the mass of `S+_lo` (0.236 vs 0.025),
so keep-only comparisons conflate R with effect magnitude. Not separable without
mass-matched buckets, which are deliberately out of MVP scope.

### The old §8 table was mislabelled — do not reinstate it

The superseded table's rows read "Mask `R < 0.8 ∧ D < 0` → 93.47 (+10.03)". That
run actually masked **114,609 pairs**: the notebook did `filter("R",[(0.7,None)])`
then `filter("D",[(1e-4,None)])`, which *keeps* high-R supportive pairs, and
`get_mask()` then ablates the complement. It was **keep-only distributed support**,
not a harmful-concept mask — the polarity inversion documented in §3.

Re-read correctly it reproduces: old 93.47 vs clean `keep_robust_support` 94.02.
Likewise "Mask `R < 0.8 ∧ D > 0` → 13.41" was keep-only *harmful*, hence the
collapse. And "Mask all `D < 0` → 96.66" was ungated and un-floored, so it swept in
tens of thousands of pairs rather than 138. None of those rows is comparable to the
gated buckets above; the numbers were never wrong, the labels were.

---

## 9. Immediate next steps

The MVP experiment programme is **complete** (Group A + B1–B5, 2026-07-31). The
plan is `docs/EXPERIMENTS.md`; the written-up §5.1–5.6 plus a patch list is
`docs/RESULTS_DRAFT.md`.

1. ~~Full-domain loader~~, ~~regenerate score JSON~~, ~~promote the evaluator~~,
   ~~threshold selection (τ_D = 1e-4, τ_H = τ_R = 0.7)~~, ~~Group A + B1–B5~~.
2. **Review `docs/RESULTS_DRAFT.md`** — six subsections, six tables.
3. **Apply its patch list** to `docs/TMLR_Journal_Submissions__1_.md`: replace
   §5.1–5.3, merge §5.8 into a new §5.4, repurpose §5.5, replace §5.6, cut §5.7.
   Eight subsections become six.
4. Purge the stale numbers the patch list enumerates — §4.2's "exclusively source
   domains" clause, §4.3's threshold TODOs, §5.3's sweep / Fig. 4 / Table 3, and
   §3.6's unused sign-consistency score.
5. Re-quote H/D/R for the concept grids in `analysis/` from the clean file; the
   folder names encode scores from the superseded file.
6. ~~τ_D sensitivity~~ **done 2026-08-06** (`answer_critique.py --only q8`): τ_D is
   a nuisance parameter (ratio 2.04/2.01/1.80 over 1e-5/1e-4/1e-3), but the
   **support floor is load-bearing** — the §5.3 asymmetry runs 2.49 → 1.31 across
   floors 10 → 100. Direction never inverts; magnitude does. Disclosed in §4.4.
7. **E12a, the τ curve is not monotone** — deferred to the next draft, spec in
   `docs/EXPERIMENTS.md` §10. The fine sweep shows a plateau at τ = 0.65–0.70 and a
   turnover at τ = 0.95 (ratio 3.21 → 2.86), almost certainly because the harmful
   denominator drops to single digits. **Handled for this draft by wording only** —
   "strengthens monotonically" is gone from §1/§4.4/§5.3/§7 and Figure 6 shades the
   thin region. No number changed. Confirm or truncate next time.
8. **E14, support-stratified asymmetry** — the one open cheap experiment, spec in
   `docs/EXPERIMENTS.md` §10. Harmful pairs are systematically thinner than
   supportive ones and `R` is biased toward 0 at low support, so part of the §5.3
   asymmetry may be a support artifact. Stratifying by support would convert §4.4's
   disclosure into a control. Read-only, ~1 s.
9. Mass-matched buckets if a reviewer challenges the R-vs-magnitude conflation
   acknowledged in §5.4.

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
