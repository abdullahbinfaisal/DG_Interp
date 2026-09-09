# RUNLOG

Append-only record of every command run against this repo by an agent session.
Newest last. Machine-readable counterparts live in `results/runs/<stamp>_<script>/manifest.json`
(argv, resolved config, git SHA + dirty flag, env, timings, sha256 of outputs).

Conventions: see CLAUDE.md §6. Canonical interpreter is
`C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe`, referred to below
as `$PYEXE`. PowerShell only.

---

## 2026-07-30 — establish harness, build first clean score file

Branch `refactor` at `8c3804c` (working tree dirty throughout; nothing was staged
or committed).

### Environment probe

Confirmed the project environment. Bare `python` on PATH resolves to
`C:\Program Files\Python311\python.exe`, which **cannot `import torch`** (missing
`typing_extensions`) and produced three different spurious failures on the same
56 MB JSON (segfault, a nonsense `JSONDecodeError` at char 2e12, and phantom type
errors). The conda env `interpretability` is correct:

    Python 3.11.13, torch 2.7.1+cu126, CUDA available, RTX 4090
    numpy 2.1.2, pandas 2.3.1, timm 1.0.16, overcomplete OK, domainbed OK
    pyyaml/sklearn/plotly/jupyter present; nbconvert MISSING

`nbconvert` missing means notebook cells cannot be executed programmatically —
which is why all work moved to scripts.

### Code added

- `clean_lib/config.py` — frozen dataclasses (`PACSConfig`, `SAEConfig`,
  `RunConfig`) + `PRESETS["erm_resnet_3300"]`. `RunConfig.validate()` hard-fails
  if `score_envs` intersects the held-out target.
- `clean_lib/data.py` — added `Load_PACS_full`: every image, no split, no
  shuffle, no `drop_last`. `Load_PACS` left untouched.
- `clean_lib/provenance.py` — `RunRecorder` (manifest + stdout tee + step
  timing), `set_seed`, `git_state`, `sha256_file`.
- `scripts/build_scores.py` — runs H/D/R into a score JSON.
- `scripts/inspect_scores.py` — integrity checks and H×R population tables.

### Code changed

- `clean_lib/processors/processor.py`
  - `Processor.__init__` gained `split`, `batch_size`, `pacs_root`,
    `limit_batches`; added `Processor.loader()` as the single data entry point.
    Required because H/D/R called `Load_PACS` internally, so there was no way to
    make them read full domains.
  - `from_processor` moved up to `Processor` as a classmethod. The per-subclass
    copies hardcoded five kwargs, so `limit_batches` would not have propagated —
    a smoke run would have silently executed as a full run.
  - **`dump()` rewritten to use `.tolist()`** — see the bug entry below.
- `clean_lib/processors/{H,D,R}.py` — use `self.loader(...)`; dropped the
  redundant `__init__`/`from_processor` overrides.
- `clean_lib/processors/R.py` — `process()` no longer dumps `R_mag` (a duplicate
  of `R`) or `R_sign_consistency`. Per decision, the framework stays at three
  scores; sign consistency remains recoverable from `R_acts` if ever needed.
- `Hc.py` and `M.py` deliberately untouched (experimental, still use `Load_PACS`).

### BUG FOUND AND FIXED — `Processor.dump` memory corruption

    run: results/runs/20260730T110238Z_build_scores   (crashed, no manifest)
    & $PYEXE scripts\build_scores.py --out processed\_smoke.json --limit-batches 2 --force

Exited `-1073741819` (`0xC0000005`, access violation) with no traceback, after H's
data pass completed and after the 2.44 MB template was written.

Isolated with a standalone driver calling `calculate_mean_activations`,
`calculate_invariance` and `dump` separately. `dump` failed at
`to_dumpable(scores_np[cls_idx, concept_idx])` with
`AttributeError: 'NoneType' object has no attribute 'dim'` — a tensor index
returning `None`.

Cause: `dump` recursed per element, performing ~114,688 tensor index + `.item()`
calls for a `(7, 16384)` score and ~344k for a `(7, 16384, 3)` one. Under that
load the process corrupted memory; the phantom `None`, the earlier phantom
`list_iterator`/`function` type errors and the segfault are all the same
underlying failure.

Fix: one `.tolist()` call converts the whole tensor at C level. Output JSON shape
is identical (a `(7, C)` tensor yields floats, `(7, C, D)` yields lists).
Dump time went from *crashing* to 0.3–0.9 s per score.

### Smoke test — PASS

    run: results/runs/20260730T110647Z_build_scores        elapsed 32.2s
    & $PYEXE scripts\build_scores.py --out processed\_smoke.json --limit-batches 2 --force

    H ok 7.6s | D ok 5.8s | R ok 16.9s

`scripts/inspect_scores.py processed\_smoke.json` confirmed:
`R_acts` length **3** (source domains only), **0** `R == -1` sentinels, all eight
score keys present. Populations were tiny and concentrated in `dog` only —
expected, because with `shuffle=False` the first two batches are all
`art_painting/dog`. Correctly not a valid result. Smoke artifacts deleted.

### Full run — build FINAL score file

    & $PYEXE scripts\build_scores.py --out processed\FINAL_ERM_ResNet_3300_T3.json `
        --notes "First clean score file: magnitude R, source domains only, full-domain deterministic loader."

Config: preset `erm_resnet_3300`, ckpt 3300 (non-oracle selection), score domains
`[art_painting, cartoon, photo]`, split `full`, batch_size 64, seed 42.
SAE normalizer: mean 0.242166, std 0.927707.

Purpose: produce the first score file that is *both* source-domain-only *and*
magnitude-R. Supersedes `processed/NEW_ERM_ResNet_3300_T3.json` (source-only but
legacy signed R, 349 sentinels) and all of `archive/paper_results*` (magnitude R
but computed over four domains including the sketch target).

Outcome: **failed twice, then succeeded stage by stage.** See below.

### Crash sequence and what it taught us

1. First full attempt (`results/runs/20260730T121517Z_build_scores`): H succeeded
   (20.3s), D's GPU pass completed, then the process died with **no traceback**.
   Windows Application log: `python.exe` faulting in `python311.dll`, exception
   `0xC0000409` (fail-fast), at the exact second a 0-byte `.tmp` appeared.
   Diagnosis: `json.dumps(..., indent=4)`. CPython only uses the C encoder when
   `indent is None`; with `indent` it falls back to the pure-Python generator
   encoder, which builds tens of millions of small string chunks for a 34 MB
   output. **Fix: `json.dump(data, fh, separators=(",", ":"))`.** Side benefits —
   the finished file is 16.6 MB instead of ~56 MB, and loads much faster.

2. Second attempt (`--only D`): D's GPU pass completed (176.6s), then
   `TypeError: write() argument must be str, not bool` from inside the JSON
   write — a `str` chunk arriving as a `bool`, which is not reachable through
   any `json` dispatch path.

3. Attempting to reproduce that in isolation, `import torch` itself failed with
   `OSError: [WinError 1114] DLL initialization routine failed` loading
   `shm.dll` — on the same interpreter that had imported fine twice and run
   three minutes of CUDA work minutes earlier.

Conclusion: **this machine has intermittent memory instability, not a code bug.**
Five distinct type-confusion failures were observed across the session
(`'function' > int`, `list_iterator`, a tensor index returning `None`, a `str`
chunk arriving as `bool`, and a DLL init failure), all under heavy allocation
churn, all non-deterministic. The user confirmed the machine "crashes quite
frequently". Operational consequence: **run one command at a time, and keep every
stage independently resumable** (`--only`), so a crash costs one stage.

Re-running `--only D` and then `--only R` individually both succeeded.

### Result — `processed/FINAL_ERM_ResNet_3300_T3.json`

    python scripts\inspect_scores.py processed\FINAL_ERM_ResNet_3300_T3.json

    16.6 MB | scores: D, D_counts, H, H_counts, H_mean_acts, R, R_acts, R_counts
    domains in R_acts : 3     <-- source only, correct
    R == -1 sentinels : 0     <-- magnitude R, correct
    R == 0 (inert)    : 111530
    sign-flipping R_acts : 376 (0.33%)
    statistics population (H_counts >= 30) : 1542 pairs

VERDICT: **the first score file that is both source-domain-only and
magnitude-R.** Supersedes `NEW_ERM_ResNet_3300_T3.json` and all of
`archive/paper_results*`. The blocker in CLAUDE.md §5 is cleared.

Key comparison against the old signed-R file (paper §5.3's central claim):

| quantity | old (signed R, 80% split) | new (magnitude R, full domains) |
|---|---|---|
| support >= 30 | 1363 | 1542 |
| non-neutral, tau_D=1e-4 | 217 (128+ / 89-) | 273 (135+ / 138-) |
| high-H supportive also high-R | 72.5% (74/102) | 72.4% (76/105) |
| high-H harmful also high-R | 29.7% (19/64) | 36.0% (32/89) |
| low-H & high-R, non-neutral | 3 | 8 |
| neutral in high-H/high-R cell | 35.2% | 38.4% |

The supportive figure is unchanged; the harmful figure rose, so the asymmetry
weakened from 2.44x to 2.0x but survives. It strengthens with tau: 2.0x at 0.7,
2.4x at 0.8, 3.2x at 0.9. Negative-D pairs grew far more than positive
(89 -> 138 vs 128 -> 135) under full-domain scoring.

Harmful-invariant bucket size (high-H, high-R, D < -1e-4), which is paper 5.5's
central category: **32 pairs at tau=0.7**, 18 at 0.8, 6 at 0.9. Only tau=0.7
leaves a defensible population, and it is also the only value where the
log2/log3 = 0.63 bound argument holds.

---

## 2026-08-26/27 — D1: MMD and DANN cross-backbone comparison

Executes `docs/ABLATIONS_AND_EXPERIMENTS.md` T3.1 / `docs/DIRECTIONS.md` D1 — the
paper's own named falsification test (§7): does an invariance-trained backbone
(MMD, DANN) actually have fewer harmful-invariant concepts than ERM? Extended
beyond D1's original MMD-only scope to include DANN, per user request. Full
writeup of the result: `docs/RESULTS_LEDGER.md`.

### Code added

- `clean_lib/config.py` — `PRESETS["mmd_resnet_1800"]` (backbone_dir
  `PACS_ResNet_Sketch_Test_Only/MMD_ResNet_T3`, ckpt 1800, the non-oracle top-1
  over source envs) and `PRESETS["dann_resnet_5000"]` (same directory pattern,
  ckpt 5000), each pointing at its own standalone (not tied-USAE) SAE.
- `scripts/compare_backbones.py` — new, read-only (no model load, no GPU).
  Loads each backbone's score file, tabulates bucket sizes via the existing
  `build_buckets`, computes the "of high-H {supportive,harmful} pairs, also
  high-R" asymmetry numbers, and runs `scipy.stats.fisher_exact` on
  harmful-invariant and supportive-invariant retention between a reference
  backbone and each other backbone. Writes
  `results/W1_backbone_comparison.json`.

### Code fixed — `Processor.dump()` read the score file in text mode

`clean_lib/processors/processor.py`'s `dump()` opened the existing score file
with `open(path, "r", encoding="utf-8")` before merging in new scores — the
exact text-mode pattern CLAUDE.md already documents as producing spurious
errors on this machine for 50MB+ files (why `inspect_scores.py` and
`clean_lib/masks.py` both read as bytes instead). This one read site was never
updated to match. It surfaced repeatedly while scoring DANN and MMD, as three
different-looking exceptions on different runs — same failure class as the
July session's "five distinct type-confusion failures, all non-deterministic,
under heavy allocation churn", except this one has a fixable code-level
contributor (`R.process()` calls `dump()` three times — `R`, `R_acts`,
`R_counts` — so by the time R runs the file already has H and D merged in and
is at its largest, making R's dump calls the ones most likely to trip it):

    results/runs/20260826T120549Z_build_scores (DANN)  ValueError: Circular reference detected
    results/runs/20260826T122324Z_build_scores (DANN)  TypeError: 'int' object is not iterable
    results/runs/20260827T081353Z_build_scores (MMD)   TypeError: write() argument must be str, not cell

Fix: read as bytes then decode, matching the pattern already used elsewhere —

```python
with builtins.open(path_obj, "rb") as fh:
    data = json.loads(fh.read().decode("utf-8"))
```

### DANN SAE training

Two early attempts failed on a wrong PACS path
(`results/runs/20260819T095754Z_train_sae`, `...095920Z`); a third
(`...135131Z`) failed with `AttributeError: 'DANN' object has no attribute
'network'` — DANN algorithm objects don't expose `.network` the way ERM/MMD do.
Fixed and retrained by the user; succeeded 2026-08-26
(`results/runs/20260826T103022Z_train_sae`, 3049s elapsed): `ckpt=5000`,
`train_envs=[0,1,2]`, `feature_dim=2048 topk=16 nb_concepts=16384 w=7`, output
`SAEs/DANN_ResNet_T3/USAE_ckpt5000_DANN_ResNet_T3.pt`, matching the
`dann_resnet_5000` preset above.

### Score files

Both backbones scored with `--only H` / `--only D` / `--only R` as three
separate process invocations (per-stage resumability against the crash above;
`--force` required from the second call on since the file already exists).

- `processed/FINAL_MMD_ResNet_1800_T3.json` — final successful run
  `results/runs/20260827T083857Z_build_scores`.
- `processed/FINAL_DANN_ResNet_5000_T3.json` — final successful run
  `results/runs/20260827T082226Z_build_scores`.

Both integrity-checked with `inspect_scores.py`: `R_acts` length 3
(source-domains-only, correct), zero `R == -1` legacy sentinels.

### B1–B5 for both backbones

Run into separate `--out-dir` (`results/mmd`, `results/dann`) so as not to
overwrite ERM's `results/summary.csv`. B1 passed for both — SAE reconstruction
within ~0.4–0.5pp of the original backbone on every domain, comparable in
magnitude to ERM's own reconstruction check. B2–B5 replicated the paper's
central causal asymmetry in both backbones: masking harmful buckets recovers
sketch accuracy and beats matched random controls by double-digit margins;
masking supportive buckets collapses it and the control moves the opposite
direction.

### Cross-backbone comparison

`scripts/compare_backbones.py` first attempt
(`results/runs/20260827T083125Z_compare_backbones`) failed `KeyError:
'R_acts'` against a DANN score file whose R stage hadn't yet completed
successfully (see the dump() bug above). Final run
`results/runs/20260827T102151Z_compare_backbones` →
`results/W1_backbone_comparison.json`. Headline result — the harmful-invariant
Fisher exact test, ERM vs. MMD: p=0.048, MMD's rate significantly *higher*
(51.7% vs. 36.0%); ERM vs. DANN: p=0.372, not significant. Neither alignment
objective reduces the paper's central bucket; MMD significantly increases it.
Full result and interpretation: `docs/RESULTS_LEDGER.md`.

### Checkpoint-selection sanity check

Verified directly from `out.txt` (parsed programmatically, not hand-read,
given DANN's log changes header mid-file — an extra `disc_loss` column
appears from step 300 onward): both `mmd_resnet_1800` and `dann_resnet_5000`
are genuine non-oracle selections (top-1 mean accuracy over source envs
[0,1,2] only, matching ERM's own selection protocol). For both backbones the
non-oracle pick also happens to be the oracle-best checkpoint (envs
[0,1,2,3]) — MMD step 1800: non-oracle mean 0.9663, oracle mean 0.9355, both
top-1; DANN step 5000: non-oracle mean 0.9768, oracle mean 0.9406, both
top-1. No leakage, and no "picked a weaker checkpoint by not looking at the
target" concern either.

---

## 2026-08-27 — D7: does R carry information beyond D's magnitude?

Executes `docs/DIRECTIONS.md` D7, revised per its own design caveat: match the
per-pair |D| **distribution** (as `stratified_random_control` already does),
not just the total mass, per `docs/DISCUSSION.md` §5's point that aggregate
mass is a poor predictor of consequence. Run across all three backbones (ERM,
MMD, DANN) rather than ERM alone, since scoring infrastructure for all three
already existed from the D1 work above. Full result and interpretation:
`docs/RESULTS_LEDGER.md` F2.

### Code added

- `clean_lib/masks.py` — extracted the quantile-binning core already used by
  `stratified_random_control` into a shared `_quantile_matched_indices()`
  helper (no behavior change to that function), then added
  `distribution_matched_subset(target, pool, scores, seed)`: draws a
  same-size, same-|D|-shape subset from a *named* candidate bucket (not
  "everything not in target," like the existing control) — e.g. matching
  `S-_hi`'s profile out of the `S-_lo` pool, so masking the matched subset and
  masking the true bucket differ only in R, not in effect size.
- `scripts/run_experiments.py` — new `--stage E3E4`. Two comparisons per
  backbone: harmful (target=`S-_hi`, pool=`S-_lo`, since harm is enriched at
  low R) and supportive (target=`S+_lo`, pool=`S+_hi`, direction reversed
  since `S+_lo`'s mass is below `S+_hi`'s in every class). 3 seeds each,
  reusing the existing `run_bucket`/summary.csv machinery unchanged. `S-_hi`,
  `S-_lo`, `S+_hi`, `S+_lo` themselves were already masked and recorded by B2
  for all three backbones, so only the 6 new matched-subset configs per
  backbone needed running.

### Runs

    python scripts\run_experiments.py --stage E3E4 --preset erm_resnet_3300  --scores processed\FINAL_ERM_ResNet_3300_T3.json   --out-dir results
    python scripts\run_experiments.py --stage E3E4 --preset mmd_resnet_1800  --scores processed\FINAL_MMD_ResNet_1800_T3.json   --out-dir results\mmd
    python scripts\run_experiments.py --stage E3E4 --preset dann_resnet_5000 --scores processed\FINAL_DANN_ResNet_5000_T3.json  --out-dir results\dann

All 18 configs (6 per backbone) succeeded on first attempt — no retries, unlike
the D1 scoring runs above (this stage does no fresh forward-pass scoring, only
masked-accuracy evaluation against already-built score files).

### Result

Harmful side: true `S-_hi` recovers 2.3–5.5x more sketch accuracy than a
magnitude-matched low-R subset of `S-_lo`, in all three backbones (ERM +3.68
vs +1.32; MMD +2.51 vs +1.07; DANN +2.46 vs +0.45). Supportive side: matched
subset costs 8–10x more than the true `S+_lo` in ERM and MMD (consistent with
the harmful-side story); DANN inverted (-18.77 true vs -1.15 matched) but the
matcher only found 30 of 74 needed pairs per-class, so this one is flagged as
unresolved rather than a clean contradiction. Full numbers and the DANN
coverage caveat: `docs/RESULTS_LEDGER.md` F2.
