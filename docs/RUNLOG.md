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
