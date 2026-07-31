r"""
Build a concept score file: H, D and R for every (class, concept) pair.

Scores are estimated on SOURCE DOMAINS ONLY. The script refuses to run if the
configured score_envs include the held-out target domain.

Canonical invocation (PowerShell, from repo root):

  $PY = "C:\Users\sproj_ha\miniconda3\envs\interpretability\python.exe"
  & $PY scripts\build_scores.py --out processed\FINAL_ERM_ResNet_3300_T3.json

Smoke test the whole path first (~30s, result is NOT valid):

  & $PY scripts\build_scores.py --out processed\_smoke.json --limit-batches 2 --force

Flags:
  --only H,D,R      run a subset of processors (default all three)
  --limit-batches N cap batches per domain; smoke testing only
  --force           allow writing to an existing output path
  --split full|train  'train' reproduces the legacy 80% shuffled split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from clean_lib.config import get_preset, PRESETS
from clean_lib.provenance import RunRecorder, set_seed
from clean_lib.checkpoints import CheckpointManager
from clean_lib.sae import SparseAEs, Normalizer
from clean_lib.processors.processor import Processor
from clean_lib.processors.H import H
from clean_lib.processors.D import D
from clean_lib.processors.R import R

# The SAE checkpoint was pickled from a notebook, where Normalizer lived in
# __main__. Without this shim torch.load raises AttributeError.
setattr(sys.modules["__main__"], "Normalizer", Normalizer)

PROCESSORS = {"H": H, "D": D, "R": R}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="erm_resnet_3300", choices=sorted(PRESETS))
    p.add_argument("--out", required=True, help="output score JSON path")
    p.add_argument("--only", default="H,D,R", help="comma-separated subset of H,D,R")
    p.add_argument("--split", default=None, choices=["full", "train"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_preset(args.preset)
    overrides = {}
    if args.split is not None:
        overrides["split"] = args.split
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    cfg.validate()  # hard-fails if the target domain leaked into score_envs

    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(steps) - set(PROCESSORS)
    if unknown:
        raise SystemExit(f"--only contains unknown processors: {sorted(unknown)}")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Pass --force to overwrite, or choose a new path.\n"
            "Score files are never clobbered silently: a half-overwritten file would "
            "mix scores from two configurations."
        )

    manifest_cfg = cfg.to_dict()
    manifest_cfg["_resolved"] = {
        "out": str(out_path),
        "only": steps,
        "limit_batches": args.limit_batches,
        "score_domains": cfg.score_domains,
    }

    with RunRecorder("build_scores", config=manifest_cfg, notes=args.notes) as rec:
        if args.limit_batches:
            print(f"[warn] --limit-batches {args.limit_batches}: SMOKE RUN, "
                  f"output is NOT a valid result")

        set_seed(cfg.seed)

        print(f"[cfg] preset={cfg.name} ckpt={cfg.ckpt} split={cfg.split} "
              f"seed={cfg.seed} batch_size={cfg.batch_size}")
        print(f"[cfg] score domains (sources only) = {cfg.score_domains}")
        print(f"[cfg] processors = {steps}")
        print(f"[cfg] out = {out_path}")

        with rec.step("load_backbone"):
            manager = CheckpointManager(directory=cfg.backbone_dir)
            manager.load_checkpoints([cfg.ckpt])
            print(f"[ckpt] algorithm={manager.algorithm} arch={manager.architecture} "
                  f"train_envs={manager.trainenvs} test_envs={manager.testenvs}")

        with rec.step("load_sae"):
            sae_manager = SparseAEs(
                feature_dim=cfg.sae.feature_dim,
                topk=cfg.sae.topk,
                nb_concepts=cfg.sae.nb_concepts,
                rearrange_string=cfg.sae.rearrange_string,
                checkpointManager=manager,
                train_envs=list(cfg.train_envs),
                w=cfg.sae.w,
            )
            sae_manager.load_checkpoint(cfg.sae.checkpoint_path)
            sae = sae_manager.get_sae(cfg.ckpt)
            print(f"[sae] nb_concepts={cfg.sae.nb_concepts} topk={cfg.sae.topk} "
                  f"from {cfg.sae.checkpoint_path}")
            print(f"[sae] normalizer mean={float(sae.normalizer.mean):.6f} "
                  f"std={float(sae.normalizer.std):.6f}")

        base = Processor(
            sae_manager=sae_manager,
            ckpt=cfg.ckpt,
            process_domains=list(cfg.score_envs),
            file_path=str(out_path),
            dataset="PACS",
            split=cfg.split,
            batch_size=cfg.batch_size,
            pacs_root=cfg.pacs.root,
            limit_batches=args.limit_batches,
        )

        for name in steps:
            with rec.step(name):
                PROCESSORS[name].from_processor(base).process()

        rec.add_output(out_path)
        rec.add(
            processors_run=steps,
            score_domains=cfg.score_domains,
            smoke=bool(args.limit_batches),
        )

    print("done")


if __name__ == "__main__":
    main()
