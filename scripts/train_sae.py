r"""
Train a Sparse Autoencoder (SAE) for one DomainBed backbone checkpoint.

Uses clean_lib.sae.SparseAEs — the current, correct trainer (source-envs-only
Normalizer, tied-checkpoint decode loss). SAE_Train_V2.py is superseded: it
imports lib.gpu_pacs / lib.loaders, which no longer match this repo.

Checkpoint selection mirrors build_scores.py / CheckpointManager: ranked by
mean env{0,1,2}_out_acc from out.txt (non-oracle, source-domains-only). Pass
--ckpt to override with a specific step.

Canonical invocation (PowerShell, from repo root, interpretability env active):

  python scripts\train_sae.py --backbone-dir PACS_ResNet_Sketch_Test_Only\MMD_ResNet_T3 --save-dir SAEs\MMD_ResNet_T3

Smoke test first (~1-2 min, result is NOT valid):

  python scripts\train_sae.py --backbone-dir PACS_ResNet_Sketch_Test_Only\MMD_ResNet_T3 --save-dir SAEs\_smoke --epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clean_lib.provenance import RunRecorder, set_seed
from clean_lib.checkpoints import CheckpointManager
from clean_lib.sae import SparseAEs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone-dir", required=True,
                    help=r"e.g. PACS_ResNet_Sketch_Test_Only\MMD_ResNet_T3")
    p.add_argument("--ckpt", type=int, default=None,
                    help="override the auto-selected checkpoint step")
    p.add_argument("--train-envs", type=int, nargs="+", default=[0, 1, 2],
                    help="envs to train the SAE on; must exclude the held-out target")
    p.add_argument("--feature-dim", type=int, default=2048)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--nb-concepts", type=int, default=2048 * 8)
    p.add_argument("--w", type=int, default=7, help="spatial side of the feature map")
    p.add_argument("--rearrange-string", default="n c w h -> (n w h) c")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--flag", default=None,
                    help="save-file prefix; defaults to 'ckpt<step>'. "
                         "Final name is SAEs/USAE_<flag>_<algo>_<arch>_T<testenvs>.pt")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()

    if 3 in args.train_envs:
        raise SystemExit(
            "train_envs includes env 3 (sketch), the held-out target domain. "
            "The SAE must never be trained on it."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manager = CheckpointManager(directory=args.backbone_dir)
    print(f"[ckpt] algorithm={manager.algorithm} arch={manager.architecture} "
          f"train_envs={manager.trainenvs} test_envs={manager.testenvs}")

    if args.ckpt is not None:
        ckpt = args.ckpt
        print(f"[ckpt] using --ckpt override: step {ckpt}")
    else:
        top_steps, accs = manager.get_top_k_checkpoints(envs=[0, 1, 2], k=1)
        if not top_steps:
            raise SystemExit(f"no checkpoints found in {args.backbone_dir}/out.txt")
        ckpt = top_steps[0]
        print(f"[ckpt] auto-selected step {ckpt} (non-oracle, source-envs-only): {accs[ckpt]}")

    flag = args.flag or f"ckpt{ckpt}"

    manifest_cfg = {
        "backbone_dir": args.backbone_dir,
        "ckpt": ckpt,
        "train_envs": args.train_envs,
        "feature_dim": args.feature_dim,
        "topk": args.topk,
        "nb_concepts": args.nb_concepts,
        "w": args.w,
        "rearrange_string": args.rearrange_string,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "flag": flag,
        "save_dir": str(save_dir),
        "seed": args.seed,
    }

    with RunRecorder("train_sae", config=manifest_cfg, notes=args.notes) as rec:
        if args.epochs < 250:
            print(f"[warn] --epochs {args.epochs}: SMOKE RUN, resulting SAE is NOT valid for scoring")

        set_seed(args.seed)

        with rec.step("load_backbone"):
            manager.load_checkpoints([ckpt])

        with rec.step("configure_sae"):
            sae_manager = SparseAEs(
                feature_dim=args.feature_dim,
                topk=args.topk,
                nb_concepts=args.nb_concepts,
                rearrange_string=args.rearrange_string,
                checkpointManager=manager,
                train_envs=args.train_envs,
                w=args.w,
            )
            sae_manager.configure_training(learning_rate=args.learning_rate)
            print(f"[sae] nb_concepts={args.nb_concepts} topk={args.topk} feature_dim={args.feature_dim}")

        with rec.step("train"):
            sae_manager.train(
                flag=flag,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_dir=str(save_dir),
                dataset="PACS",
            )

        test_envs_str = "".join(str(e) for e in manager.testenvs)
        out_path = save_dir / f"USAE_{flag}_{manager.algorithm}_{manager.architecture}_T{test_envs_str}.pt"
        rec.add_output(out_path)
        rec.add(ckpt=ckpt, flag=flag, smoke=bool(args.epochs < 250))

    print("done")


if __name__ == "__main__":
    main()
