r"""
Train a Sparse Autoencoder (SAE) for one DomainBed backbone checkpoint.

Uses clean_lib.sae.SparseAEs — the current, correct trainer (source-envs-only
Normalizer, tied-checkpoint decode loss). SAE_Train_V2.py is superseded: it
imports lib.gpu_pacs / lib.loaders, which no longer match this repo.

Supports PACS, VLCS and OfficeHome via --dataset (default PACS). VLCS and
OfficeHome share PACS's domain/class/*.jpg folder layout, so only the domain
name/root-dir tables in clean_lib.data (DATASET_DOMAINS / DATASET_ROOTS) differ
per dataset — no other lib code is dataset-specific.

Checkpoint selection mirrors build_scores.py / CheckpointManager: ranked by
mean out_acc over the backbone's own declared train envs from out.txt
(non-oracle, source-domains-only). Pass --ckpt to override with a specific
step. --train-envs defaults to the backbone's declared train envs (everything
but its held-out test env(s), parsed from the "..._T<digits>" directory name)
and is validated to never include a held-out env, whatever it is for this
dataset/backbone.

Canonical invocation (PowerShell, from repo root, interpretability env active):

  python scripts\train_sae.py --backbone-dir PACS_ResNet_Sketch_Test_Only\MMD_ResNet_T3 --save-dir SAEs\MMD_ResNet_T3
  python scripts\train_sae.py --dataset VLCS --backbone-dir VLCS_ResNet_T3 --save-dir SAEs\VLCS_ERM_ResNet_T3
  python scripts\train_sae.py --dataset OfficeHome --backbone-dir OfficeHome_ResNet_T1 --save-dir SAEs\OfficeHome_ERM_ResNet_T1

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
    p.add_argument("--dataset", choices=["PACS", "VLCS", "OfficeHome"], default="PACS",
                    help="which dataset the backbone was trained on")
    p.add_argument("--backbone-dir", required=True,
                    help=r"e.g. PACS_ResNet_Sketch_Test_Only\MMD_ResNet_T3")
    p.add_argument("--ckpt", type=int, default=None,
                    help="override the auto-selected checkpoint step")
    p.add_argument("--train-envs", type=int, nargs="+", default=None,
                    help="envs to train the SAE on; must exclude the held-out target(s). "
                         "Defaults to the backbone's own declared train envs "
                         "(parsed from its directory name).")
    p.add_argument("--feature-dim", type=int, default=2048)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--nb-concepts", type=int, default=2048 * 8)
    p.add_argument("--w", type=int, default=7, help="spatial side of the feature map")
    p.add_argument("--rearrange-string", default="n c w h -> (n w h) c")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers for the one caching pass over the images. "
                         "0 decodes single-threaded; raise it on datasets with large "
                         "source images (VLCS/LabelMe averages ~1 MB each).")
    p.add_argument("--cache-dir", default=".feature_cache",
                    help="where the cached backbone activations are written; deleted on "
                         "completion unless --keep-cache")
    p.add_argument("--keep-cache", action="store_true",
                    help="keep the cached activations, so a rerun skips the caching pass")
    p.add_argument("--flag", default=None,
                    help="save-file prefix; defaults to 'ckpt<step>'. "
                         "Final name is SAEs/USAE_<flag>_<algo>_<arch>_T<testenvs>.pt")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manager = CheckpointManager(directory=args.backbone_dir, dataset=args.dataset)
    print(f"[ckpt] dataset={args.dataset} algorithm={manager.algorithm} arch={manager.architecture} "
          f"train_envs={manager.trainenvs} test_envs={manager.testenvs}")

    train_envs = args.train_envs if args.train_envs is not None else manager.trainenvs
    leaked = [e for e in train_envs if e in manager.testenvs]
    if leaked:
        raise SystemExit(
            f"--train-envs {train_envs} includes held-out test env(s) {leaked} "
            f"(backbone_dir {args.backbone_dir!r} declares test_envs={manager.testenvs}). "
            "The SAE must never be trained on the held-out target domain."
        )
    args.train_envs = train_envs

    if args.ckpt is not None:
        ckpt = args.ckpt
        print(f"[ckpt] using --ckpt override: step {ckpt}")
    else:
        top_steps, accs = manager.get_top_k_checkpoints(envs=manager.trainenvs, k=1)
        if not top_steps:
            raise SystemExit(f"no checkpoints found in {args.backbone_dir}/out.txt")
        ckpt = top_steps[0]
        print(f"[ckpt] auto-selected step {ckpt} (non-oracle, source-envs-only): {accs[ckpt]}")

    flag = args.flag or f"ckpt{ckpt}"

    manifest_cfg = {
        "dataset": args.dataset,
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
        "num_workers": args.num_workers,
        "cache_dir": args.cache_dir,
        "keep_cache": args.keep_cache,
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
                dataset=args.dataset,
            )
            sae_manager.configure_training(learning_rate=args.learning_rate,
                                           num_workers=args.num_workers)
            print(f"[sae] nb_concepts={args.nb_concepts} topk={args.topk} feature_dim={args.feature_dim}")

        with rec.step("train"):
            sae_manager.train(
                flag=flag,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_dir=str(save_dir),
                dataset=args.dataset,
                num_workers=args.num_workers,
                cache_dir=args.cache_dir,
                keep_cache=args.keep_cache,
            )

        test_envs_str = "".join(str(e) for e in manager.testenvs)
        out_path = save_dir / f"USAE_{flag}_{manager.algorithm}_{manager.architecture}_T{test_envs_str}.pt"
        rec.add_output(out_path)
        rec.add(ckpt=ckpt, flag=flag, smoke=bool(args.epochs < 250))

    print("done")


if __name__ == "__main__":
    main()
