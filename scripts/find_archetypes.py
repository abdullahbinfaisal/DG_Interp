r"""
Find inspectable concepts for each archetype in docs/DISCUSSION.md, then render
them.

The framework's advantage over activation statistics is that its categories are
lookable-at. This script closes the gap between counting them and seeing them.

  python scripts\find_archetypes.py                       # list candidates only
  python scripts\find_archetypes.py --render 2979 4501     # render specific concepts
  python scripts\find_archetypes.py --render-top 1         # render the top of each archetype

Archetypes located (see DISCUSSION.md section 2):

  bystander      H, R high, |D| ~ 0, present in many classes
                 -> "stable but useless". FIG-1.
  robust_witness H, R high, D > tau_d
                 -> the 76 pairs that alone give 94% on the target domain.
  saboteur       H, R high, D < -tau_d
                 -> invisible to alignment; 32 pairs, 22.6% of target errors.
  local_harm     H high, R low, D < -tau_d
                 -> fires everywhere, hurts in one domain.
  local_support  H high, R low, D > tau_d
                 -> fires everywhere, helps in one domain. FIG-3.
  janus          the SAME concept is a robust witness for one class and a
                 consistent saboteur for another
                 -> FIG-2, the chest/forelimb figure. The most important one.

Rendering writes 7 grids per concept (one per class) to results/figures/<id>/ via
clean_lib.visualize, and prints a caption block with the clean-file scores so a
figure caption can be written without re-deriving numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from clean_lib.config import get_preset
from clean_lib.masks import load_scores


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", default="processed/FINAL_ERM_ResNet_3300_T3.json")
    p.add_argument("--preset", default="erm_resnet_3300")
    p.add_argument("--tau-d", type=float, default=1e-4)
    p.add_argument("--tau-h", type=float, default=0.7)
    p.add_argument("--tau-r", type=float, default=0.7)
    p.add_argument("--support-floor", type=int, default=30)
    p.add_argument("--top", type=int, default=6, help="candidates to list per archetype")
    p.add_argument("--render", type=int, nargs="*", default=None,
                   help="concept ids to render (7 grids each)")
    p.add_argument("--render-top", type=int, default=0,
                   help="render the top N candidates of every archetype")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--n-images", type=int, default=None,
                   help="subsample images per class when rendering (faster)")
    return p.parse_args()


def profile_lines(s, concept, sup, tau_d):
    """Per-class H/D/R for one concept, as printable lines."""
    out = []
    for k, name in enumerate(s.class_names):
        if not sup[k, concept]:
            out.append(f"      {name:<9}  (below support floor)")
            continue
        d = s.D[k, concept]
        role = "SUPPORTS" if d > tau_d else ("HARMS   " if d < -tau_d else "neutral ")
        acts = ", ".join(f"{v:+.2e}" for v in s.R_acts[k, concept])
        out.append(
            f"      {name:<9}  {role}  D={d:+.3e}  H={s.H[k, concept]:.3f}  "
            f"R={s.R[k, concept]:.3f}  n={int(s.H_counts[k, concept]):>4}  "
            f"D_d=[{acts}]"
        )
    return out


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    s = load_scores(args.scores, class_names=list(cfg.pacs.class_names))

    sup = s.support(args.support_floor)
    hiH = s.H >= args.tau_h
    hiR = s.R >= args.tau_r
    pos = sup & (s.D > args.tau_d)
    neg = sup & (s.D < -args.tau_d)
    absd = np.abs(s.D)

    tau_d = args.tau_d
    print(f"[scores] {args.scores}")
    print(f"[cfg] tau_D={tau_d:g} tau_H={args.tau_h} tau_R={args.tau_r} "
          f"support_floor={args.support_floor}")

    archetypes = {}

    # --- janus: robust witness for one class, consistent saboteur for another ---
    rw = pos & hiH & hiR          # (K, C)
    sb = neg & hiH & hiR
    janus_c = np.flatnonzero(rw.any(axis=0) & sb.any(axis=0))
    if janus_c.size:
        contrast = s.D[:, janus_c].max(axis=0) - s.D[:, janus_c].min(axis=0)
        order = np.argsort(contrast)[::-1]
        archetypes["janus  (FIG-2, the key figure)"] = [
            (int(janus_c[i]), None) for i in order
        ]

    def rank_pairs(selection, key, descending=True):
        ks, cs = np.nonzero(selection)
        if ks.size == 0:
            return []
        vals = key[ks, cs]
        order = np.argsort(vals)
        if descending:
            order = order[::-1]
        return [(int(cs[i]), int(ks[i])) for i in order]

    # --- bystander: invariant, consistent, no effect, broad class presence ---
    inert = sup & hiH & hiR & (absd <= 1e-6)
    breadth = inert.sum(axis=0)                       # in how many classes
    cand = np.flatnonzero(breadth >= 4)
    if cand.size:
        order = np.argsort(breadth[cand] * 1e6 + s.H[:, cand].mean(axis=0))[::-1]
        archetypes["bystander  (FIG-1, stable but useless)"] = [
            (int(cand[i]), None) for i in order
        ]

    archetypes["robust_witness"] = rank_pairs(rw, s.D)
    archetypes["saboteur  (invisible to alignment)"] = rank_pairs(sb, absd)
    archetypes["local_harm  (H high, R low)"] = rank_pairs(neg & hiH & ~hiR, absd)
    archetypes["local_support  (FIG-3, H high, R low)"] = rank_pairs(
        pos & hiH & ~hiR, s.D
    )

    for name, items in archetypes.items():
        print(f"\n{'=' * 78}\n  {name}   ({len(items)} candidates)\n{'=' * 78}")
        for concept, k in items[: args.top]:
            if k is None:
                print(f"\n  concept {concept}")
            else:
                print(f"\n  concept {concept}   (ranked on class '{s.class_names[k]}')")
            for line in profile_lines(s, concept, sup, tau_d):
                print(line)

    # ------------------------------------------------------------------ render
    to_render = list(args.render or [])
    if args.render_top:
        for items in archetypes.values():
            to_render += [c for c, _ in items[: args.render_top]]
    to_render = sorted(set(to_render))

    if not to_render:
        print("\nNo rendering requested. Use --render <ids> or --render-top N.")
        return

    # Imported here so listing candidates needs no GPU or model load.
    import torch
    from clean_lib.checkpoints import CheckpointManager
    from clean_lib.sae import SparseAEs, Normalizer
    from clean_lib.visualize import visualize_concept_on_class
    setattr(sys.modules["__main__"], "Normalizer", Normalizer)

    print(f"\n[render] loading model for {len(to_render)} concept(s): {to_render}")
    manager = CheckpointManager(directory=cfg.backbone_dir)
    manager.load_checkpoints([cfg.ckpt])
    sae_manager = SparseAEs(
        feature_dim=cfg.sae.feature_dim, topk=cfg.sae.topk,
        nb_concepts=cfg.sae.nb_concepts,
        rearrange_string=cfg.sae.rearrange_string,
        checkpointManager=manager, train_envs=list(cfg.train_envs), w=cfg.sae.w,
    )
    sae_manager.load_checkpoint(cfg.sae.checkpoint_path)

    out_root = Path(args.out_dir)
    for concept in to_render:
        save_dir = out_root / f"concept_{concept}"
        print(f"\n[render] concept {concept} -> {save_dir}")
        for k in range(s.n_classes):
            visualize_concept_on_class(
                concept=concept, class_idx=k, sae_manager=sae_manager,
                ckpt=cfg.ckpt, save_dir=str(save_dir), n_images=args.n_images,
            )

        # Caption block, so figure captions never re-derive numbers by hand.
        print(f"\n  --- caption data for concept {concept} ---")
        for line in profile_lines(s, concept, sup, tau_d):
            print(line)

    print(f"\ndone. grids under {out_root}")


if __name__ == "__main__":
    main()
