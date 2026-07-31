"""
python generate_tmlr_results.py --backbone_dir ./PACS_ResNet_Sketch_Test_Only/ERM_ResNet_T3 --sae_checkpoint ./SAEs/normalization_testing/USAE_ERM_Multi_test_3300_2100.pt --out_dir ./paper_results_signCons --ckpts 3300 --process_domains 0 1 2 3 --train_envs 0 1 2 --d_abs_tau 0.001 --min_d_count 5 --tau_h 0.8 --tau_r 0.8 --tau_s 0.75
""""""
Generate all TMLR paper Results-section outputs for Sparse Concept Diagnostics.

Final revised version for the single-checkpoint paper setting.

Key fixes:
1. No KMeans D-labeling.
2. D labels are sign-respecting:
      D > +tau_D  => support
      D < -tau_D  => harm
      otherwise   => neutral
3. Adds minimum active-count filtering via --min_d_count.
4. Computes:
      R_mag(k,c): magnitude consistency over |D_d(k,c)|
      S(k,c): sign consistency across domain-wise D_d(k,c)
5. Uses sign consistency as a GATE:
      high consistency iff R_mag >= tau_R and S >= tau_S
6. Keeps the 8-category typology unchanged.
7. Adds original and SAE-reconstruction baseline rows to interventions.csv.
8. Reports both overall and macro accuracy.
9. Adds sanity-check outputs for validating D labels and consistency gating.

Run from repo root, e.g.

python scripts/generate_tmlr_results.py \
  --backbone_dir ./PACS_ResNet_Sketch_Test_Only/ERM_ResNet_T3 \
  --sae_checkpoint ./SAEs/normalization_testing/USAE_ERM_Multi_test_3300_2100.pt \
  --out_dir ./paper_results_signed_consistency \
  --ckpts 3300 \
  --process_domains 0 1 2 3 \
  --train_envs 0 1 2 \
  --d_abs_tau 0.001 \
  --min_d_count 5 \

Assumptions:
- The SAE checkpoint is a torch-saved dictionary keyed by checkpoint step.
- clean_lib is importable from the repo root.
- PACS domain ids follow:
    0 = art_painting
    1 = cartoon
    2 = photo
    3 = sketch
- Main ResNet feature maps are C x H x W with H = W = 7.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from timm.layers import SelectAdaptivePool2d

from clean_lib.checkpoints import CheckpointManager
from clean_lib.sae import SparseAEs, Normalizer
from clean_lib.processors.processor import Processor
from clean_lib.processors.H import H
from clean_lib.processors.D import D
from clean_lib.processors.R import R
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features

try:
    from clean_lib.processors.M import M
    HAS_M = True
except Exception:
    HAS_M = False


# Compatibility shim for SAE checkpoints saved from notebooks where
# Normalizer was pickled as __main__.Normalizer.
setattr(sys.modules["__main__"], "Normalizer", Normalizer)

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PACS_DOMAINS = {
    0: "art_painting",
    1: "cartoon",
    2: "photo",
    3: "sketch",
}

CLASS_NAMES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]

# key = H_bit, Consistency_bit, D_positive_bit
# Consistency_bit = 1 iff R_mag >= tau_R AND S >= tau_S.
BUCKET_NAMES = {
    "111": "robust_support",
    "110": "harmful_invariant",
    "101": "domain_contingent_support",
    "100": "domain_contingent_harm",
    "011": "domain_specific_consistent_support",
    "010": "domain_specific_consistent_harm",
    "001": "domain_specific_contingent_support",
    "000": "domain_specific_contingent_harm",
}


# ---------------------------------------------------------------------
# JSON score utilities
# ---------------------------------------------------------------------

def load_score_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def infer_nb_concepts(data: dict) -> int:
    return max(int(c) for c in data["0"].keys()) + 1


def read_score(
    json_path: Path,
    name: str,
    nb_classes: int = 7,
    nb_concepts: Optional[int] = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Reads a score saved by clean_lib.processors.processor.Processor.dump.

    Supports scalar scores:
        H, D, R, D_counts, H_counts, R_counts, R_mag, R_sign_consistency
        shape -> (classes, concepts)

    Supports vector scores:
        H_mean_acts, R_acts, M_profiles, etc.
        shape -> (classes, concepts, vector_dim)
    """
    data = load_score_json(json_path)

    if nb_concepts is None:
        nb_concepts = infer_nb_concepts(data)

    sample = None
    for k in range(nb_classes):
        for c in range(nb_concepts):
            obj = data.get(str(k), {}).get(str(c), {})
            if name in obj:
                sample = obj[name]
                break
        if sample is not None:
            break

    if sample is None:
        raise KeyError(f"Could not find score '{name}' in {json_path}")

    if isinstance(sample, list):
        vec_dim = len(sample)
        arr = np.full((nb_classes, nb_concepts, vec_dim), fill_value, dtype=np.float32)
    else:
        arr = np.full((nb_classes, nb_concepts), fill_value, dtype=np.float32)

    for k in range(nb_classes):
        row = data.get(str(k), {})
        for c in range(nb_concepts):
            obj = row.get(str(c), {})
            if name not in obj:
                continue
            val = obj[name]
            arr[k, c] = np.array(val, dtype=np.float32)

    return arr


def write_score(json_path: Path, name: str, arr: np.ndarray) -> None:
    """
    Adds a score array back into the same clean_lib JSON structure.
    """
    data = load_score_json(json_path)
    nb_classes = arr.shape[0]
    nb_concepts = arr.shape[1]

    for k in range(nb_classes):
        if str(k) not in data:
            data[str(k)] = {}
        for c in range(nb_concepts):
            if str(c) not in data[str(k)]:
                data[str(k)][str(c)] = {}
            val = arr[k, c]
            if isinstance(val, np.ndarray):
                data[str(k)][str(c)][name] = val.tolist()
            elif hasattr(val, "tolist"):
                data[str(k)][str(c)][name] = val.tolist()
            else:
                data[str(k)][str(c)][name] = float(val)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# Model / SAE loading
# ---------------------------------------------------------------------

def build_sae_manager(args, ckpts: List[int]) -> SparseAEs:
    backbone_manager = CheckpointManager(directory=args.backbone_dir)
    backbone_manager.load_checkpoints(set(ckpts))

    try:
        sae_manager = SparseAEs(
            feature_dim=args.feature_dim,
            topk=args.topk,
            nb_concepts=args.nb_concepts,
            rearrange_string=args.rearrange_string,
            checkpointManager=backbone_manager,
            train_envs=args.train_envs,
            w=args.spatial_w,
        )
    except TypeError:
        sae_manager = SparseAEs(
            feature_dim=args.feature_dim,
            sae_dim=args.sae_dim,
            topk=args.topk,
            nb_concepts=args.nb_concepts,
            rearrange_string=args.rearrange_string,
            checkpointManager=backbone_manager,
            train_envs=args.train_envs,
            w=args.spatial_w,
        )

    sae_manager.load_checkpoint(args.sae_checkpoint)
    return sae_manager


# ---------------------------------------------------------------------
# Processor execution
# ---------------------------------------------------------------------

# def run_processors_for_ckpt(args, sae_manager: SparseAEs, ckpt: int, score_json: Path) -> None:
#     """
#     Runs H, D, R, and optionally M.

#     The existing R processor is still used because it saves R_acts, i.e.
#     the domain-wise effects D_d(k,c). We then compute:
#       - R_mag from |D_d(k,c)|
#       - R_sign_consistency from the signed D_d(k,c)
#     """
#     print(f"\n=== Running processors for checkpoint {ckpt} ===")
#     base = Processor(
#         sae_manager=sae_manager,
#         ckpt=ckpt,
#         process_domains=args.process_domains,
#         file_path=str(score_json),
#         dataset="PACS",
#     )

#     H.from_processor(base).process()
#     D.from_processor(base).process()
#     R.from_processor(base).process()

#     if HAS_M and args.run_m:
#         M.from_processor(base).process()

#     compute_fixed_r_scores(score_json, eps=args.eps)

def compute_and_write_domain_effects_for_r(
    args,
    sae_manager: SparseAEs,
    ckpt: int,
    score_json: Path,
) -> None:
    """
    Computes domain-wise discrimination effects:

        R_acts[k,c,d] = D_d(k,c)

    where D_d(k,c) is the mean probability drop for class k, concept c,
    restricted to active images from domain d.

    This replaces clean_lib.processors.R.R.process(), avoiding the old
    signed-R JSON dump that crashes before R_acts is saved.
    """
    print(f"Computing domain-wise effects R_acts directly for checkpoint {ckpt}")

    backbone = sae_manager.get_backbone(ckpt).to(device).eval()
    sae = sae_manager.get_sae(ckpt).to(device).eval()

    nb_classes = 7
    nb_concepts = args.nb_concepts
    nb_domains = len(args.process_domains)

    accumulated_scores = torch.zeros(
        nb_classes,
        nb_concepts,
        nb_domains,
        device=device,
        dtype=torch.float32,
    )

    activation_counts = torch.zeros(
        nb_classes,
        nb_concepts,
        nb_domains,
        device=device,
        dtype=torch.float32,
    )

    pool = SelectAdaptivePool2d(pool_type="avg", flatten=True)

    for local_d, domain_id in enumerate(args.process_domains):
        domain_name = PACS_DOMAINS[domain_id]
        loader, _ = Load_PACS(domains=[domain_name], batch_size=args.eval_batch_size)

        for x, y in tqdm(loader, desc=f"Direct R_acts [{domain_name}]"):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z_raw = extract_features(backbone, x)
                z_norm = sae.normalizer(z_raw)

                if z_norm.dim() != 4:
                    raise NotImplementedError(
                        "Direct R_acts computation is written for ResNet CxHxW features."
                    )

                n, c_dim, h, w = z_norm.shape
                z_flat = rearrange(z_norm, "n c h w -> (n h w) c")

                _, z_sae = sae.encode(z_flat)

                # Unmasked SAE reconstruction baseline.
                z_recon_flat = sae.decode(z_sae)
                z_recon = rearrange(
                    z_recon_flat,
                    "(n h w) c -> n c h w",
                    n=n,
                    h=h,
                    w=w,
                )
                z_recon = sae.normalizer.denormalize(z_recon)

                logits_unmasked = backbone.classifier(pool(z_recon))
                probs_unmasked = torch.softmax(logits_unmasked, dim=1)
                p_true_unmasked = probs_unmasked[torch.arange(n, device=device), y]

                z_sae_img = rearrange(
                    z_sae,
                    "(n h w) c -> n (h w) c",
                    n=n,
                    h=h,
                    w=w,
                )

                concept_max_per_img, _ = z_sae_img.max(dim=1)

                active_concepts_batch = (
                    concept_max_per_img.max(dim=0)[0] > 0
                ).nonzero(as_tuple=False).squeeze(1)

                for concept_idx in active_concepts_batch:
                    active_img_mask = concept_max_per_img[:, concept_idx] > 0

                    if not active_img_mask.any():
                        continue

                    original_col = z_sae[:, concept_idx].clone()
                    z_sae[:, concept_idx] = 0.0

                    z_recon_flat_m = sae.decode(z_sae)
                    z_recon_m = rearrange(
                        z_recon_flat_m,
                        "(n h w) c -> n c h w",
                        n=n,
                        h=h,
                        w=w,
                    )
                    z_recon_m = sae.normalizer.denormalize(z_recon_m)

                    logits_masked = backbone.classifier(pool(z_recon_m))
                    probs_masked = torch.softmax(logits_masked, dim=1)
                    p_true_masked = probs_masked[torch.arange(n, device=device), y]

                    z_sae[:, concept_idx] = original_col

                    score_drop = p_true_unmasked - p_true_masked

                    active_classes = y[active_img_mask]
                    active_drops = score_drop[active_img_mask]

                    accumulated_scores[:, concept_idx, local_d].scatter_add_(
                        0,
                        active_classes,
                        active_drops,
                    )

                    activation_counts[:, concept_idx, local_d].scatter_add_(
                        0,
                        active_classes,
                        torch.ones_like(active_drops),
                    )

    R_acts = accumulated_scores / activation_counts.clamp(min=1)

    write_score(score_json, "R_acts", R_acts.detach().cpu().numpy())
    write_score(score_json, "R_counts", activation_counts.detach().cpu().numpy())

def run_processors_for_ckpt(args, sae_manager: SparseAEs, ckpt: int, score_json: Path) -> None:
    """
    Runs H and D using existing processors.

    IMPORTANT:
    We do NOT call clean_lib.processors.R.R.process(), because the existing
    R processor tries to dump the old signed/legacy R before dumping R_acts,
    and this can crash during JSON serialization.

    Instead, we compute the domain-wise effects R_acts directly here, then
    compute the paper-ready R_mag and sign consistency S.
    """
    print(f"\n=== Running processors for checkpoint {ckpt} ===")
    base = Processor(
        sae_manager=sae_manager,
        ckpt=ckpt,
        process_domains=args.process_domains,
        file_path=str(score_json),
        dataset="PACS",
    )

    H.from_processor(base).process()
    D.from_processor(base).process()

    compute_and_write_domain_effects_for_r(
        args=args,
        sae_manager=sae_manager,
        ckpt=ckpt,
        score_json=score_json,
    )

    if HAS_M and args.run_m:
        M.from_processor(base).process()

    compute_fixed_r_scores(score_json, eps=args.eps)

def compute_fixed_r_scores(score_json: Path, eps: float = 1e-12) -> None:
    """
    Computes:
      R_mag(k,c): entropy over |D_d(k,c)| across domains.
      R_sign_consistency(k,c): |sum_d D_d(k,c)| / (sum_d |D_d(k,c)| + eps).

    The sign-consistency score is in [0,1] by the triangle inequality.
    It is later used as a gate in the typology.
    """
    print(f"Computing R_mag and sign consistency from {score_json}")

    R_acts = read_score(score_json, "R_acts")  # shape: classes x concepts x domains

    mag = np.abs(R_acts)
    denom = mag.sum(axis=2, keepdims=True)
    probs = mag / (denom + eps)

    log_m = math.log(R_acts.shape[2])
    R_mag = -(probs * np.log(probs + eps)).sum(axis=2) / log_m
    R_mag[denom.squeeze(-1) <= eps] = 0.0

    sign_consistency = np.abs(R_acts.sum(axis=2)) / (mag.sum(axis=2) + eps)
    sign_consistency[denom.squeeze(-1) <= eps] = 0.0

    # Numerical clipping protects against tiny floating point overshoots.
    sign_consistency = np.clip(sign_consistency, 0.0, 1.0)
    R_mag = np.clip(R_mag, 0.0, 1.0)

    write_score(score_json, "R_mag", R_mag.astype(np.float32))
    write_score(score_json, "R_sign_consistency", sign_consistency.astype(np.float32))


# ---------------------------------------------------------------------
# Classification utilities
# ---------------------------------------------------------------------

def classify_features(backbone, feature_map):
    """
    Works for ResNet-style DomainBed models.
    Falls back to classifier(feature_map) for non-4D features.
    """
    pool = SelectAdaptivePool2d(pool_type="avg", flatten=True)

    if hasattr(backbone, "featurizer") and backbone.featurizer.__class__.__name__ == "ResNet":
        return backbone.classifier(pool(feature_map))

    if feature_map.dim() == 4:
        return backbone.classifier(pool(feature_map))

    return backbone.classifier(feature_map)


def sae_reconstruct(backbone, sae, x, rearrange_string: str):
    """
    Returns raw features, SAE codes, reconstructed features, and spatial shape.
    """
    z_raw = extract_features(backbone, x)
    z_norm = sae.normalizer(z_raw)

    if z_norm.dim() == 4:
        n, c, h, w = z_norm.shape
        z_flat = rearrange(z_norm, "n c h w -> (n h w) c")
        _, z_sae = sae.encode(z_flat)
        z_recon_flat = sae.decode(z_sae)
        z_recon = rearrange(z_recon_flat, "(n h w) c -> n c h w", n=n, h=h, w=w)
        z_recon = sae.normalizer.denormalize(z_recon)
        return z_raw, z_sae, z_recon, (n, h, w)

    n, t, c = z_norm.shape
    z_flat = rearrange(z_norm, "n t c -> (n t) c")
    _, z_sae = sae.encode(z_flat)
    z_recon_flat = sae.decode(z_sae)
    z_recon = rearrange(z_recon_flat, "(n t) c -> n t c", n=n, t=t)
    z_recon = sae.normalizer.denormalize(z_recon)
    return z_raw, z_sae, z_recon, (n, t)


# ---------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------

def class_macro(correct: np.ndarray, total: np.ndarray) -> float:
    valid = total > 0
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(correct[valid] / np.maximum(total[valid], 1)))


def overall_acc(correct: np.ndarray, total: np.ndarray) -> float:
    return float(correct.sum() / max(total.sum(), 1))


def add_per_class_cols(row: dict, correct: np.ndarray, total: np.ndarray, prefix: str = "") -> dict:
    for cls, name in enumerate(CLASS_NAMES):
        col = f"{prefix}{name}" if prefix else name
        row[col] = float(correct[cls] / max(total[cls], 1))
    return row


# ---------------------------------------------------------------------
# Reconstruction fidelity
# ---------------------------------------------------------------------

def evaluate_reconstruction_fidelity(args, sae_manager: SparseAEs, ckpt: int) -> pd.DataFrame:
    backbone = sae_manager.get_backbone(ckpt).to(device).eval()
    sae = sae_manager.get_sae(ckpt).to(device).eval()

    rows = []

    for domain_id in args.process_domains:
        domain_name = PACS_DOMAINS[domain_id]
        loader, _ = Load_PACS(domains=[domain_name], batch_size=args.eval_batch_size)

        correct_orig = np.zeros(7)
        correct_recon = np.zeros(7)
        total = np.zeros(7)

        for x, y in tqdm(loader, desc=f"Recon fidelity ckpt={ckpt} domain={domain_name}"):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z_raw, _, z_recon, _ = sae_reconstruct(
                    backbone,
                    sae,
                    x,
                    args.rearrange_string,
                )

                logits_orig = classify_features(backbone, z_raw)
                logits_recon = classify_features(backbone, z_recon)

                pred_orig = logits_orig.argmax(dim=1)
                pred_recon = logits_recon.argmax(dim=1)

            for cls in range(7):
                mask = y == cls
                total[cls] += mask.sum().item()
                correct_orig[cls] += ((pred_orig == y) & mask).sum().item()
                correct_recon[cls] += ((pred_recon == y) & mask).sum().item()

        row = {
            "ckpt": ckpt,
            "domain": domain_name,
            "original_accuracy": overall_acc(correct_orig, total),
            "sae_reconstructed_accuracy": overall_acc(correct_recon, total),
            "drop": overall_acc(correct_orig, total) - overall_acc(correct_recon, total),
            "original_macro_accuracy": class_macro(correct_orig, total),
            "sae_reconstructed_macro_accuracy": class_macro(correct_recon, total),
            "macro_drop": class_macro(correct_orig, total) - class_macro(correct_recon, total),
            "n_examples": int(total.sum()),
        }

        row = add_per_class_cols(row, correct_orig, total, prefix="orig_")
        row = add_per_class_cols(row, correct_recon, total, prefix="recon_")
        rows.append(row)

    return pd.DataFrame(rows)


def make_baseline_intervention_rows(recon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds original and SAE-reconstruction rows to the intervention table so that
    intervention results are directly comparable to their baselines.
    """
    rows = []

    for _, r in recon_df.iterrows():
        row_orig = {
            "ckpt": int(r["ckpt"]),
            "domain": r["domain"],
            "intervention": "original_features",
            "uses_true_label": False,
            "overall_accuracy": float(r["original_accuracy"]),
            "macro_avg": float(r["original_macro_accuracy"]),
            "n_examples": int(r["n_examples"]),
        }
        for name in CLASS_NAMES:
            row_orig[name] = float(r[f"orig_{name}"])
        rows.append(row_orig)

        row_recon = {
            "ckpt": int(r["ckpt"]),
            "domain": r["domain"],
            "intervention": "sae_reconstruction",
            "uses_true_label": False,
            "overall_accuracy": float(r["sae_reconstructed_accuracy"]),
            "macro_avg": float(r["sae_reconstructed_macro_accuracy"]),
            "n_examples": int(r["n_examples"]),
        }
        for name in CLASS_NAMES:
            row_recon[name] = float(r[f"recon_{name}"])
        rows.append(row_recon)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# D thresholding, consistency gating, and typology
# ---------------------------------------------------------------------

def label_d_scores(
    D_scores: np.ndarray,
    D_counts: np.ndarray,
    abs_tau: float,
    min_count: int,
) -> np.ndarray:
    """
    Sign-respecting D labels.

    Returns:
        +1 helpful/support
         0 neutral or invalid
        -1 harmful

    A concept is valid for class k only if:
        D_counts[k,c] >= min_count
        D_scores[k,c] is finite

    Then:
        D > +abs_tau => +1
        D < -abs_tau => -1
        otherwise    => 0
    """
    labels = np.zeros_like(D_scores, dtype=np.int8)
    valid = (D_counts >= min_count) & np.isfinite(D_scores)

    labels[valid & (D_scores > abs_tau)] = 1
    labels[valid & (D_scores < -abs_tau)] = -1

    return labels


def compute_consistency_gate(
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    tau_r: float,
    tau_s: float,
) -> np.ndarray:
    """
    High discriminative consistency iff:
        R_mag(k,c) >= tau_R and S(k,c) >= tau_S.

    This keeps the typology at 8 categories while preventing mixed-sign
    concepts from being classified as robust merely because their effect
    magnitudes are spread across domains.
    """
    return (R_mag_scores >= tau_r) & (S_scores >= tau_s)


def build_typology(
    H_scores: np.ndarray,
    D_scores: np.ndarray,
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    D_counts: np.ndarray,
    tau_h: float,
    tau_r: float,
    tau_s: float,
    d_abs_tau: float,
    min_count: int,
) -> Dict[str, object]:
    d_label = label_d_scores(
        D_scores=D_scores,
        D_counts=D_counts,
        abs_tau=d_abs_tau,
        min_count=min_count,
    )

    valid = (D_counts >= min_count) & np.isfinite(D_scores)
    salient = d_label != 0

    H_bin = H_scores >= tau_h
    consistency_bin = compute_consistency_gate(
        R_mag_scores=R_mag_scores,
        S_scores=S_scores,
        tau_r=tau_r,
        tau_s=tau_s,
    )

    nb_classes, nb_concepts = D_scores.shape
    concept_rows = []
    count_rows = []
    class_profile_rows = []

    # Conflict only among sign-respecting salient labels.
    conflict = ((d_label == 1).any(axis=0) & (d_label == -1).any(axis=0))

    for k in range(nb_classes):
        bucket_counts = {name: 0 for name in BUCKET_NAMES.values()}
        bucket_mass = {f"{name}_mass": 0.0 for name in BUCKET_NAMES.values()}

        abs_d = np.abs(D_scores[k])
        valid_k = valid[k]
        salient_k = salient[k]

        # Denominator: total valid |D| mass, not invalid low-count concepts.
        denom = abs_d[valid_k].sum() + 1e-12

        for c in range(nb_concepts):
            if not salient_k[c]:
                continue

            h = int(H_bin[k, c])
            cons = int(consistency_bin[k, c])
            d_bit = int(d_label[k, c] == 1)

            key = f"{h}{cons}{d_bit}"
            name = BUCKET_NAMES[key]

            bucket_counts[name] += 1
            bucket_mass[f"{name}_mass"] += float(abs_d[c])

            concept_rows.append({
                "class_id": k,
                "class_name": CLASS_NAMES[k],
                "concept": c,
                "H": float(H_scores[k, c]),
                "D": float(D_scores[k, c]),
                "R_mag": float(R_mag_scores[k, c]),
                "S": float(S_scores[k, c]),
                "consistent_effect": bool(consistency_bin[k, c]),
                # Backward-compatible alias; in the final paper, call this R_mag.
                "R": float(R_mag_scores[k, c]),
                "D_count": int(D_counts[k, c]),
                "D_label": int(d_label[k, c]),
                "bucket_key": key,
                "bucket": name,
                "absD": float(abs_d[c]),
                "conflicting_concept": bool(conflict[c]),
            })

        robust_support_mass = bucket_mass["robust_support_mass"] / denom
        harmful_invariant_mass = bucket_mass["harmful_invariant_mass"] / denom
        domain_contingent_mass = (
            bucket_mass["domain_contingent_support_mass"]
            + bucket_mass["domain_contingent_harm_mass"]
        ) / denom
        conflict_mass = abs_d[valid_k & conflict].sum() / denom

        count_row = {
            "class_id": k,
            "class_name": CLASS_NAMES[k],
            "total_concepts": nb_concepts,
            "valid_min_count": int(valid_k.sum()),
            "invalid_low_count_or_nan": int((~valid_k).sum()),
            "neutral_valid": int((valid_k & (d_label[k] == 0)).sum()),
            "salient_positive": int((d_label[k] == 1).sum()),
            "salient_negative": int((d_label[k] == -1).sum()),
            "salient_consistent": int((salient_k & consistency_bin[k]).sum()),
            "salient_inconsistent": int((salient_k & (~consistency_bin[k])).sum()),
            **bucket_counts,
        }
        count_rows.append(count_row)

        profile_row = {
            "class_id": k,
            "class_name": CLASS_NAMES[k],
            "valid_min_count": int(valid_k.sum()),
            "salient_count": int(salient_k.sum()),
            "salient_consistent": int((salient_k & consistency_bin[k]).sum()),
            "salient_inconsistent": int((salient_k & (~consistency_bin[k])).sum()),
            "RSM": robust_support_mass,
            "HIM": harmful_invariant_mass,
            "DCM": domain_contingent_mass,
            "ConflictMass": conflict_mass,
            **{key: val / denom for key, val in bucket_mass.items()},
        }
        class_profile_rows.append(profile_row)

    concept_df = pd.DataFrame(concept_rows)
    counts_df = pd.DataFrame(count_rows)
    class_profiles_df = pd.DataFrame(class_profile_rows)

    model_profile = {
        "RSM": float(class_profiles_df["RSM"].mean()),
        "HIM": float(class_profiles_df["HIM"].mean()),
        "DCM": float(class_profiles_df["DCM"].mean()),
        "ConflictMass": float(class_profiles_df["ConflictMass"].mean()),
        "mean_valid_min_count": float(class_profiles_df["valid_min_count"].mean()),
        "mean_salient_count": float(class_profiles_df["salient_count"].mean()),
        "mean_salient_consistent": float(class_profiles_df["salient_consistent"].mean()),
        "mean_salient_inconsistent": float(class_profiles_df["salient_inconsistent"].mean()),
    }

    d_sanity = sanity_check_labels(D_scores, D_counts, d_label, d_abs_tau, min_count)
    consistency_sanity = sanity_check_consistency(
        R_mag_scores=R_mag_scores,
        S_scores=S_scores,
        consistency_bin=consistency_bin,
        tau_r=tau_r,
        tau_s=tau_s,
        valid=valid,
    )

    return {
        "concepts": concept_df,
        "counts": counts_df,
        "class_profiles": class_profiles_df,
        "model_profile": model_profile,
        "d_label": d_label,
        "valid": valid,
        "conflict": conflict,
        "consistency_bin": consistency_bin,
        "d_sanity": d_sanity,
        "consistency_sanity": consistency_sanity,
    }


def sanity_check_labels(
    D_scores: np.ndarray,
    D_counts: np.ndarray,
    D_label: np.ndarray,
    d_abs_tau: float,
    min_count: int,
) -> pd.DataFrame:
    """
    Produces sanity checks for D labeling.

    In a valid run:
    - positive labels should never have D <= tau_D
    - negative labels should never have D >= -tau_D
    - low-count concepts should never be non-neutral
    """
    rows = []

    for k, name in enumerate(CLASS_NAMES):
        pos = D_label[k] == 1
        neg = D_label[k] == -1
        low_count = D_counts[k] < min_count

        rows.append({
            "class_id": k,
            "class_name": name,
            "n_positive_labels": int(pos.sum()),
            "n_negative_labels": int(neg.sum()),
            "n_neutral_labels": int((D_label[k] == 0).sum()),
            "positive_label_with_nonpositive_D": int((pos & (D_scores[k] <= 0)).sum()),
            "negative_label_with_nonnegative_D": int((neg & (D_scores[k] >= 0)).sum()),
            "positive_label_below_tau": int((pos & (D_scores[k] <= d_abs_tau)).sum()),
            "negative_label_above_minus_tau": int((neg & (D_scores[k] >= -d_abs_tau)).sum()),
            "low_count_labeled_non_neutral": int((low_count & (D_label[k] != 0)).sum()),
            "min_positive_D": float(np.min(D_scores[k][pos])) if pos.any() else np.nan,
            "max_negative_D": float(np.max(D_scores[k][neg])) if neg.any() else np.nan,
        })

    return pd.DataFrame(rows)


def sanity_check_consistency(
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    consistency_bin: np.ndarray,
    tau_r: float,
    tau_s: float,
    valid: np.ndarray,
) -> pd.DataFrame:
    """
    Checks that the consistency gate is doing exactly what it should.

    In a valid run:
    - no concept marked consistent should have R_mag < tau_R
    - no concept marked consistent should have S < tau_S
    - no S value should be outside [0,1]
    - no R_mag value should be outside [0,1]
    """
    rows = []

    for k, name in enumerate(CLASS_NAMES):
        consistent = consistency_bin[k]
        valid_k = valid[k]

        rows.append({
            "class_id": k,
            "class_name": name,
            "valid_count": int(valid_k.sum()),
            "consistent_valid_count": int((valid_k & consistent).sum()),
            "inconsistent_valid_count": int((valid_k & (~consistent)).sum()),
            "consistent_with_R_below_tau": int((valid_k & consistent & (R_mag_scores[k] < tau_r)).sum()),
            "consistent_with_S_below_tau": int((valid_k & consistent & (S_scores[k] < tau_s)).sum()),
            "R_mag_below_0_or_above_1": int((valid_k & ((R_mag_scores[k] < -1e-6) | (R_mag_scores[k] > 1 + 1e-6))).sum()),
            "S_below_0_or_above_1": int((valid_k & ((S_scores[k] < -1e-6) | (S_scores[k] > 1 + 1e-6))).sum()),
            "mean_R_mag_valid": float(np.mean(R_mag_scores[k][valid_k])) if valid_k.any() else np.nan,
            "mean_S_valid": float(np.mean(S_scores[k][valid_k])) if valid_k.any() else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def save_score_plots(
    out_fig_dir: Path,
    ckpt: int,
    H_scores: np.ndarray,
    D_scores: np.ndarray,
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    D_counts: np.ndarray,
    D_label: np.ndarray,
    consistency_bin: np.ndarray,
    min_count: int,
):
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    valid = (D_counts >= min_count) & np.isfinite(D_scores)
    h = H_scores[valid]
    d = D_scores[valid]
    r = R_mag_scores[valid]
    s = S_scores[valid]

    # H vs D
    plt.figure(figsize=(6, 4.5))
    plt.scatter(h, d, s=3, alpha=0.25)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Activation invariance H")
    plt.ylabel("Discriminative effect D")
    plt.title(f"H vs D, checkpoint {ckpt}")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"h_vs_d_ckpt{ckpt}.png", dpi=300)
    plt.close()

    # R_mag vs D
    plt.figure(figsize=(6, 4.5))
    plt.scatter(r, d, s=3, alpha=0.25)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Magnitude consistency R_mag")
    plt.ylabel("Discriminative effect D")
    plt.title(f"R_mag vs D, checkpoint {ckpt}")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"rmag_vs_d_ckpt{ckpt}.png", dpi=300)
    plt.close()

    # S vs D
    plt.figure(figsize=(6, 4.5))
    plt.scatter(s, d, s=3, alpha=0.25)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Sign consistency S")
    plt.ylabel("Discriminative effect D")
    plt.title(f"S vs D, checkpoint {ckpt}")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"s_vs_d_ckpt{ckpt}.png", dpi=300)
    plt.close()

    # R_mag vs S
    plt.figure(figsize=(6, 4.5))
    plt.scatter(r, s, s=3, alpha=0.25)
    plt.xlabel("Magnitude consistency R_mag")
    plt.ylabel("Sign consistency S")
    plt.title(f"R_mag vs S, checkpoint {ckpt}")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"rmag_vs_s_ckpt{ckpt}.png", dpi=300)
    plt.close()

    # H vs consistency, colored by D label
    plt.figure(figsize=(6, 4.5))
    valid_flat = valid.reshape(-1)
    H_flat = H_scores.reshape(-1)[valid_flat]
    consistency_flat = consistency_bin.reshape(-1)[valid_flat].astype(float)
    label_flat = D_label.reshape(-1)[valid_flat]

    for lab, label_name in [(-1, "harmful"), (0, "neutral"), (1, "support")]:
        mask = label_flat == lab
        if mask.sum() == 0:
            continue
        plt.scatter(H_flat[mask], consistency_flat[mask], s=3, alpha=0.25, label=label_name)

    plt.xlabel("Activation invariance H")
    plt.ylabel("Consistency gate")
    plt.title(f"H vs gated consistency by D label, checkpoint {ckpt}")
    plt.yticks([0, 1], ["low", "high"])
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"h_vs_gated_consistency_by_d_label_ckpt{ckpt}.png", dpi=300)
    plt.close()

    # Distributions
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    axes[0].hist(h, bins=50)
    axes[0].set_title("H")
    axes[1].hist(d, bins=50)
    axes[1].set_title("D")
    axes[2].hist(r, bins=50)
    axes[2].set_title("R_mag")
    axes[3].hist(s, bins=50)
    axes[3].set_title("S")
    for ax in axes:
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"h_d_rmag_s_distributions_ckpt{ckpt}.png", dpi=300)
    plt.close()


def save_bucket_barplot(out_fig_dir: Path, ckpt: int, counts_df: pd.DataFrame):
    bucket_cols = list(BUCKET_NAMES.values())
    plot_df = counts_df.set_index("class_name")[bucket_cols]

    plt.figure(figsize=(12, 5))
    plot_df.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.ylabel("Concept count")
    plt.title(f"Concept typology counts by class, checkpoint {ckpt}")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"typology_counts_ckpt{ckpt}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------

def make_global_negative_mask(
    D_scores: np.ndarray,
    D_counts: np.ndarray,
    d_abs_tau: float,
    min_count: int,
) -> np.ndarray:
    """
    Safer label-free global negative mask.

    This is only a sanity/control intervention, not a proposed method.
    """
    valid = (D_counts >= min_count) & np.isfinite(D_scores)

    masked_D = np.where(valid, D_scores, np.nan)

    with np.errstate(all="ignore"):
        mean_D = np.nanmean(masked_D, axis=0)
        max_D = np.nanmax(masked_D, axis=0)

    any_valid = valid.any(axis=0)

    global_mask = any_valid & (mean_D < -d_abs_tau) & (max_D <= d_abs_tau)
    global_mask = np.nan_to_num(global_mask, nan=False).astype(bool)

    return global_mask


def make_mask_for_rule(
    rule_name: str,
    y_batch: torch.Tensor,
    D_label: np.ndarray,
    H_scores: np.ndarray,
    consistency_bin: np.ndarray,
    global_negative_mask: np.ndarray,
) -> List[np.ndarray]:
    """
    Returns list of boolean masks over concepts, one per image.
    True means mask this concept.
    """
    masks = []
    y_cpu = y_batch.detach().cpu().numpy()

    for y in y_cpu:
        if rule_name == "remove_negative_D":
            mask = D_label[y] == -1

        elif rule_name == "remove_harmful_invariant":
            mask = (
                (D_label[y] == -1)
                & (H_scores[y] >= 0)  # harmless no-op; H gate is included below for readability
                & consistency_bin[y]
            )
            # Apply H explicitly:
            # high-H, high-consistency, negative-D
            # H threshold already used to form buckets, but we pass H separately in evaluate_intervention.
            # The actual thresholded H mask is supplied as consistency independent, so this is set below.
            raise RuntimeError(
                "Internal error: remove_harmful_invariant requires H_bin; use make_mask_for_rule_with_hbin."
            )

        elif rule_name == "keep_robust_support_only":
            raise RuntimeError(
                "Internal error: keep_robust_support_only requires H_bin; use make_mask_for_rule_with_hbin."
            )

        elif rule_name == "global_no_label_negative_mask":
            mask = global_negative_mask

        else:
            raise ValueError(f"Unknown intervention rule: {rule_name}")

        masks.append(mask)

    return masks


def make_mask_for_rule_with_hbin(
    rule_name: str,
    y_batch: torch.Tensor,
    D_label: np.ndarray,
    H_bin: np.ndarray,
    consistency_bin: np.ndarray,
    global_negative_mask: np.ndarray,
) -> List[np.ndarray]:
    """
    Returns list of boolean masks over concepts, one per image.
    True means mask this concept.

    Uses the gated consistency condition:
        consistency_bin = (R_mag >= tau_R) & (S >= tau_S)
    """
    masks = []
    y_cpu = y_batch.detach().cpu().numpy()

    for y in y_cpu:
        if rule_name == "remove_negative_D":
            mask = D_label[y] == -1

        elif rule_name == "remove_harmful_invariant":
            mask = (
                (D_label[y] == -1)
                & H_bin[y]
                & consistency_bin[y]
            )

        elif rule_name == "keep_robust_support_only":
            keep = (
                (D_label[y] == 1)
                & H_bin[y]
                & consistency_bin[y]
            )
            mask = ~keep

        elif rule_name == "global_no_label_negative_mask":
            mask = global_negative_mask

        else:
            raise ValueError(f"Unknown intervention rule: {rule_name}")

        masks.append(mask)

    return masks


def evaluate_intervention(
    args,
    sae_manager: SparseAEs,
    ckpt: int,
    rule_name: str,
    D_label: np.ndarray,
    H_bin: np.ndarray,
    consistency_bin: np.ndarray,
    global_negative_mask: np.ndarray,
) -> pd.DataFrame:
    backbone = sae_manager.get_backbone(ckpt).to(device).eval()
    sae = sae_manager.get_sae(ckpt).to(device).eval()

    rows = []

    uses_true_label = rule_name in {
        "remove_negative_D",
        "remove_harmful_invariant",
        "keep_robust_support_only",
    }

    for domain_id in args.process_domains:
        domain_name = PACS_DOMAINS[domain_id]
        loader, _ = Load_PACS(domains=[domain_name], batch_size=args.eval_batch_size)

        correct = np.zeros(7)
        total = np.zeros(7)

        for x, y in tqdm(loader, desc=f"Intervention {rule_name}, ckpt={ckpt}, domain={domain_name}"):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z_raw = extract_features(backbone, x)
                z_norm = sae.normalizer(z_raw)

                if z_norm.dim() != 4:
                    raise NotImplementedError(
                        "Intervention masking below is written for ResNet CxHxW features."
                    )

                n, c_dim, h, w = z_norm.shape
                z_flat = rearrange(z_norm, "n c h w -> (n h w) c")
                _, z_sae = sae.encode(z_flat)

                z_sae_img = rearrange(z_sae, "(n h w) c -> n (h w) c", n=n, h=h, w=w).clone()

                masks = make_mask_for_rule_with_hbin(
                    rule_name=rule_name,
                    y_batch=y,
                    D_label=D_label,
                    H_bin=H_bin,
                    consistency_bin=consistency_bin,
                    global_negative_mask=global_negative_mask,
                )

                for i, mask_np in enumerate(masks):
                    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
                    z_sae_img[i, :, mask_t] = 0.0

                z_sae_masked = rearrange(z_sae_img, "n t c -> (n t) c")
                z_recon_flat = sae.decode(z_sae_masked)
                z_recon = rearrange(z_recon_flat, "(n h w) c -> n c h w", n=n, h=h, w=w)
                z_recon = sae.normalizer.denormalize(z_recon)

                logits = classify_features(backbone, z_recon)
                pred = logits.argmax(dim=1)

            for cls in range(7):
                cls_mask = y == cls
                total[cls] += cls_mask.sum().item()
                correct[cls] += ((pred == y) & cls_mask).sum().item()

        row = {
            "ckpt": ckpt,
            "domain": domain_name,
            "intervention": rule_name,
            "uses_true_label": uses_true_label,
            "overall_accuracy": overall_acc(correct, total),
            "macro_avg": class_macro(correct, total),
            "n_examples": int(total.sum()),
        }

        row = add_per_class_cols(row, correct, total)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Representative concepts
# ---------------------------------------------------------------------

def select_representative_concepts(concept_df: pd.DataFrame, top_n_per_bucket: int = 1) -> pd.DataFrame:
    """
    Selects representative concepts for visualization.

    Selection score:
        H * R_mag * S * |D| * log(1 + D_count)

    This prioritizes concepts that are activation-invariant, magnitude-consistent,
    sign-consistent, discriminatively strong, and observed enough times.
    """
    if concept_df.empty:
        return pd.DataFrame()

    df = concept_df.copy()
    df["selection_score"] = (
        df["H"].astype(float)
        * df["R_mag"].astype(float)
        * df["S"].astype(float)
        * df["absD"].astype(float)
        * np.log1p(df["D_count"].astype(float))
    )

    rows = []
    for bucket in BUCKET_NAMES.values():
        sub = df[df["bucket"] == bucket].sort_values("selection_score", ascending=False)
        if len(sub) > 0:
            rows.append(sub.head(top_n_per_bucket))

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------

def run_sensitivity(
    H_scores: np.ndarray,
    D_scores: np.ndarray,
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    D_counts: np.ndarray,
    tau_h_values: List[float],
    tau_r_values: List[float],
    tau_s_values: List[float],
    d_abs_tau_values: List[float],
    min_count_values: List[int],
) -> pd.DataFrame:
    rows = []

    for tau_h in tau_h_values:
        for tau_r in tau_r_values:
            for tau_s in tau_s_values:
                for d_tau in d_abs_tau_values:
                    for min_count in min_count_values:
                        typ = build_typology(
                            H_scores=H_scores,
                            D_scores=D_scores,
                            R_mag_scores=R_mag_scores,
                            S_scores=S_scores,
                            D_counts=D_counts,
                            tau_h=tau_h,
                            tau_r=tau_r,
                            tau_s=tau_s,
                            d_abs_tau=d_tau,
                            min_count=min_count,
                        )
                        profile = typ["model_profile"]
                        rows.append({
                            "tau_H": tau_h,
                            "tau_R": tau_r,
                            "tau_S": tau_s,
                            "D_abs_tau": d_tau,
                            "min_D_count": min_count,
                            **profile,
                        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_d_distribution(
    D_scores: np.ndarray,
    D_counts: np.ndarray,
    D_label: np.ndarray,
    min_count: int,
) -> pd.DataFrame:
    rows = []

    for k, name in enumerate(CLASS_NAMES):
        valid = (D_counts[k] >= min_count) & np.isfinite(D_scores[k])
        vals = D_scores[k][valid]

        row = {
            "class_id": k,
            "class_name": name,
            "valid_count": int(valid.sum()),
            "positive_count": int((D_label[k] == 1).sum()),
            "negative_count": int((D_label[k] == -1).sum()),
            "neutral_count": int((valid & (D_label[k] == 0)).sum()),
            "D_min": float(np.min(vals)) if vals.size else np.nan,
            "D_q01": float(np.quantile(vals, 0.01)) if vals.size else np.nan,
            "D_q05": float(np.quantile(vals, 0.05)) if vals.size else np.nan,
            "D_median": float(np.median(vals)) if vals.size else np.nan,
            "D_q95": float(np.quantile(vals, 0.95)) if vals.size else np.nan,
            "D_q99": float(np.quantile(vals, 0.99)) if vals.size else np.nan,
            "D_max": float(np.max(vals)) if vals.size else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_consistency_distribution(
    R_mag_scores: np.ndarray,
    S_scores: np.ndarray,
    consistency_bin: np.ndarray,
    D_counts: np.ndarray,
    D_label: np.ndarray,
    min_count: int,
) -> pd.DataFrame:
    rows = []

    for k, name in enumerate(CLASS_NAMES):
        valid = (D_counts[k] >= min_count)
        salient = valid & (D_label[k] != 0)

        r_vals = R_mag_scores[k][salient]
        s_vals = S_scores[k][salient]

        rows.append({
            "class_id": k,
            "class_name": name,
            "salient_count": int(salient.sum()),
            "salient_consistent_count": int((salient & consistency_bin[k]).sum()),
            "salient_inconsistent_count": int((salient & (~consistency_bin[k])).sum()),
            "R_mag_mean_salient": float(np.mean(r_vals)) if r_vals.size else np.nan,
            "R_mag_median_salient": float(np.median(r_vals)) if r_vals.size else np.nan,
            "S_mean_salient": float(np.mean(s_vals)) if s_vals.size else np.nan,
            "S_median_salient": float(np.median(s_vals)) if s_vals.size else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone_dir", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./paper_results")

    parser.add_argument("--ckpts", type=int, nargs="+", required=True)
    parser.add_argument("--process_domains", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--train_envs", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--sae_dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--nb_concepts", type=int, default=2048 * 8)
    parser.add_argument("--spatial_w", type=int, default=7)
    parser.add_argument("--rearrange_string", type=str, default="n c w h -> (n w h) c")

    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--tau_h", type=float, default=0.8)
    parser.add_argument("--tau_r", type=float, default=0.8)
    parser.add_argument("--tau_s", type=float, default=0.75)

    parser.add_argument("--d_abs_tau", type=float, default=0.001)
    parser.add_argument("--min_d_count", type=int, default=5)

    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument("--skip_processors", action="store_true")
    parser.add_argument("--run_m", action="store_true")

    args = parser.parse_args()

    if args.d_abs_tau <= 0:
        raise ValueError("--d_abs_tau must be positive.")
    if args.min_d_count < 1:
        raise ValueError("--min_d_count must be at least 1.")
    if not (0.0 <= args.tau_h <= 1.0):
        raise ValueError("--tau_h must be in [0, 1].")
    if not (0.0 <= args.tau_r <= 1.0):
        raise ValueError("--tau_r must be in [0, 1].")
    if not (0.0 <= args.tau_s <= 1.0):
        raise ValueError("--tau_s must be in [0, 1].")

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    score_dir = out_dir / "scores"

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    sae_manager = build_sae_manager(args, args.ckpts)

    all_recon = []
    all_model_profiles = []
    all_interventions = []
    all_sensitivity = []
    all_representatives = []
    all_d_sanity = []
    all_consistency_sanity = []
    all_d_summaries = []
    all_consistency_summaries = []

    for ckpt in args.ckpts:
        score_json = score_dir / f"scores_ckpt{ckpt}.json"

        if not args.skip_processors:
            run_processors_for_ckpt(args, sae_manager, ckpt, score_json)
        else:
            if not score_json.exists():
                raise FileNotFoundError(f"--skip_processors was used, but {score_json} does not exist.")
            compute_fixed_r_scores(score_json, eps=args.eps)

        print(f"\n=== Loading scores for checkpoint {ckpt} ===")
        H_scores = read_score(score_json, "H", nb_concepts=args.nb_concepts)
        D_scores = read_score(score_json, "D", nb_concepts=args.nb_concepts)
        D_counts = read_score(score_json, "D_counts", nb_concepts=args.nb_concepts)
        R_mag_scores = read_score(score_json, "R_mag", nb_concepts=args.nb_concepts)
        S_scores = read_score(score_json, "R_sign_consistency", nb_concepts=args.nb_concepts)

        # 1. Reconstruction fidelity
        recon_df = evaluate_reconstruction_fidelity(args, sae_manager, ckpt)
        recon_df.to_csv(table_dir / f"reconstruction_fidelity_ckpt{ckpt}.csv", index=False)
        all_recon.append(recon_df)

        # 2. Typology and profiles with:
        #    - sign-respecting D labels
        #    - consistency gate: R_mag >= tau_R and S >= tau_S
        typ = build_typology(
            H_scores=H_scores,
            D_scores=D_scores,
            R_mag_scores=R_mag_scores,
            S_scores=S_scores,
            D_counts=D_counts,
            tau_h=args.tau_h,
            tau_r=args.tau_r,
            tau_s=args.tau_s,
            d_abs_tau=args.d_abs_tau,
            min_count=args.min_d_count,
        )

        concept_df = typ["concepts"]
        counts_df = typ["counts"]
        class_profiles_df = typ["class_profiles"]
        D_label = typ["d_label"]
        consistency_bin = typ["consistency_bin"]
        d_sanity_df = typ["d_sanity"]
        consistency_sanity_df = typ["consistency_sanity"]

        concept_df.to_csv(table_dir / f"concept_typology_long_ckpt{ckpt}.csv", index=False)
        counts_df.to_csv(table_dir / f"typology_counts_ckpt{ckpt}.csv", index=False)
        class_profiles_df.to_csv(table_dir / f"class_profiles_ckpt{ckpt}.csv", index=False)
        d_sanity_df.to_csv(table_dir / f"d_label_sanity_ckpt{ckpt}.csv", index=False)
        consistency_sanity_df.to_csv(table_dir / f"consistency_gate_sanity_ckpt{ckpt}.csv", index=False)

        all_d_sanity.append(d_sanity_df.assign(ckpt=ckpt))
        all_consistency_sanity.append(consistency_sanity_df.assign(ckpt=ckpt))

        d_summary_df = summarize_d_distribution(
            D_scores=D_scores,
            D_counts=D_counts,
            D_label=D_label,
            min_count=args.min_d_count,
        )
        d_summary_df["ckpt"] = ckpt
        d_summary_df.to_csv(table_dir / f"d_distribution_summary_ckpt{ckpt}.csv", index=False)
        all_d_summaries.append(d_summary_df)

        consistency_summary_df = summarize_consistency_distribution(
            R_mag_scores=R_mag_scores,
            S_scores=S_scores,
            consistency_bin=consistency_bin,
            D_counts=D_counts,
            D_label=D_label,
            min_count=args.min_d_count,
        )
        consistency_summary_df["ckpt"] = ckpt
        consistency_summary_df.to_csv(table_dir / f"consistency_distribution_summary_ckpt{ckpt}.csv", index=False)
        all_consistency_summaries.append(consistency_summary_df)

        # 3. Plots
        save_score_plots(
            fig_dir,
            ckpt,
            H_scores,
            D_scores,
            R_mag_scores,
            S_scores,
            D_counts,
            D_label,
            consistency_bin,
            args.min_d_count,
        )
        save_bucket_barplot(fig_dir, ckpt, counts_df)

        # 4. Single-checkpoint profile
        model_profile = {
            "ckpt": ckpt,
            "tau_H": args.tau_h,
            "tau_R": args.tau_r,
            "tau_S": args.tau_s,
            "D_abs_tau": args.d_abs_tau,
            "min_D_count": args.min_d_count,
            **typ["model_profile"],
        }

        model_profile["mean_original_accuracy"] = recon_df["original_accuracy"].mean()
        model_profile["mean_sae_reconstructed_accuracy"] = recon_df["sae_reconstructed_accuracy"].mean()
        model_profile["mean_reconstruction_drop"] = recon_df["drop"].mean()
        model_profile["mean_original_macro_accuracy"] = recon_df["original_macro_accuracy"].mean()
        model_profile["mean_sae_reconstructed_macro_accuracy"] = recon_df["sae_reconstructed_macro_accuracy"].mean()
        model_profile["mean_macro_drop"] = recon_df["macro_drop"].mean()

        all_model_profiles.append(model_profile)

        # 5. Interventions
        H_bin = H_scores >= args.tau_h

        global_negative_mask = make_global_negative_mask(
            D_scores=D_scores,
            D_counts=D_counts,
            d_abs_tau=args.d_abs_tau,
            min_count=args.min_d_count,
        )

        baseline_df = make_baseline_intervention_rows(recon_df)

        intervention_rules = [
            "remove_negative_D",
            "remove_harmful_invariant",
            "keep_robust_support_only",
            "global_no_label_negative_mask",
        ]

        ckpt_interventions = [baseline_df]

        for rule in intervention_rules:
            inter_df = evaluate_intervention(
                args=args,
                sae_manager=sae_manager,
                ckpt=ckpt,
                rule_name=rule,
                D_label=D_label,
                H_bin=H_bin,
                consistency_bin=consistency_bin,
                global_negative_mask=global_negative_mask,
            )
            ckpt_interventions.append(inter_df)

        ckpt_interventions = pd.concat(ckpt_interventions, axis=0, ignore_index=True)
        ckpt_interventions.to_csv(table_dir / f"interventions_ckpt{ckpt}.csv", index=False)
        all_interventions.append(ckpt_interventions)

        # 6. Sensitivity analysis
        sensitivity_df = run_sensitivity(
            H_scores=H_scores,
            D_scores=D_scores,
            R_mag_scores=R_mag_scores,
            S_scores=S_scores,
            D_counts=D_counts,
            tau_h_values=[0.7, 0.8, 0.9],
            tau_r_values=[0.7, 0.8, 0.9],
            tau_s_values=[0.6, 0.75, 0.9],
            d_abs_tau_values=[1e-4, 1e-3, 5e-3, 1e-2],
            min_count_values=[1, 5, 10],
        )
        sensitivity_df["ckpt"] = ckpt
        sensitivity_df.to_csv(table_dir / f"sensitivity_ckpt{ckpt}.csv", index=False)
        all_sensitivity.append(sensitivity_df)

        # 7. Representative concepts for visualization
        reps_df = select_representative_concepts(concept_df, top_n_per_bucket=1)
        if not reps_df.empty:
            reps_df["ckpt"] = ckpt
        reps_df.to_csv(table_dir / f"representative_concepts_ckpt{ckpt}.csv", index=False)
        all_representatives.append(reps_df)

    # Combined tables for paper
    pd.concat(all_recon, axis=0).to_csv(table_dir / "reconstruction_fidelity.csv", index=False)
    pd.DataFrame(all_model_profiles).to_csv(table_dir / "single_checkpoint_profile.csv", index=False)
    pd.DataFrame(all_model_profiles).to_csv(table_dir / "model_profiles.csv", index=False)
    pd.concat(all_interventions, axis=0).to_csv(table_dir / "interventions.csv", index=False)
    pd.concat(all_sensitivity, axis=0).to_csv(table_dir / "sensitivity.csv", index=False)
    pd.concat(all_d_sanity, axis=0).to_csv(table_dir / "d_label_sanity.csv", index=False)
    pd.concat(all_consistency_sanity, axis=0).to_csv(table_dir / "consistency_gate_sanity.csv", index=False)
    pd.concat(all_d_summaries, axis=0).to_csv(table_dir / "d_distribution_summary.csv", index=False)
    pd.concat(all_consistency_summaries, axis=0).to_csv(
        table_dir / "consistency_distribution_summary.csv",
        index=False,
    )

    nonempty_reps = [df for df in all_representatives if df is not None and not df.empty]
    if nonempty_reps:
        pd.concat(nonempty_reps, axis=0).to_csv(table_dir / "representative_concepts.csv", index=False)
    else:
        pd.DataFrame().to_csv(table_dir / "representative_concepts.csv", index=False)

    print("\nDone. Main outputs written to:")
    print(f"  Tables:  {table_dir}")
    print(f"  Figures: {fig_dir}")
    print(f"  Scores:  {score_dir}")

    print("\nImportant validation files to inspect first:")
    print(f"  {table_dir / 'd_label_sanity.csv'}")
    print(f"  {table_dir / 'consistency_gate_sanity.csv'}")
    print(f"  {table_dir / 'd_distribution_summary.csv'}")
    print(f"  {table_dir / 'consistency_distribution_summary.csv'}")
    print(f"  {table_dir / 'reconstruction_fidelity.csv'}")
    print(f"  {table_dir / 'interventions.csv'}")
    print(f"  {table_dir / 'typology_counts_ckpt3300.csv'}")
    print(f"  {table_dir / 'class_profiles_ckpt3300.csv'}")


if __name__ == "__main__":
    main()