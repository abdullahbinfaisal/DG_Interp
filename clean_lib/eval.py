"""
Masked-accuracy evaluation of a frozen classifier through the SAE.

Promoted out of refactor.ipynb, with three changes that the paper protocol
requires:

  1. Both micro and macro accuracy are reported for every domain. Macro is the
     unweighted mean over classes; micro is total correct / total images. The two
     differ substantially on sketch (~83.4 vs ~80.3) because `house` has 80
     images and `dog` has 772 while `dog` is the worst class.
  2. A predicted-label histogram per domain, so chance-level collapse (1/7 =
     14.29%) is distinguishable from genuine degradation.
  3. Full-domain deterministic loading, so repeated evaluations of the same
     configuration are bit-identical.

The evaluator takes a backbone and SAE directly rather than a Processor, so no
empty score-file template needs to exist just to describe the eval domains.

Mask polarity: True means "mask this out". A 2-D mask is (classes, concepts) and
is applied class-conditionally using each image's TRUE label, which makes every
2-D result an oracle diagnostic. Label it as such in output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import SelectAdaptivePool2d
from tqdm import tqdm

from clean_lib.data import Load_PACS_full
from clean_lib.utils import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedAccuracyEvaluator:
    def __init__(
        self,
        backbone,
        sae,
        domains: List[str],
        class_names: List[str],
        nb_concepts: int,
        batch_size: int = 64,
        pacs_root: Optional[str] = None,
        limit_batches: Optional[int] = None,
    ):
        self.backbone = backbone
        self.sae = sae
        self.domains = list(domains)
        self.class_names = list(class_names)
        self.classes = len(class_names)
        self.nb_concepts = nb_concepts
        self.batch_size = batch_size
        self.pacs_root = pacs_root
        self.limit_batches = limit_batches

        self.backbone.to(device).eval()
        self.sae.to(device).eval()
        self._pool = SelectAdaptivePool2d(pool_type="avg", flatten=True)
        self._is_resnet = self.backbone.featurizer.__class__.__name__ == "ResNet"

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------

    def _loader(self, domain: str):
        loader = Load_PACS_full(
            domains=[domain], batch_size=self.batch_size, root_dir=self.pacs_root
        )
        if self.limit_batches:
            from clean_lib.processors.processor import _TruncatedLoader

            loader = _TruncatedLoader(loader, self.limit_batches)
        return loader

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate_original(self) -> dict:
        """Accuracy of the unmodified backbone. No SAE in the path."""
        return self._evaluate(mode="original", concept_mask=None)

    def evaluate_reconstruction(self) -> dict:
        """Accuracy through the SAE with nothing masked. The reconstruction baseline."""
        return self._evaluate(mode="sae", concept_mask=None)

    def evaluate_masked(self, concept_mask: torch.Tensor) -> dict:
        """Accuracy through the SAE with `concept_mask` zeroed in code space."""
        self._validate_mask(concept_mask)
        return self._evaluate(mode="sae", concept_mask=concept_mask)

    def evaluate_posteriors(self, domain: str, concept_mask: Optional[torch.Tensor] = None):
        """
        Per-image softmax posteriors for one domain, through the SAE, with an
        optional 1-D concept mask applied uniformly to every image -- NOT keyed
        by each image's own true label, unlike the 2-D mask path elsewhere in
        this class. For label-free / any-class-conditioned experiments
        (docs/DIRECTIONS.md W2/D4) where the mask's conditioning class need not
        be the image's true label, e.g. testing what happens if every image in
        the batch is masked "as if" it belonged to some class k.

        Returns (probs, y): probs is (N, classes) softmax output, y is (N,)
        true labels, in the domain's deterministic (Load_PACS_full) image
        order -- so results from separate calls with different masks are
        index-aligned and can be compared/subtracted directly.
        """
        if concept_mask is not None:
            self._validate_mask(concept_mask)
            if concept_mask.ndim != 1:
                raise ValueError(
                    "evaluate_posteriors only supports a 1-D mask (the same "
                    "concepts zeroed for every image); a 2-D mask would be "
                    "keyed by each image's true label, defeating the point "
                    "of a label-free / any-class-conditioned evaluation."
                )
            concept_mask = concept_mask.to(device)

        all_probs, all_y = [], []
        for x, y in tqdm(self._loader(domain), desc=f"posteriors[{domain}]", leave=False):
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            with torch.no_grad():
                logits = self._sae_forward(x, n, concept_mask, is_2d=False, y=y)
                all_probs.append(F.softmax(logits, dim=1).cpu())
                all_y.append(y.cpu())

        return torch.cat(all_probs, dim=0), torch.cat(all_y, dim=0)

    def evaluate_posteriors_indexed(
        self, domain: str, mask_table: torch.Tensor, mask_id: torch.Tensor
    ):
        """
        Per-image softmax posteriors for one domain, with a PER-IMAGE concept
        mask selected from `mask_table` by `mask_id[i]` -- distinct from both
        evaluate_posteriors()'s single shared mask and evaluate_masked()'s
        true-label-keyed 2-D mask. For experiments where the masking class (or
        the concept SET) varies per image based on some model-derived
        quantity -- e.g. each image's own top-1/top-2 predicted-class pair --
        not the true label and not one fixed class for the whole domain.

        mask_table: (M, nb_concepts) bool -- M distinct concept masks.
        mask_id: (N,) long, in the domain's deterministic (Load_PACS_full)
          image order -- mask_id[i] indexes into mask_table for image i.

        Returns (probs, y), same convention as evaluate_posteriors().
        """
        if mask_table.dtype != torch.bool or mask_table.ndim != 2 or mask_table.shape[1] != self.nb_concepts:
            raise ValueError(f"mask_table must be a bool (M, {self.nb_concepts}) tensor")
        if mask_id.ndim != 1:
            raise ValueError("mask_id must be 1-D")
        mask_table = mask_table.to(device)
        mask_id = mask_id.to(device)

        all_probs, all_y = [], []
        offset = 0
        for x, y in tqdm(self._loader(domain), desc=f"posteriors[indexed:{domain}]", leave=False):
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            batch_id = mask_id[offset:offset + n]
            offset += n
            with torch.no_grad():
                logits = self._sae_forward_indexed(x, n, mask_table, batch_id)
                all_probs.append(F.softmax(logits, dim=1).cpu())
                all_y.append(y.cpu())

        return torch.cat(all_probs, dim=0), torch.cat(all_y, dim=0)

    def _sae_forward_indexed(self, x, n, mask_table, batch_id):
        """Same encode/decode path as _sae_forward, but the concept set zeroed
        for each image is looked up per-image from `mask_table[batch_id]`
        rather than being one mask shared by the whole batch or 2-D-keyed by
        true label."""
        z_raw = extract_features(self.backbone, x)
        z_norm = self.sae.normalizer(z_raw)
        _, _, h, w = z_norm.shape

        z_flat = rearrange(z_norm, "n c h w -> (n h w) c")
        _, z_sae = self.sae.encode(z_flat)

        per_token = mask_table[batch_id].repeat_interleave(h * w, dim=0)
        z_sae = z_sae * (~per_token).to(z_sae.dtype)

        z_recon = rearrange(
            self.sae.decode(z_sae), "(n h w) c -> n c h w", n=n, h=h, w=w
        )
        z_recon = self.sae.normalizer.denormalize(z_recon)
        return self._classify(z_recon)

    def _validate_mask(self, concept_mask: torch.Tensor) -> None:
        if concept_mask.dtype != torch.bool:
            raise TypeError("concept_mask must be a boolean tensor")
        if concept_mask.ndim == 1:
            if concept_mask.shape[0] != self.nb_concepts:
                raise ValueError(
                    f"1-D mask length {concept_mask.shape[0]} != nb_concepts {self.nb_concepts}"
                )
        elif concept_mask.ndim == 2:
            expected = (self.classes, self.nb_concepts)
            if tuple(concept_mask.shape) != expected:
                raise ValueError(
                    f"2-D mask shape {tuple(concept_mask.shape)} != {expected}"
                )
        else:
            raise ValueError(
                f"mask must be 1-D or 2-D, got shape {tuple(concept_mask.shape)}"
            )

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _evaluate(self, mode: str, concept_mask: Optional[torch.Tensor]) -> dict:
        if concept_mask is not None:
            concept_mask = concept_mask.to(device)
            is_2d = concept_mask.ndim == 2
        else:
            is_2d = False

        per_domain = {}

        for domain in self.domains:
            correct = torch.zeros(self.classes, dtype=torch.long, device=device)
            total = torch.zeros(self.classes, dtype=torch.long, device=device)
            pred_hist = torch.zeros(self.classes, dtype=torch.long, device=device)

            desc = f"eval[{mode}] {domain}"
            for x, y in tqdm(self._loader(domain), desc=desc, leave=False):
                x, y = x.to(device), y.to(device)
                n = x.size(0)

                with torch.no_grad():
                    if mode == "original":
                        logits = self._original_forward(x)
                    else:
                        logits = self._sae_forward(x, n, concept_mask, is_2d, y)
                    preds = logits.argmax(dim=1)

                    pred_hist += torch.bincount(preds, minlength=self.classes)
                    total += torch.bincount(y, minlength=self.classes)
                    hit = preds == y
                    correct += torch.bincount(y[hit], minlength=self.classes)

            per_domain[domain] = {
                "correct": correct.cpu(),
                "total": total.cpu(),
                "pred_hist": pred_hist.cpu(),
            }

        return self._build_report(per_domain)

    def _classify(self, feature_map):
        """Feature map -> logits.

        CheckpointManager replaces the ResNet's global_pool with Identity so the
        featurizer emits the 7x7x2048 map the SAE needs, so the adaptive average
        pool has to be reinstated here before the linear head. Same convention as
        the D and R processors.
        """
        if self._is_resnet:
            return self.backbone.classifier(self._pool(feature_map))
        return self.backbone.classifier(feature_map)

    def _original_forward(self, x):
        """Unmodified model: features straight to the classifier, no SAE.

        The DomainBed algorithm wrapper (ERM, MMD, ...) defines no `forward`, so
        calling self.backbone(x) raises NotImplementedError. Go through
        extract_features + .classifier instead.
        """
        return self._classify(extract_features(self.backbone, x))

    def _sae_forward(self, x, n, concept_mask, is_2d, y):
        z_raw = extract_features(self.backbone, x)
        z_norm = self.sae.normalizer(z_raw)
        _, _, h, w = z_norm.shape

        z_flat = rearrange(z_norm, "n c h w -> (n h w) c")
        _, z_sae = self.sae.encode(z_flat)

        if concept_mask is not None:
            if is_2d:
                # Each image gets the mask row for its own (true) class, repeated
                # across its h*w spatial tokens.
                per_token = concept_mask[y].repeat_interleave(h * w, dim=0)
                z_sae = z_sae * (~per_token).to(z_sae.dtype)
            else:
                z_sae = z_sae.clone()
                z_sae[:, concept_mask] = 0.0

        z_recon = rearrange(
            self.sae.decode(z_sae), "(n h w) c -> n c h w", n=n, h=h, w=w
        )
        z_recon = self.sae.normalizer.denormalize(z_recon)
        return self._classify(z_recon)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def _build_report(self, per_domain: dict) -> dict:
        """
        {
          "accuracy": { domain: {"micro":f, "macro":f, "per_class":{name:f}} },
          "counts":   { domain: {"correct":int, "total":int, "per_class_total":{}} },
          "predicted_label_histogram": { domain: {name:int} },
        }

        Deliberately absent: any average over all four domains. Three are in
        distribution and one is the held-out target; pooling them hides the only
        number that carries the argument.
        """
        accuracy: Dict[str, dict] = {}
        counts: Dict[str, dict] = {}
        hist: Dict[str, dict] = {}

        for domain, s in per_domain.items():
            correct, total, ph = s["correct"], s["total"], s["pred_hist"]

            per_class = {}
            for i, name in enumerate(self.class_names):
                t = int(total[i])
                per_class[name] = (float(correct[i]) / t) if t > 0 else float("nan")

            present = [v for v in per_class.values() if v == v]  # drop NaN
            macro = sum(present) / len(present) if present else float("nan")
            tot = int(total.sum())
            micro = (float(correct.sum()) / tot) if tot > 0 else float("nan")

            accuracy[domain] = {"micro": micro, "macro": macro, "per_class": per_class}
            counts[domain] = {
                "correct": int(correct.sum()),
                "total": tot,
                "per_class_total": {
                    n: int(total[i]) for i, n in enumerate(self.class_names)
                },
            }
            hist[domain] = {n: int(ph[i]) for i, n in enumerate(self.class_names)}

        return {
            "accuracy": accuracy,
            "counts": counts,
            "predicted_label_histogram": hist,
        }

    # ------------------------------------------------------------------
    # pretty printing
    # ------------------------------------------------------------------

    def print_report(self, report: dict, title: str = "") -> None:
        acc = report["accuracy"]
        width = 22 + 12 * (len(self.class_names) + 2)

        print("=" * width)
        if title:
            print(f"  {title}")
            print("-" * width)
        header = f"{'domain':<22}" + "".join(f"{n:>12}" for n in self.class_names)
        print(header + f"{'MICRO':>12}{'MACRO':>12}")
        print("-" * width)

        for domain, a in acc.items():
            row = f"{domain:<22}"
            for n in self.class_names:
                v = a["per_class"][n]
                row += f"{v * 100:>11.2f}%" if v == v else f"{'n/a':>12}"
            row += f"{a['micro'] * 100:>11.2f}%{a['macro'] * 100:>11.2f}%"
            print(row)
        print("=" * width)

        # Surface collapse: if predictions pile onto one label, say so loudly.
        for domain, h in report["predicted_label_histogram"].items():
            tot = sum(h.values()) or 1
            top, cnt = max(h.items(), key=lambda kv: kv[1])
            if cnt / tot > 0.5:
                print(
                    f"  [collapse] {domain}: {cnt / tot * 100:.1f}% of predictions "
                    f"are '{top}'  (chance = {100 / self.classes:.2f}%)"
                )
