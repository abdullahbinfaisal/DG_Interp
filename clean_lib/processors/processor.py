import json
import os
import torch
from pathlib import Path
import builtins
from clean_lib.data import pacs_domains, Load_PACS, Load_PACS_full
from clean_lib.config import PACSConfig


class _TruncatedLoader:
    """Wraps a DataLoader to yield at most `n` batches, keeping len() honest so
    tqdm still reports a sensible total. Used only by --limit-batches smoke runs."""

    def __init__(self, loader, n):
        self._loader = loader
        self._n = min(n, len(loader))

    def __len__(self):
        return self._n

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                return
            yield batch


class Processor:
    def __init__(
        self,
        sae_manager,
        ckpt,
        process_domains,
        file_path,
        dataset="PACS",
        split="full",
        batch_size=64,
        pacs_root=None,
        limit_batches=None,
    ):
        """
        split:
            "full"  every image in the domain, no split / shuffle / drop_last.
                    Deterministic. Default, and what the paper reports.
            "train" legacy 80% shuffled train split with drop_last=True. Kept
                    only to reproduce pre-2026-07 numbers; it silently drops a
                    different tail of images on every call.
        limit_batches:
            Cap batches per domain. For smoke-testing a code path only; any run
            using it is not a valid result.
        """

        self.sae_manager = sae_manager
        self.ckpt = ckpt

        self.sae = self.sae_manager.get_sae(self.ckpt)
        self.backbone = self.sae_manager.get_backbone(self.ckpt)

        ## All Processing Needs to be in Eval Mode
        self.sae.eval()
        self.backbone.eval()


        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.limit_batches = limit_batches
        self.pacs_root = pacs_root or PACSConfig().root

        if split not in ("full", "train"):
            raise ValueError(f"split must be 'full' or 'train', got {split!r}")

        # Configure Dataset-specific parameters
        if self.dataset == "PACS":
            self.classes = 7
            self.all_domains = pacs_domains
            self.class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


        ## Configure FIles
        self.file_path = file_path
        self.create_template()

        self.process_domains = process_domains
        self.domains = [self.all_domains[e] for e in self.process_domains]

        if self.dataset == "PACS":
            note = f" limit_batches={limit_batches}" if limit_batches else ""
            print(
                f"Configured for PACS dataset. domains={self.domains} "
                f"split={split} batch_size={batch_size}{note}"
            )

    @classmethod
    def from_processor(cls, processor: "Processor"):
        """Build a score processor sharing another processor's exact data
        configuration. Defined once here so subclasses cannot drift."""
        return cls(
            sae_manager=processor.sae_manager,
            ckpt=processor.ckpt,
            process_domains=processor.process_domains,
            file_path=processor.file_path,
            dataset=processor.dataset,
            split=processor.split,
            batch_size=processor.batch_size,
            pacs_root=processor.pacs_root,
            limit_batches=processor.limit_batches,
        )

    def loader(self, domains, batch_size=None):
        """The single place any processor obtains data. Honours split and
        limit_batches so every processor sees exactly the same images."""
        if isinstance(domains, str):
            domains = [domains]
        bs = batch_size or self.batch_size

        if self.split == "full":
            loader = Load_PACS_full(
                domains=domains, batch_size=bs, root_dir=self.pacs_root
            )
        else:
            loader, _ = Load_PACS(
                root_dir=self.pacs_root, domains=domains, batch_size=bs
            )

        if self.limit_batches:
            loader = _TruncatedLoader(loader, self.limit_batches)
        return loader



    def create_template(self):
        path_obj = Path(self.file_path)
        
        if path_obj.exists():
            return

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        template = {
            str(cls_idx): {
                str(concept_idx): {}
                for concept_idx in range(self.sae_manager.nb_concepts)
            }
            for cls_idx in range(self.classes)
        }

        # with open(self.file_path, "w") as f:
                #     json.dump(template, f, indent=4)
        with builtins.open(self.file_path, "w", encoding="utf-8") as fh:
            json.dump(template, fh, separators=(",", ":"))


    # def dump(self, scores: torch.Tensor, name: str):

    #     assert scores.shape[0] == self.classes, (f"Expected first dim {self.classes}, got {scores.shape[0]}")
    #     assert scores.shape[1] == self.sae_manager.nb_concepts, (f"Expected second dim {self.sae_manager.nb_concepts}, got {scores.shape[1]}")

    #     scores_np = scores.detach().cpu()

    #     def to_dumpable(tensor):
    #         if tensor.dim() == 0:
    #             return tensor.item()
    #         return [to_dumpable(tensor[i]) for i in range(tensor.shape[0])]

    #     with open(self.file_path, "r") as f:
    #         data = json.load(f)

    #     # Clear all existing values for this name before writing new ones
    #     for cls_idx in range(len(data)):
    #         for concept_idx in range(len(data[str(cls_idx)])):
    #             data[str(cls_idx)][str(concept_idx)].pop(name, None)

    #     for cls_idx in range(self.classes):
    #         for concept_idx in range(self.sae_manager.nb_concepts):
    #             value = to_dumpable(scores_np[cls_idx, concept_idx])
    #             data[str(cls_idx)][str(concept_idx)][name] = value

    #     with open(self.file_path, "w") as f:
    #         json.dump(data, f, indent=4)

    def dump(self, scores: torch.Tensor, name: str):
        assert scores.shape[0] == self.classes, (
            f"Expected first dim {self.classes}, got {scores.shape[0]}"
        )
        assert scores.shape[1] == self.sae_manager.nb_concepts, (
            f"Expected second dim {self.sae_manager.nb_concepts}, got {scores.shape[1]}"
        )

        # One C-level conversion of the whole tensor into nested Python lists.
        # The previous implementation recursed per element, i.e. ~114k tensor
        # index + .item() calls per score; that was pathologically slow and on
        # this machine it corrupted memory (a tensor index returning None, and
        # hard access violations on the full run). Output shape is identical:
        # a (7, C) tensor yields floats, a (7, C, D) tensor yields lists.
        scores_list = scores.detach().cpu().contiguous().tolist()

        path_obj = Path(self.file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Load existing JSON safely.
        with builtins.open(path_obj, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Overwrite this score name for every pair, leaving other scores intact.
        for cls_idx in range(self.classes):
            cls_data = data[str(cls_idx)]
            cls_scores = scores_list[cls_idx]
            for concept_idx in range(self.sae_manager.nb_concepts):
                cls_data[str(concept_idx)][name] = cls_scores[concept_idx]

        # Write atomically to avoid corrupting the score file if the process crashes.
        #
        # Compact separators, NOT indent=4. json.dumps only uses the C encoder
        # when indent is None; with indent it falls back to the pure-Python
        # generator encoder, which builds tens of millions of small string chunks
        # for a file this size. That is where the run died with 0xC0000409
        # (fail-fast) on a 34MB output. Compact output is also ~40% smaller and
        # much faster to load. json.load reads either form, so nothing
        # downstream cares.
        tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
        with builtins.open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))

        os.replace(tmp_path, path_obj)







# import math
# import torch
# from tqdm import tqdm
# from einops import rearrange
# from lib.data_handlers import Load_PACS
# import json
# import os
# import json
# from collections import defaultdict, Counter
# from overcomplete.visualization.plot_utils import (interpolate_cv2, get_image_dimensions, show)
# from overcomplete.visualization.cmaps import VIRIDIS_ALPHA



# domains = ["photo", "art_painting", "cartoon", "sketch"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def calculate_mean_activations(backbone, sae, rearrange_string, w=14, domains=domains, nb_concepts=7680):
#     activations = {}
#     for cls in range(7):
        
#         activations[cls] = {}
#         z_d = torch.zeros((len(domains), nb_concepts)).to(device)
        
#         for d, domain in enumerate(domains):
#             loader, _ = Load_PACS(domains=[domain])
#             for i, batch in enumerate(tqdm(loader)):
#                 with torch.no_grad():
#                     img, y = batch
#                     img, y = img.to(device), y.to(device)
                    
#                     x = extract_features(backbone, img)
#                     x = sae.normalizer(x)
#                     x = rearrange(x, rearrange_string)

#                     _, heatmaps = sae.encode(x)

#                     mask = (y == cls).squeeze().to(device)  # (batch_size,)
#                     heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=w, h=w)  # (n, t, d)
#                     heatmaps_filtered = heatmaps[mask]  # (n_cls, t, d)
                    
#                     z_d[d] += heatmaps_filtered.sum(dim=0).sum(dim=0).sum(dim=0)
                    

#         activations[cls] = z_d
        
#     return activations

# def save_json(data, filepath):
#     try:
#         with open(filepath, 'w') as f:
#             json.dump(data, f, indent=4)
#         print(f"Successfully saved logs to {filepath}")
#     except TypeError as e:
#         print(f"Error saving JSON: {e}. Check for non-serializable types (like tensors).")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# def calculate_invariance(activations, ent_thresh=0.7, act_thresh=0, domains=domains, nb_concepts=7680):
#     clss = 7
#     logs = {
#         "model_invariance" : 0,
#         "final_invariance_per_class": {},
#         "thresholded_concept_entropies": {}
#     }
#     for cls in range(clss):
#         # mask = probabilities[cls] != 0.25
#         processed = activations[cls] # * mask
        
#         sum_entropy = 0.0
#         class_concept_logs = []

#         for i in range(nb_concepts):
#             if processed[:, i].sum() == 0:
#                 continue

#             score = processed[:, i] / processed[:, i].sum()

#             entropy = -1 / torch.log(torch.tensor(len(domains))) * (score * torch.log(score + 1e-12)).sum()

#             if entropy > ent_thresh and processed[:, i].sum() > act_thresh:
#                 sum_entropy += entropy

#                 # --- LOG INDIVIDUAL ENTROPY ---
#                 class_concept_logs.append({
#                     "concept_index": i,
#                     "entropy": entropy.item(),
#                     "scores": [s.item() for s in score],
#                     "mean_acts": [val.item() for val in processed[:, i]]
#                 })

#         invariance_val = (sum_entropy)
#         if isinstance(invariance_val, torch.Tensor):
#             invariance_float = invariance_val.item()
#         else:
#             invariance_float = invariance_val # It might already be a float (if sum_entropy was 0.0)

#         # --- LOG FINAL INVARIANCE ---
#         logs["final_invariance_per_class"][cls] = invariance_float
#         logs["model_invariance"] += invariance_float
#         # --- LOG ALL CONCEPT ENTROPIES FOR THIS CLASS ---
#         logs["thresholded_concept_entropies"][cls] = class_concept_logs

#         print(f"Total Thresholded Entropy (INVARIANCE) for class {cls}: {invariance_float}")


#     logs["model_invariance"] /= 7
#     return logs

