"""
Central configuration for Sparse Concept Diagnostics runs.

Everything a run needs — paths, SAE geometry, which domains are sources, which
checkpoint — lives here as frozen dataclasses. Scripts take `--preset NAME` and
may override individual fields from the command line; whatever is finally
resolved gets serialised into the run manifest, so any number on disk can be
traced back to the exact configuration that produced it.

Editing guide:
    - New machine / moved dataset  -> PACSConfig.root
    - New SAE checkpoint           -> SAEConfig.checkpoint_path
    - New backbone or checkpoint   -> add a RunConfig entry to PRESETS

Nothing here imports torch, so this module is cheap to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PACSConfig:
    """PACS layout and the source/target split."""

    root: str = r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS"

    # Index order matters: it is the DomainBed env numbering used throughout.
    domains: Tuple[str, ...] = ("art_painting", "cartoon", "photo", "sketch")

    # Alphabetical directory order, which is what PACSDataset assigns.
    class_names: Tuple[str, ...] = (
        "dog",
        "elephant",
        "giraffe",
        "guitar",
        "horse",
        "house",
        "person",
    )

    # Sketch is held out. H, D and R are never computed on the target.
    source_envs: Tuple[int, ...] = (0, 1, 2)
    target_envs: Tuple[int, ...] = (3,)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def domain_name(self, env: int) -> str:
        return self.domains[env]

    def domain_names(self, envs: Tuple[int, ...]) -> list[str]:
        return [self.domains[e] for e in envs]


# ---------------------------------------------------------------------------
# Sparse autoencoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SAEConfig:
    """Top-K SAE geometry, matching the trained checkpoint."""

    feature_dim: int = 2048
    topk: int = 16
    nb_concepts: int = 2048 * 8  # 16,384
    w: int = 7  # spatial side of the ResNet-50 final feature map

    # NOTE: SparseAEs uses "w h" ordering while the D/R processors hardcode
    # "h w". These agree only because w == h == 7 here.
    rearrange_string: str = "n c w h -> (n w h) c"

    checkpoint_path: str = "./SAEs/normalization_testing/USAE_ERM_Multi_test_3300_2100.pt"


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One backbone checkpoint plus everything needed to score and evaluate it."""

    name: str
    backbone_dir: str
    ckpt: int

    # Envs the backbone was trained on (DomainBed test-env complement).
    train_envs: Tuple[int, ...] = (0, 1, 2)

    # Envs used to estimate H, D, R. MUST exclude the target domain.
    score_envs: Tuple[int, ...] = (0, 1, 2)

    # Envs reported when evaluating accuracy. Includes the target.
    eval_envs: Tuple[int, ...] = (0, 1, 2, 3)

    # "full"  -> every image in the domain, no split, no shuffle, no dropped
    #            batch. Deterministic. This is the default.
    # "train" -> the legacy 80% shuffled train split with drop_last=True, kept
    #            only for reproducing pre-2026-07 numbers.
    split: str = "full"

    batch_size: int = 64
    seed: int = 42

    pacs: PACSConfig = field(default_factory=PACSConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)

    # ---- derived helpers -------------------------------------------------

    @property
    def score_domains(self) -> list[str]:
        return self.pacs.domain_names(self.score_envs)

    @property
    def eval_domains(self) -> list[str]:
        return self.pacs.domain_names(self.eval_envs)

    def validate(self) -> None:
        """Fail loudly on the mistakes that would silently invalidate a paper number."""
        leaked = set(self.score_envs) & set(self.pacs.target_envs)
        if leaked:
            names = [self.pacs.domain_name(e) for e in sorted(leaked)]
            raise ValueError(
                f"score_envs includes the held-out target domain(s) {names}. "
                "H, D and R must be estimated on source domains only. "
                "This is the paper's central methodological commitment."
            )
        if self.split not in ("full", "train"):
            raise ValueError(f"split must be 'full' or 'train', got {self.split!r}")
        if self.ckpt is None:
            raise ValueError("ckpt must be set")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs) -> "RunConfig":
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

#: The paper's primary configuration. Checkpoint 3300 is the *non-oracle*
#: selection (highest mean accuracy over source envs 0,1,2). Step 2100 is the
#: oracle selection and must not be used for headline numbers.
ERM_RESNET_3300 = RunConfig(
    name="ERM_ResNet_3300_T3",
    backbone_dir="./PACS_ResNet_Sketch_Test_Only/ERM_ResNet_T3",
    ckpt=3300,
)

PRESETS: Dict[str, RunConfig] = {
    "erm_resnet_3300": ERM_RESNET_3300,
}


def get_preset(name: str) -> RunConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]
