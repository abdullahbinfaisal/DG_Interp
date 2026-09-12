import os
import torch
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from einops import rearrange
from overcomplete import TopKSAE
from clean_lib.data import Load_PACS, pacs_domains, DATASET_DOMAINS, Load_Dataset, truncated_report
from clean_lib.utils import extract_features
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

os.environ["WANDB_DISABLED"] = "true"



class Normalizer(nn.Module): 
    
    def __init__(self, model, dataset, domains=None, num_workers=0):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.dataset = dataset
        if self.dataset == "PACS":
            dl, _ = Load_PACS(domains=domains, batch_size=1024, num_workers=num_workers)
        else:
            dl, _ = Load_Dataset(dataset=self.dataset, domains=domains, batch_size=1024,
                                 num_workers=num_workers)
        x, _ = next(iter(dl))

        # .eval() matters: these are frozen ResNets, and in train mode their
        # BatchNorms would normalise by this batch's own statistics (and mutate
        # the checkpoint's running stats). processor.py and eval.py both score in
        # eval mode, so μ/σ have to be measured there too.
        model.to(device).eval()
        activations = extract_features(model, x.to(self.device))

        flat = activations.flatten()
        
        self.register_buffer('mean', flat.mean())
        self.register_buffer('std', flat.std())
                
    def forward(self, activations): 
        activations = (activations - self.mean)
        activations = activations / (self.std + 1e-12)
        return activations

    def denormalize(self, normalized_activations):
        return (normalized_activations * (self.std + 1e-12)) + self.mean




class SparseAEs():
    def __init__(self, feature_dim, topk, nb_concepts, rearrange_string, checkpointManager, train_envs, w, dataset="PACS"):
        self.topk = topk
        self.feature_dim = feature_dim
        self.nb_concepts = nb_concepts
        self.rearrange_string = rearrange_string
        self.train_envs = train_envs
        self.checkpointManager = checkpointManager
        self.backbones = checkpointManager.get_models()
        self.SAEs = None
        self.w = w
        self.dataset = dataset

    def get_sae(self, ckpt):
        return self.SAEs[ckpt]

    
    def get_backbone(self, ckpt):
        return self.backbones[ckpt]


    def load_checkpoint(self, checkpoint_path):
        self.SAEs = {}
        for key in self.backbones.keys():
            checkpoints = torch.load(checkpoint_path, weights_only=False)
            self.SAEs[key] = checkpoints[key]
            self.SAEs[key].train()
        

    def save_checkpoint(self, save_path, flag):
        run_name = f"{flag}_{self.checkpointManager.algorithm}_{self.checkpointManager.architecture}_T{''.join([str(e) for e in self.checkpointManager.testenvs])}"
        final_save_path = os.path.join(save_path, f"USAE_{run_name}.pt")
        torch.save(self.SAEs, final_save_path)
        print(f"SAEs saved successfully to {final_save_path}")


    def configure_training(self, learning_rate=3e-4, num_workers=0):
        self.optimizers = {}
        self.schedulers = {}

        if self.SAEs is None:
            self.SAEs = {}
            normalizer_domains = [DATASET_DOMAINS[self.dataset][e] for e in self.train_envs]
            for key in self.backbones.keys():
                self.SAEs[key] = TopKSAE(self.feature_dim, nb_concepts=self.nb_concepts, top_k=self.topk, device="cuda")
                self.SAEs[key].train()
                self.SAEs[key].normalizer = Normalizer(self.backbones[key], self.dataset,
                                                       domains=normalizer_domains, num_workers=num_workers)


        for key in self.backbones.keys():    

            self.optimizers[key] = torch.optim.Adam(self.SAEs[key].parameters(), lr=learning_rate)
            warmup_scheduler = LinearLR(self.optimizers[key], start_factor=1e-6 / learning_rate, end_factor=1.0, total_iters=10)
            cosine_scheduler = CosineAnnealingLR(self.optimizers[key], T_max=50, eta_min=1e-6)
            self.schedulers[key] = SequentialLR(self.optimizers[key], schedulers=[warmup_scheduler, cosine_scheduler], milestones=[25])
        

            self.backbones[key].to(device)
            self.SAEs[key].to(device)

        self.criterion = nn.L1Loss(reduction="mean")


    def _cache_features(self, loader, n_images, cache_subdir):
        """One pass per backbone: decode + frozen forward -> an fp32 .pt of activations.

        The backbone never updates and the transform has no random augmentation,
        so a given image yields identical activations on every epoch. Computing
        them once turns 250 epochs of JPEG decode + ResNet forward into 250
        epochs of matmul - the difference between hours and minutes on datasets
        like VLCS, whose LabelMe images average ~1 MB each.
        """
        cache_subdir.mkdir(parents=True, exist_ok=True)
        caches, written = {}, []

        for key, backbone in self.backbones.items():
            path = cache_subdir / f"features_ckpt{key}.pt"

            if path.exists():
                feats = torch.load(path, map_location="cpu")
                if feats.shape[0] == n_images:
                    print(f"[cache] reusing {path} ({n_images} images)")
                    caches[key] = feats
                    written.append(path)
                    continue
                print(f"[cache] {path} holds {feats.shape[0]} images, expected {n_images} - rebuilding")
                del feats

            backbone.to(device).eval()
            feats, offset = None, 0
            for images, _ in tqdm(loader, desc=f"[cache] ckpt{key}", leave=False):
                out = extract_features(backbone, images)
                if feats is None:
                    # preallocated, not torch.cat'd: one allocation instead of a
                    # per-batch churn this machine has crashed on before
                    feats = torch.empty((n_images, *out.shape[1:]), dtype=torch.float32)
                feats[offset:offset + out.shape[0]] = out.float().cpu()
                offset += out.shape[0]

            if offset != n_images:
                raise RuntimeError(f"cached {offset} activations, expected {n_images}")

            torch.save(feats, path)
            print(f"[cache] wrote {path} "
                  f"({feats.numel() * feats.element_size() / 1e9:.2f} GB, {n_images} images)")
            caches[key] = feats
            written.append(path)

        return caches, written


    def _place_cache(self, caches):
        """Hold the activations on GPU when they comfortably fit, else stream per batch."""
        total = sum(c.numel() * c.element_size() for c in caches.values())

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if total * 1.25 < free:
                print(f"[cache] holding {total / 1e9:.2f} GB of activations on GPU")
                return {k: v.to(device) for k, v in caches.items()}, True
            print(f"[cache] {total / 1e9:.2f} GB does not fit in {free / 1e9:.2f} GB of free GPU "
                  f"memory with headroom - streaming batches from CPU")

        return caches, False


    @staticmethod
    def _clear_cache(cache_subdir, cache_files):
        """Delete only the activation files we wrote, then the dirs if they empty out.

        Deliberately not rmtree: --cache-dir is user-supplied, and a typo there
        should cost nothing.
        """
        for path in cache_files:
            path.unlink(missing_ok=True)
        for directory in (cache_subdir, cache_subdir.parent):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        print(f"[cache] removed {len(cache_files)} activation file(s) from {cache_subdir}")


    def train(self, flag="USAE", epochs=250, batch_size=64, full_data_gpu=True, save_dir="./SAEs",
              dataset=None, num_workers=0, cache_dir=".feature_cache", keep_cache=False):
        dataset = dataset or self.dataset

        run_name = f"{flag}_{self.checkpointManager.algorithm}_{self.checkpointManager.architecture}_T{''.join([str(e) for e in self.checkpointManager.testenvs])}"
        
        # wandb.init(
        #     project="SAE_training",
        #     name=run_name,
        #     config={
        #         "flag": flag,
        #         "epochs": epochs,
        #         "batch_size": batch_size,
        #         "algorithm": self.checkpointManager.algorithm,
        #         "architecture": self.checkpointManager.architecture,
        #         "test_envs": self.checkpointManager.testenvs
        #     },
        #     mode="offline"
        # )
        # wandb.watch(list(self.SAEs.values()), log="all")

        # shuffle_train=False / drop_last=False so the caching pass walks the same
        # 80% split (random_split is seeded) exactly once, in a stable order.
        # Shuffling and the dropped tail move into the epoch loop below.
        loader_kwargs = dict(batch_size=batch_size, shuffle_train=False, drop_last=False,
                             num_workers=num_workers)
        if dataset == "PACS":
            train_dl, test_dl = Load_PACS(domains=[pacs_domains[e] for e in self.train_envs],
                                          **loader_kwargs)
        else:
            train_dl, test_dl = Load_Dataset(
                dataset=dataset,
                domains=[DATASET_DOMAINS[dataset][e] for e in self.train_envs],
                **loader_kwargs,
            )

        n_images = len(train_dl.dataset)
        n_batches = n_images // batch_size  # matches the old drop_last=True
        if n_batches == 0:
            raise ValueError(f"batch_size {batch_size} exceeds the {n_images} training images")

        cache_subdir = Path(cache_dir) / f"{dataset}_{run_name}_E{''.join(str(e) for e in self.train_envs)}"
        caches, cache_files = self._cache_features(train_dl, n_images, cache_subdir)
        caches, cache_on_gpu = self._place_cache(caches)
        cache_device = device if cache_on_gpu else torch.device("cpu")

        rotator = 0
        pbar = tqdm(range(epochs), desc=f"Training: {run_name}")
        for epoch in pbar:
            epoch_loss = 0.0
            perm = torch.randperm(n_images, device=cache_device)
            for b in range(n_batches):
                total_loss = 0.0
                names = list(self.optimizers.keys())

                idx = perm[b * batch_size:(b + 1) * batch_size]

                for k in names:
                    self.optimizers[k].zero_grad()

                # Current SAE
                current = names[rotator]

                # Current SAE SAEs
                sae = self.SAEs[current]
                sae.train()

                # Cached activations stand in for the frozen backbone's forward pass
                x = caches[current][idx]
                if not cache_on_gpu:
                    x = x.to(device, non_blocking=True)
                x = sae.normalizer(x) # Normalize
                x = rearrange(x, self.rearrange_string) # Rearrange
                _, z = sae.encode(x)

                # Decoder across all models & accumulate loss
                for n, m in self.SAEs.items():
                    if n == current:
                        x_hat = m.decode(z)
                    else:
                        x_hat = m.decode(z.detach())

                    loss = self.criterion(x_hat, x)
                    total_loss += loss

                total_loss.backward()
                
                self.optimizers[current].step()
                if self.schedulers:
                    self.schedulers[current].step()

                # Rotator Update
                rotator += 1
                rotator = rotator % len(names)

                # FIX: Accumulate batch loss for logging!
                epoch_loss += total_loss.item()
                
                # --- W&B STEP LOG ---
                # wandb.log({
                #     "batch_loss": total_loss.item(),
                #     "epoch": epoch
                # })

            # Update tqdm with loss info
            avg_epoch_loss = epoch_loss / n_batches
            pbar.set_postfix(loss=f"{avg_epoch_loss:.6f}")

            # --- W&B EPOCH LOG ---
            # wandb.log({
            #     "epoch_loss": avg_epoch_loss,
            #     "epoch": epoch
            # })

        n_truncated, n_total = truncated_report(train_dl)
        pct = 100.0 * n_truncated / n_total if n_total else 0.0
        print(f"[data] {n_truncated}/{n_total} ({pct:.2f}%) training images were truncated "
              f"and loaded partially")

        self.save_checkpoint(save_path=save_dir, flag=flag)

        if keep_cache:
            print(f"[cache] keeping {cache_subdir} (--keep-cache)")
        else:
            self._clear_cache(cache_subdir, cache_files)