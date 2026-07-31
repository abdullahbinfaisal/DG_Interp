import torch
from tqdm import tqdm
from clean_lib.utils import extract_features
from clean_lib.processors.processor import Processor
from einops import rearrange

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class H(Processor):
    """Activation invariance: normalised entropy of mean activation across the
    processed domains. Says where a concept fires, not whether it helps.

    __init__ and from_processor are inherited from Processor.
    """

    def calculate_mean_activations(self):
        z = torch.zeros((7, self.sae_manager.nb_concepts, len(self.domains))).to(device)
        counts = torch.zeros((7, self.sae_manager.nb_concepts)).to(device)
        n_images = torch.zeros((7, len(self.domains))).to(device)
        self.backbone.to(device)
        self.sae.to(device)


        for d, domain in enumerate(self.domains):

            loader = self.loader(domain)

            for i, batch in enumerate(tqdm(loader, desc=f"H [{domain}]")):
                with torch.no_grad():
                    img, y = batch
                    img, y = img.to(device), y.to(device)

                    x = extract_features(self.backbone, img)
                    x = self.sae.normalizer(x)
                    x = rearrange(x, self.sae_manager.rearrange_string)

                    _, heatmaps = self.sae.encode(x)

                    heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=self.sae_manager.w, h=self.sae_manager.w)
                    heatmaps_summed = heatmaps.sum(dim=1).sum(dim=1)  # (n, d)

                    for cls in range(7):
                        mask = (y == cls).squeeze()
                        if mask.sum() == 0:
                            continue

                        cls_heatmaps = heatmaps_summed[mask]           # (n_cls, d)
                        z[cls, :, d] += cls_heatmaps.sum(dim=0)
                        n_images[cls, d] += mask.sum()
                        counts[cls] += (cls_heatmaps > 0).sum(dim=0)

        n_images_safe = n_images.unsqueeze(1).clamp(min=1)
        z = z / n_images_safe

        return z, counts
    
    def calculate_invariance(self, z):

        invariance = torch.zeros((self.classes, self.sae_manager.nb_concepts)).to(z.device)

        for cls in range(self.classes):
            processed = z[cls]  # (nb_concepts, domains)

            for i in range(self.sae_manager.nb_concepts):
                if processed[i, :].sum() == 0:
                    continue

                score = processed[i, :] / processed[i, :].sum()
                entropy = -1 / torch.log(torch.tensor(float(len(self.domains)))) * (score * torch.log(score + 1e-12)).sum()
                invariance[cls, i] = entropy

        return invariance

    def process(self):
        activations, counts = self.calculate_mean_activations()
        invariance = self.calculate_invariance(activations)

        self.dump(invariance, "H")
        self.dump(activations, "H_mean_acts")
        self.dump(counts, "H_counts")
        