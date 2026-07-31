import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_uniform_concept_mask(mask: torch.Tensor, reduce: str = "any") -> torch.Tensor:
    """
    Collapse a class-by-concept mask to a uniform concept mask and broadcast it
    back across classes.

    Args:
        mask: Boolean tensor shaped (classes, concepts).
        reduce: "any" masks a concept everywhere if any class masks it.
                "all" masks a concept everywhere only if all classes mask it.

    Returns:
        Boolean tensor shaped like the input mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask shaped (classes, concepts), got {tuple(mask.shape)}.")

    if reduce == "any":
        concept_mask = mask.any(dim=0)
    elif reduce == "all":
        concept_mask = mask.all(dim=0)
    else:
        raise ValueError("reduce must be 'any' or 'all'.")

    return concept_mask.unsqueeze(0).expand(mask.shape[0], -1).clone()

def extract_features(backbone, images):
    
    with torch.no_grad():
        if hasattr(backbone, 'featurizer'): # for models trained using domainbed
            if backbone.featurizer.__class__.__name__ == "DinoV2" :
                activations = backbone.featurizer.network.forward_features(images.to(device))['x_norm_patchtokens']
            elif backbone.featurizer.__class__.__name__ == "ViT":
                activations = backbone.featurizer.network.forward_features(images.to(device))[:, 1:, :]
            else:
                activations = backbone.network[0](images.to(device))

        if hasattr(backbone, 'forward_features'): # for models directly from the overcomplete library
            activations = backbone.forward_features(images.to(device))

    return activations

