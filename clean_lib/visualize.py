import os
import torch
import random
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision import transforms
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA
from overcomplete.visualization.plot_utils import (interpolate_cv2, show)
from clean_lib.utils import extract_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



dirs = {
    "art_painting"  : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\art_painting",
    "sketch"        : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\sketch",
    "photo"         : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\photo",
    "cartoon"       : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\cartoon",
}

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _class_name(class_idx, domain_roots):
    """Resolve a class index the same way the dataloader does: sorted dir names."""
    all_classes = set()
    for d_path in domain_roots.values():
        for entry in os.scandir(d_path):
            if entry.is_dir():
                all_classes.add(entry.name)
    return sorted(all_classes)[class_idx]


def visualize_all_activating_on_class(
    concept,
    class_idx,
    sae_manager,
    ckpt,
    domain_roots=dirs,
    save_dir=None,
    n_cols=10,
    page_size=60,
    max_images=None,
    sort_by_activation=True,
    batch_size=32,
    thumb=160,
    dpi=110,
):
    """
    Every image of a class on which a concept fires, not just the top 8.

    Same scoring path as visualize_concept_on_class -- same transform, same
    rearrange, same activation definition (a concept is active on an image when
    its spatial map sums above zero) -- so the population shown here is exactly
    the one H_counts counts. Summing the printed per-domain counts over the three
    SOURCE domains reproduces H_counts[class_idx, concept]; sketch is included
    here for inspection only and contributes to no score.

    Written for a machine that falls over under allocation churn (CLAUDE.md 6.1).
    Four things keep it survivable, and all of them matter:

      1. Batched forward passes. One image at a time meant a separate encode and
         a separate host copy per image; batches of 32 cut both by ~30x.
      2. The concept column is sliced on the GPU. The full sparse code is
         (n, 7, 7, 16384): copying it to host is ~100 MB per batch of 32, and
         doing that a thousand times is what kills the kernel. Slicing first
         moves 49 floats per image instead.
      3. Only a float and a 7x7 map survive pass 1; images are reloaded in pass 2
         one page at a time, as uint8 thumbnails. Matplotlib holds every array
         alive inside its artist until the figure is closed, so a page of
         224x224 float RGB would cost ~600 KB per tile.
      4. The 7x7 map is handed to imshow directly with bilinear interpolation
         rather than being upsampled to full size first, so no interpolated
         array is ever materialised.

    Long tails are paginated rather than crammed into one unreadable figure: a
    concept firing on 949 images at 10 columns would otherwise be a 95-row,
    150-inch-tall canvas that matplotlib will not render usefully.

    Args:
        concept: SAE latent index.
        class_idx: class index in sorted-directory order (0 dog ... 6 person).
        n_cols: images per row.
        page_size: images per figure; further images spill onto more pages.
            Lower this first if memory is still tight.
        max_images: hard cap per domain after sorting. None keeps everything.
        sort_by_activation: strongest first. Set False for corpus order.
        batch_size: images per forward pass. Lower it if VRAM is the limit.
        thumb: display size in pixels per tile. 160 is ample at 1.5in/110dpi.
        dpi: savefig resolution.

    Returns:
        {domain: n_active} so the counts can be checked against the score file.
    """
    sae = sae_manager.get_sae(ckpt)
    backbone = sae_manager.get_backbone(ckpt)
    backbone.to(device)
    sae.to(device)
    w = sae_manager.w

    target_class_name = _class_name(class_idx, domain_roots)
    counts = {}

    for domain, dir_path in domain_roots.items():
        class_dir = os.path.join(dir_path, target_class_name)
        if not os.path.exists(class_dir):
            continue

        fnames = sorted(os.listdir(class_dir))

        # ---- pass 1: score in batches, keep only (sum, path, 7x7 map) --------
        found = []
        for start in range(0, len(fnames), batch_size):
            batch_names = fnames[start:start + batch_size]
            tensors = []
            for fname in batch_names:
                with Image.open(os.path.join(class_dir, fname)) as im:
                    tensors.append(_TRANSFORM(im.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            del tensors

            with torch.no_grad():
                x = extract_features(backbone, batch)
                x = sae.normalizer(x)
                x = rearrange(x, sae_manager.rearrange_string)
                _, z = sae.encode(x)
                # unpack with the pattern that matches rearrange_string; using
                # '(n h w)' here would silently transpose the map, and h == w
                # means nothing would error
                z = rearrange(z, '(n w h) d -> n w h d', w=w, h=w)
                # slice the single concept BEFORE leaving the GPU: the full code
                # is (n, 7, 7, 16384) and copying that to host is ~100 MB per
                # batch of 32, which is what makes a naive loop unsurvivable here
                maps = z[:, :, :, concept].detach().cpu().numpy()

            del batch, x, z
            for fname, hm in zip(batch_names, maps):
                total = float(hm.sum())
                if total > 0:
                    found.append((total, os.path.join(class_dir, fname),
                                  hm.astype(np.float32)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if sort_by_activation:
            found.sort(key=lambda t: t[0], reverse=True)
        if max_images is not None:
            found = found[:max_images]

        counts[domain] = len(found)
        print(f"[{domain:<13}] {len(found):>4} of {len(fnames):>4} "
              f"{target_class_name} images activate concept {concept}")
        if not found:
            continue

        # ---- pass 2: draw, reloading images one page at a time ---------------
        n_pages = (len(found) + page_size - 1) // page_size
        for page in range(n_pages):
            chunk = found[page * page_size:(page + 1) * page_size]
            n_rows = (len(chunk) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 1.5, n_rows * 1.6),
                                     squeeze=False)

            for i, ax in enumerate(axes.ravel()):
                ax.axis("off")
                if i >= len(chunk):
                    continue
                total, path, hm = chunk[i]
                # uint8 thumbnail, not a normalised float tensor: matplotlib
                # keeps the array alive inside the artist until the figure is
                # closed, so a page of 224x224 float RGB costs ~600 KB each
                with Image.open(path) as im:
                    thumb_img = im.convert("RGB").resize((thumb, thumb))
                    arr = np.asarray(thumb_img, dtype=np.uint8)
                ax.imshow(arr)
                # let matplotlib smooth the 7x7 at draw time instead of
                # materialising a full-size interpolated array per tile
                ax.imshow(hm, cmap=VIRIDIS_ALPHA, alpha=1.0,
                          interpolation="bilinear",
                          extent=(0, thumb, thumb, 0))
                ax.set_title(f"{total:.2f}", fontsize=6, pad=1)
                del arr

            suffix = f"  (page {page + 1}/{n_pages})" if n_pages > 1 else ""
            fig.suptitle(
                f"{target_class_name} — concept {concept} — {domain} — "
                f"{len(found)} activating{suffix}", fontsize=11)
            # subplots_adjust rather than tight_layout: the latter solves a
            # constrained layout over every axes and is very slow past ~100
            fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                                wspace=0.05, hspace=0.22)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                tag = f"_p{page + 1}" if n_pages > 1 else ""
                fig.savefig(
                    os.path.join(
                        save_dir,
                        f"Class{class_idx}_Concept{concept}_{domain}_all{tag}.png"),
                    dpi=dpi)
            else:
                plt.show()
            plt.close(fig)          # always closed, on both paths

    print(f"\ntotal activating across the domains shown: {sum(counts.values())}")
    return counts


def visualize_concept_on_class(concept, class_idx, sae_manager, ckpt, domain_roots=dirs, save_dir=None, n_images=None):
    

    sae = sae_manager.get_sae(ckpt)
    backbone = sae_manager.get_backbone(ckpt)
    backbone.to(device)
    rearrange_string = sae_manager.rearrange_string
    w = sae_manager.w

    domain_top_images = {}  # domain -> list of (heatmap_sum, img_tensor, heatmap)
    
    # 1. Build the global sorted class list EXACTLY like the dataloader
    all_classes = set()
    for d_path in domain_roots.values():
        for entry in os.scandir(d_path):
            if entry.is_dir():
                all_classes.add(entry.name)
    sorted_classes = sorted(list(all_classes))
    
    # 2. Map the index to the actual string name
    target_class_name = sorted_classes[class_idx]

    for domain, dir_path in domain_roots.items():
        # 3. Safely build the path using the string name
        class_dir = os.path.join(dir_path, target_class_name)
        
        if not os.path.exists(class_dir):
            continue
            
        images = [Image.open(os.path.join(class_dir, path)) for path in os.listdir(class_dir)]

        if n_images is not None and n_images < len(images):
            images = random.sample(images, n_images)

        results = []  # (heatmap_sum, img_tensor, heatmap)

        for i, img in enumerate(images):
            img = img.convert("RGB")
            transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])

            img_tensor = transform(img).unsqueeze(dim=0).to(device) # Don't forget to send to device!

            x = extract_features(backbone, img_tensor)
            x = sae.normalizer(x)
            
            x = rearrange(x, rearrange_string)
            
            _, z = sae.encode(x)
            
            # FIX 2: Use dynamic spatial dimensions
            z = rearrange(z, '(n w h) d -> n w h d', w=w, h=w)
            
            width, height = img_tensor.shape[-1], img_tensor.shape[-2]
            
            # FIX 3: Isolate the specific 2D image, detach, move to CPU, and convert to numpy
            heatmap_2d = z[0, :, :, concept].detach().cpu().numpy()
            
            heatmap = interpolate_cv2(heatmap_2d, (width, height))
            heatmap_sum = heatmap.sum()

            if heatmap_sum > 0:
                results.append((heatmap_sum, img_tensor.cpu(), heatmap)) # Move img_tensor back to CPU for storage/plotting

        # Sort by activation and keep top 8
        results.sort(key=lambda x: x[0], reverse=True)
        domain_top_images[domain] = results[:8]

    # Build grid: rows = domains, cols = top-8 images
    domains = list(domain_top_images.keys())
    n_domains = len(domains)
    n_cols = 8

    fig, axes = plt.subplots(n_domains, n_cols, figsize=(n_cols * 2, n_domains * 2))

    # Ensure axes is always 2D
    if n_domains == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, domain in enumerate(domains):
        top = domain_top_images[domain]
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(top):
                _, img_tensor, heatmap = top[col]
                # Convert image tensor to HWC numpy for display
                show(img_tensor, ax=ax)
                show(heatmap, ax=ax, cmap=VIRIDIS_ALPHA, alpha=1.0)
            if col == 0:
                ax.set_title(domain, fontsize=8, loc='left', pad=2)

    plt.suptitle(f"Class {target_class_name} — Concept {concept} | Top 8 Activations", fontsize=11, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Class{class_idx}_Concept{concept}_Grid.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()