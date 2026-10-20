"""
ELEC 475: Lab 3 - Dataset Reality Check (Pre-Lab Task 3, Step 2)

1. Iterates through every sample of both splits and confirms that every image
   has shape (3, 227, 227) and every label lies inside the 227x227 frame.
2. Plots eight training images with their nose labels drawn as dots and saves
   the figure to plots/dataset_check.png.

Run from the lab folder:
    python -m utils.check_dataset
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save to file without opening a window
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import IMG_SIZE, PetNoseDataset, denormalize


def check_split(split: str) -> bool:
    """Check image shapes and label bounds for every sample in one split."""
    ds = PetNoseDataset(split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    bad_shape, bad_label, seen = 0, 0, 0
    for images, labels, _ in loader:
        if images.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
            bad_shape += images.shape[0]
        # Labels must satisfy 0 <= u, v <= 227
        outside = ((labels < 0) | (labels > IMG_SIZE)).any(dim=1)
        bad_label += int(outside.sum())
        seen += images.shape[0]

    ok = bad_shape == 0 and bad_label == 0 and seen == len(ds)
    print(f"  {split:5s}: {seen:,} samples checked | bad shapes: {bad_shape} "
          f"| labels outside frame: {bad_label} | {'OK' if ok else 'FAIL'}")
    return ok


def plot_samples(out_path: Path, n: int = 8) -> None:
    """Plot n training images with their nose labels drawn as dots."""
    ds = PetNoseDataset(split="train")
    fig, axes = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 8))
    for i, ax in enumerate(axes.flat):
        image, label, orig_size = ds[i]
        # Undo normalization (multiply by STD, add MEAN), then (3, H, W) -> (H, W, 3)
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        ax.plot(label[0].item(), label[1].item(), "o", color="red",
                markersize=8, markeredgecolor="white")
        ax.set_title(f"{ds.samples[i][0]}\n({label[0]:.1f}, {label[1]:.1f})", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Checking every sample:")
    all_ok = all([check_split("train"), check_split("test")])
    print("All samples passed." if all_ok else "Some samples FAILED, see above.")

    plot_samples(Path("plots") / "dataset_check.png")
