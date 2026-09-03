#!/usr/bin/env python3

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-count", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    src_images = src / "images"
    src_labels = src / "labels"

    images = sorted(
        x for x in src_images.iterdir()
        if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )

    if len(images) <= args.val_count:
        raise RuntimeError(
            f"Found only {len(images)} images; "
            f"cannot reserve {args.val_count} for validation."
        )

    if dst.exists():
        shutil.rmtree(dst)

    for split in ["train", "val"]:
        (dst / "images" / split).mkdir(parents=True)
        (dst / "labels" / split).mkdir(parents=True)

    rng = random.Random(args.seed)

    shuffled = images.copy()
    rng.shuffle(shuffled)

    val_names = {
        x.name for x in shuffled[:args.val_count]
    }

    n_train = 0
    n_val = 0

    for image in images:
        split = "val" if image.name in val_names else "train"

        label = src_labels / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(
                f"Missing label for {image.name}: {label}"
            )

        shutil.copy2(
            image,
            dst / "images" / split / image.name
        )

        shutil.copy2(
            label,
            dst / "labels" / split / label.name
        )

        if split == "train":
            n_train += 1
        else:
            n_val += 1

    print(f"Input images : {len(images)}")
    print(f"Train        : {n_train}")
    print(f"Validation   : {n_val}")
    print(f"Seed         : {args.seed}")
    print(f"Output       : {dst}")


if __name__ == "__main__":
    main()
