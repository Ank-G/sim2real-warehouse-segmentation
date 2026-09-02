from pathlib import Path
import json
import random
import shutil


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "real" / "raw"

RAW_VALID_IMAGES = RAW_DIR / "valid" / "images"
RAW_VALID_LABELS = RAW_DIR / "valid" / "labels"

RAW_TEST_IMAGES = RAW_DIR / "test" / "images"
RAW_TEST_LABELS = RAW_DIR / "test" / "labels"

FINETUNE_DIR = PROJECT_ROOT / "data" / "real" / "finetune_20"
FINETUNE_IMAGES = FINETUNE_DIR / "images"
FINETUNE_LABELS = FINETUNE_DIR / "labels"

TEST_DIR = PROJECT_ROOT / "data" / "real" / "test"
TEST_IMAGES = TEST_DIR / "images"
TEST_LABELS = TEST_DIR / "labels"

MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "split_manifest.json"

NUM_FINETUNE_IMAGES = 20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_images(directory: Path):
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_label(image_path: Path, label_directory: Path):
    return label_directory / f"{image_path.stem}.txt"


def check_pairs(images, label_directory):
    valid_pairs = []
    missing_labels = []

    for image in images:
        label = get_label(image, label_directory)

        if label.exists():
            valid_pairs.append((image, label))
        else:
            missing_labels.append(image)

    return valid_pairs, missing_labels


def clear_directory(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def copy_pair(image, label, output_images, output_labels):
    shutil.copy2(image, output_images / image.name)
    shutil.copy2(label, output_labels / label.name)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 65)
    print("REAL DATASET PREPARATION")
    print("=" * 65)

    # ----------------------------------------------
    # Verify required folders
    # ----------------------------------------------

    required_directories = [
        RAW_VALID_IMAGES,
        RAW_VALID_LABELS,
        RAW_TEST_IMAGES,
        RAW_TEST_LABELS,
    ]

    for directory in required_directories:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory not found:\n{directory}"
            )

    # ----------------------------------------------
    # Read validation images
    # ----------------------------------------------

    valid_images = get_images(RAW_VALID_IMAGES)

    valid_pairs, valid_missing = check_pairs(
        valid_images,
        RAW_VALID_LABELS,
    )

    print(f"\nRaw validation images: {len(valid_images)}")
    print(f"Valid image/label pairs: {len(valid_pairs)}")

    if valid_missing:
        print(
            f"WARNING: {len(valid_missing)} "
            "validation images have no label file."
        )

    if len(valid_pairs) < NUM_FINETUNE_IMAGES:
        raise RuntimeError(
            "Not enough labelled validation images "
            f"to select {NUM_FINETUNE_IMAGES} samples."
        )

    # ----------------------------------------------
    # Deterministically select 20 real images
    # ----------------------------------------------

    rng = random.Random(RANDOM_SEED)

    selected_pairs = rng.sample(
        valid_pairs,
        NUM_FINETUNE_IMAGES,
    )

    selected_pairs = sorted(
        selected_pairs,
        key=lambda pair: pair[0].name,
    )

    # ----------------------------------------------
    # Prepare output directories
    # ----------------------------------------------

    clear_directory(FINETUNE_IMAGES)
    clear_directory(FINETUNE_LABELS)

    for image, label in selected_pairs:
        copy_pair(
            image,
            label,
            FINETUNE_IMAGES,
            FINETUNE_LABELS,
        )

    # ----------------------------------------------
    # Prepare untouched real test set
    # ----------------------------------------------

    test_images = get_images(RAW_TEST_IMAGES)

    test_pairs, test_missing = check_pairs(
        test_images,
        RAW_TEST_LABELS,
    )

    clear_directory(TEST_IMAGES)
    clear_directory(TEST_LABELS)

    for image, label in test_pairs:
        copy_pair(
            image,
            label,
            TEST_IMAGES,
            TEST_LABELS,
        )

    # ----------------------------------------------
    # Save experiment manifest
    # ----------------------------------------------

    manifest = {
        "random_seed": RANDOM_SEED,
        "num_finetune_images": NUM_FINETUNE_IMAGES,
        "finetune_source_split": "Roboflow validation",
        "test_source_split": "Roboflow test",
        "finetune_images": [
            image.name
            for image, _ in selected_pairs
        ],
        "num_test_images": len(test_pairs),
        "test_images": [
            image.name
            for image, _ in test_pairs
        ],
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 65)

    print(
        f"\nFine-tuning images: "
        f"{len(selected_pairs)}"
    )

    print(
        f"Real test images: "
        f"{len(test_pairs)}"
    )

    if test_missing:
        print(
            f"\nWARNING: {len(test_missing)} "
            "test images had no label file."
        )

    print(
        f"\nSplit manifest saved to:\n"
        f"{MANIFEST_PATH}"
    )

    print("\nSelected fine-tuning images:")

    for image, _ in selected_pairs:
        print(f"  {image.name}")

    print("\nIMPORTANT:")
    print(
        "data/real/test is the locked final "
        "evaluation set."
    )
    print(
        "Do not use these images for training "
        "or fine-tuning."
    )


if __name__ == "__main__":
    main()