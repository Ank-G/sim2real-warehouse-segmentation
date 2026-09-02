from pathlib import Path
import random

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "finetune_20": PROJECT_ROOT / "data" / "real" / "finetune_20",
    "test": PROJECT_ROOT / "data" / "real" / "test",
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "dataset_validation"
)

NUM_PREVIEWS = 6
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(directory):
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def read_segmentation_label(label_path):
    """
    Reads YOLO segmentation labels.

    Expected format:

    class_id x1 y1 x2 y2 x3 y3 ...
    """

    instances = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            # Class + minimum 3 polygon points
            if len(values) < 7:
                raise ValueError(
                    f"{label_path.name}, line {line_number}: "
                    "not enough polygon coordinates."
                )

            class_id = int(float(values[0]))

            coordinates = [
                float(value)
                for value in values[1:]
            ]

            if len(coordinates) % 2 != 0:
                raise ValueError(
                    f"{label_path.name}, line {line_number}: "
                    "odd number of polygon coordinates."
                )

            points = np.array(
                coordinates,
                dtype=np.float32,
            ).reshape(-1, 2)

            if len(points) < 3:
                raise ValueError(
                    f"{label_path.name}, line {line_number}: "
                    "polygon has fewer than 3 points."
                )

            if np.any(points < 0) or np.any(points > 1):
                raise ValueError(
                    f"{label_path.name}, line {line_number}: "
                    "coordinates outside [0, 1]."
                )

            instances.append(
                {
                    "class_id": class_id,
                    "points": points,
                }
            )

    return instances


def draw_instances(image, instances):
    height, width = image.shape[:2]

    overlay = image.copy()

    for index, instance in enumerate(instances):
        points = instance["points"].copy()

        points[:, 0] *= width
        points[:, 1] *= height

        points = points.astype(np.int32)

        # Polygon fill
        cv2.fillPoly(
            overlay,
            [points],
            (0, 255, 0),
        )

        # Polygon boundary
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Label position
        x = int(points[:, 0].min())
        y = int(points[:, 1].min())

        cv2.putText(
            image,
            f"box {index + 1}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Blend transparent masks
    result = cv2.addWeighted(
        overlay,
        0.30,
        image,
        0.70,
        0,
    )

    return result


def validate_dataset(name, dataset_dir):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    print()
    print("=" * 65)
    print(f"VALIDATING: {name}")
    print("=" * 65)

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    images = get_images(images_dir)

    print(f"Images found: {len(images)}")

    missing_labels = []
    invalid_labels = []
    total_instances = 0
    valid_images = []

    for image_path in images:
        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        try:
            instances = read_segmentation_label(
                label_path
            )

            total_instances += len(instances)

            valid_images.append(
                (
                    image_path,
                    label_path,
                    instances,
                )
            )

        except Exception as error:
            invalid_labels.append(
                (
                    label_path.name,
                    str(error),
                )
            )

    print(f"Valid image/label pairs: {len(valid_images)}")
    print(f"Total box instances: {total_instances}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    if valid_images:
        average_instances = (
            total_instances
            / len(valid_images)
        )

        print(
            f"Average boxes/image: "
            f"{average_instances:.2f}"
        )

    if missing_labels:
        print("\nImages with missing labels:")

        for filename in missing_labels[:10]:
            print(f"  {filename}")

    if invalid_labels:
        print("\nInvalid labels:")

        for filename, error in invalid_labels[:10]:
            print(f"  {filename}")
            print(f"    {error}")

    # ----------------------------------------------
    # Generate visual previews
    # ----------------------------------------------

    if not valid_images:
        return

    rng = random.Random(RANDOM_SEED)

    preview_count = min(
        NUM_PREVIEWS,
        len(valid_images),
    )

    selected = rng.sample(
        valid_images,
        preview_count,
    )

    dataset_output = OUTPUT_DIR / name
    dataset_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_path, _, instances in selected:
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"WARNING: Could not read "
                f"{image_path.name}"
            )
            continue

        visualization = draw_instances(
            image,
            instances,
        )

        output_path = (
            dataset_output
            / f"{image_path.stem}_preview.jpg"
        )

        cv2.imwrite(
            str(output_path),
            visualization,
        )

    print(
        f"\nSaved {preview_count} previews to:"
    )

    print(dataset_output)


def main():
    print("=" * 65)
    print("YOLO26 INSTANCE SEGMENTATION DATASET CHECK")
    print("=" * 65)

    for name, directory in DATASETS.items():
        validate_dataset(
            name,
            directory,
        )

    print()
    print("=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()