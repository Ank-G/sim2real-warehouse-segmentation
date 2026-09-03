#!/usr/bin/env python3

"""
Convert Isaac Sim colorized instance segmentation to YOLO segmentation.

Important:
Only objects belonging to --instance-root are exported.

For this project:
    Isaac generated boxes: /SDG_V2/Boxes/Box_XX
    YOLO class 0: package
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    # Kept for compatibility with previous commands.
    p.add_argument("--class-name", default="box")
    p.add_argument("--class-id", type=int, default=0)

    p.add_argument(
        "--instance-root",
        default="/SDG_V2/Boxes",
        help="Only instance paths below this USD path are exported",
    )

    p.add_argument(
        "--min-area",
        type=int,
        default=20,
    )

    p.add_argument(
        "--epsilon",
        type=float,
        default=0.003,
    )

    return p.parse_args()


def parse_color_tuple(text):
    value = ast.literal_eval(text)

    if not isinstance(value, tuple):
        raise ValueError(
            f"Unexpected instance color key: {text}"
        )

    value = tuple(int(v) for v in value)

    if len(value) == 3:
        value = value + (255,)

    return value


def load_mask_rgba(path):
    return np.asarray(
        Image.open(path).convert("RGBA")
    )


def load_generated_instance_groups(
    mapping_path,
    instance_root,
):
    """
    Returns:

        {
            "/SDG_V2/Boxes/Box_03": [
                rgba1,
                rgba2,
                ...
            ]
        }

    Multiple child prim colors are grouped back into the
    same logical Box_XX object.
    """

    with mapping_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    root = instance_root.rstrip("/")

    groups = {}

    for color_text, prim_path in mapping.items():

        if not isinstance(prim_path, str):
            continue

        if not prim_path.startswith(root + "/"):
            continue

        relative = prim_path[len(root):].lstrip("/")

        if not relative:
            continue

        # Example:
        #
        # Box_03/SM_CardBoxD_04/...
        #
        # becomes:
        #
        # Box_03
        logical_name = relative.split("/", 1)[0]

        logical_path = f"{root}/{logical_name}"

        rgba = parse_color_tuple(color_text)

        groups.setdefault(
            logical_path,
            [],
        ).append(rgba)

    return groups


def largest_valid_contour(
    binary_mask,
    min_area,
):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = [
        c
        for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea,
    )


def contour_to_yolo(
    contour,
    width,
    height,
    epsilon_factor,
):
    perimeter = cv2.arcLength(
        contour,
        True,
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon_factor * perimeter,
        True,
    )

    pts = approx.reshape(-1, 2)

    if len(pts) < 3:
        return None

    coords = []

    for x, y in pts:
        coords.extend(
            [
                float(
                    np.clip(
                        x / width,
                        0.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        y / height,
                        0.0,
                        1.0,
                    )
                ),
            ]
        )

    return coords


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb_files = sorted(
        input_dir.glob("rgb_*.png")
    )

    if not rgb_files:
        raise FileNotFoundError(
            f"No rgb_*.png files in {input_dir}"
        )

    total_instances = 0
    frames_with_labels = 0

    for rgb_path in rgb_files:

        frame_id = rgb_path.stem.replace(
            "rgb_",
            "",
        )

        mask_path = (
            input_dir
            / f"instance_segmentation_{frame_id}.png"
        )

        mapping_path = (
            input_dir
            / f"instance_segmentation_mapping_{frame_id}.json"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                mask_path
            )

        if not mapping_path.exists():
            raise FileNotFoundError(
                mapping_path
            )

        mask = load_mask_rgba(
            mask_path
        )

        h, w = mask.shape[:2]

        groups = load_generated_instance_groups(
            mapping_path,
            args.instance_root,
        )

        yolo_rows = []

        for logical_path, colors in sorted(
            groups.items()
        ):

            # Union all Isaac instance colors belonging
            # to this one logical Box_XX.
            binary = np.zeros(
                (h, w),
                dtype=bool,
            )

            for rgba in colors:

                target = np.asarray(
                    rgba,
                    dtype=np.uint8,
                )

                binary |= np.all(
                    mask == target,
                    axis=2,
                )

            contour = largest_valid_contour(
                binary,
                args.min_area,
            )

            if contour is None:
                continue

            coords = contour_to_yolo(
                contour,
                w,
                h,
                args.epsilon,
            )

            if coords is None:
                continue

            row = (
                str(args.class_id)
                + " "
                + " ".join(
                    f"{v:.6f}"
                    for v in coords
                )
            )

            yolo_rows.append(row)

        shutil.copy2(
            rgb_path,
            images_dir / rgb_path.name,
        )

        label_path = (
            labels_dir
            / f"rgb_{frame_id}.txt"
        )

        label_path.write_text(
            "\n".join(yolo_rows)
            + ("\n" if yolo_rows else ""),
            encoding="utf-8",
        )

        total_instances += len(
            yolo_rows
        )

        if yolo_rows:
            frames_with_labels += 1

        print(
            f"[YOLO] frame={frame_id} "
            f"generated_objects={len(groups)} "
            f"labels_written={len(yolo_rows)}"
        )

    print()
    print("[YOLO] Conversion complete")
    print(
        f"[YOLO] Frames: "
        f"{len(rgb_files)}"
    )
    print(
        f"[YOLO] Frames with labels: "
        f"{frames_with_labels}"
    )
    print(
        f"[YOLO] Box instances: "
        f"{total_instances}"
    )
    print(
        f"[YOLO] Output: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
