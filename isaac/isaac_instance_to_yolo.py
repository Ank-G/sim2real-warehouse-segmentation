#!/usr/bin/env python3
"""
Convert Isaac Sim BasicWriter RGB + colorized instance-segmentation output
to Ultralytics/YOLO instance-segmentation labels.

Expected Isaac files:
  rgb_0000.png
  instance_segmentation_0000.png
  instance_segmentation_semantics_mapping_0000.json

Output:
  <output>/images/*.png
  <output>/labels/*.txt

YOLO segmentation row:
  class_id x1 y1 x2 y2 ... xn yn
where coordinates are normalized to [0, 1].

For this project:
  class 0 = box
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
    p.add_argument("--input", required=True, help="Isaac BasicWriter output directory")
    p.add_argument("--output", required=True, help="YOLO dataset output directory")
    p.add_argument("--class-name", default="box")
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Ignore tiny mask components smaller than this many pixels",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.003,
        help="Contour simplification factor relative to contour perimeter",
    )
    return p.parse_args()


def parse_color_tuple(s: str):
    # Mapping keys look like "(197, 244, 45, 255)".
    value = ast.literal_eval(s)
    if not isinstance(value, tuple):
        raise ValueError(f"Unexpected mapping color: {s}")
    return tuple(int(v) for v in value)


def load_mask_rgba(path: Path):
    arr = np.asarray(Image.open(path).convert("RGBA"))
    return arr


def instance_colors_for_class(mapping_path: Path, class_name: str):
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    colors = []
    for color_text, semantic in mapping.items():
        if semantic.get("class") == class_name:
            rgba = parse_color_tuple(color_text)
            if len(rgba) == 3:
                rgba = rgba + (255,)
            colors.append(rgba)

    return colors


def largest_valid_contour(binary_mask, min_area: int):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_yolo(contour, width, height, epsilon_factor):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)

    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None

    coords = []
    for x, y in pts:
        coords.extend(
            [
                float(np.clip(x / width, 0.0, 1.0)),
                float(np.clip(y / height, 0.0, 1.0)),
            ]
        )
    return coords


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(input_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb_*.png files found in {input_dir}")

    total_instances = 0
    frames_with_labels = 0

    for rgb_path in rgb_files:
        frame_id = rgb_path.stem.replace("rgb_", "")

        mask_path = input_dir / f"instance_segmentation_{frame_id}.png"
        mapping_path = input_dir / f"instance_segmentation_semantics_mapping_{frame_id}.json"

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        if not mapping_path.exists():
            raise FileNotFoundError(mapping_path)

        mask = load_mask_rgba(mask_path)
        h, w = mask.shape[:2]

        colors = instance_colors_for_class(mapping_path, args.class_name)

        yolo_rows = []
        for rgba in colors:
            target = np.array(rgba, dtype=np.uint8)
            binary = np.all(mask == target, axis=2)

            contour = largest_valid_contour(binary, args.min_area)
            if contour is None:
                continue

            coords = contour_to_yolo(contour, w, h, args.epsilon)
            if coords is None:
                continue

            row = str(args.class_id) + " " + " ".join(f"{v:.6f}" for v in coords)
            yolo_rows.append(row)

        shutil.copy2(rgb_path, images_dir / rgb_path.name)

        label_path = labels_dir / f"rgb_{frame_id}.txt"
        label_path.write_text(
            "\n".join(yolo_rows) + ("\n" if yolo_rows else ""),
            encoding="utf-8",
        )

        total_instances += len(yolo_rows)
        if yolo_rows:
            frames_with_labels += 1

        print(
            f"[YOLO] frame={frame_id} "
            f"mapped_box_colors={len(colors)} "
            f"labels_written={len(yolo_rows)}"
        )

    print("\n[YOLO] Conversion complete")
    print(f"[YOLO] Frames: {len(rgb_files)}")
    print(f"[YOLO] Frames with labels: {frames_with_labels}")
    print(f"[YOLO] Box instances: {total_instances}")
    print(f"[YOLO] Output: {output_dir}")


if __name__ == "__main__":
    main()
