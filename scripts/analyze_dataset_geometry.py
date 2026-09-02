#!/usr/bin/env python3
"""
Compare geometric statistics of two YOLO instance-segmentation datasets.

Outputs:
- geometry_summary.csv
- bbox_area_histogram.png
- polygon_area_histogram.png
- object_count_histogram.png
- aspect_ratio_histogram.png
- object_center_heatmap_synthetic.png
- object_center_heatmap_real.png

Example:
python scripts/analyze_dataset_geometry.py \
  --synthetic-labels data/synthetic/randomized/train/labels \
  --real-labels data/real/package_seg/labels/train \
  --output-dir results/data_analysis_old
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def polygon_area(xs, ys):
    if len(xs) < 3:
        return 0.0
    return 0.5 * abs(
        np.dot(xs, np.roll(ys, -1))
        - np.dot(ys, np.roll(xs, -1))
    )


def read_dataset(label_dir):
    rows = []
    counts = []

    files = sorted(Path(label_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt labels found in {label_dir}")

    for f in files:
        valid = 0

        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                coords = np.asarray(
                    [float(v) for v in parts[1:]],
                    dtype=float,
                )
            except ValueError:
                continue

            if len(coords) % 2 != 0:
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            if len(xs) < 3:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            w = max(float(x2 - x1), 0.0)
            h = max(float(y2 - y1), 0.0)

            rows.append({
                "polygon_area": polygon_area(xs, ys),
                "bbox_area": w * h,
                "bbox_width": w,
                "bbox_height": h,
                "aspect_ratio": w / max(h, 1e-8),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
            })

            valid += 1

        # Empty .txt files are counted as background images.
        counts.append(valid)

    return rows, np.asarray(counts, dtype=float)


def arr(rows, key):
    return np.asarray([r[key] for r in rows], dtype=float)


def hist(a, b, title, xlabel, path, bins=40):
    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=bins, alpha=0.55, density=True, label="Synthetic")
    plt.hist(b, bins=bins, alpha=0.55, density=True, label="Real")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def heatmap(rows, title, path):
    plt.figure(figsize=(6, 5))
    plt.hist2d(
        arr(rows, "center_x"),
        arr(rows, "center_y"),
        bins=25,
        range=[[0, 1], [0, 1]],
    )
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.xlabel("Normalized center x")
    plt.ylabel("Normalized center y")
    plt.title(title)
    plt.colorbar(label="Instances")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-labels", required=True)
    parser.add_argument("--real-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    syn_rows, syn_counts = read_dataset(args.synthetic_labels)
    real_rows, real_counts = read_dataset(args.real_labels)

    hist(
        arr(syn_rows, "bbox_area"),
        arr(real_rows, "bbox_area"),
        "Bounding-box Area: Synthetic vs Real",
        "Normalized bbox area",
        out / "bbox_area_histogram.png",
    )

    hist(
        arr(syn_rows, "polygon_area"),
        arr(real_rows, "polygon_area"),
        "Polygon Area: Synthetic vs Real",
        "Normalized polygon area",
        out / "polygon_area_histogram.png",
    )

    hist(
        syn_counts,
        real_counts,
        "Instances per Image",
        "Instances / image",
        out / "object_count_histogram.png",
        bins=30,
    )

    syn_ar = arr(syn_rows, "aspect_ratio")
    real_ar = arr(real_rows, "aspect_ratio")
    combined = np.concatenate([syn_ar, real_ar])
    finite = combined[np.isfinite(combined)]
    upper = np.percentile(finite, 99) if len(finite) else 1.0

    hist(
        syn_ar[syn_ar <= upper],
        real_ar[real_ar <= upper],
        "Aspect Ratio",
        "Bounding-box width / height",
        out / "aspect_ratio_histogram.png",
    )

    heatmap(
        syn_rows,
        "Synthetic Object Centers",
        out / "object_center_heatmap_synthetic.png",
    )

    heatmap(
        real_rows,
        "Real Object Centers",
        out / "object_center_heatmap_real.png",
    )

    metrics = {
        "synthetic_images": len(syn_counts),
        "synthetic_instances": len(syn_rows),
        "synthetic_mean_instances_per_image": float(np.mean(syn_counts)),
        "synthetic_median_bbox_area": float(np.median(arr(syn_rows, "bbox_area"))),
        "synthetic_p95_bbox_area": float(np.percentile(arr(syn_rows, "bbox_area"), 95)),
        "real_images": len(real_counts),
        "real_instances": len(real_rows),
        "real_mean_instances_per_image": float(np.mean(real_counts)),
        "real_median_bbox_area": float(np.median(arr(real_rows, "bbox_area"))),
        "real_p95_bbox_area": float(np.percentile(arr(real_rows, "bbox_area"), 95)),
    }

    with (out / "geometry_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print("\nGEOMETRY SUMMARY")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved to:", out)


if __name__ == "__main__":
    main()
