#!/usr/bin/env python3
"""
Print and save detailed distribution statistics for synthetic vs real
YOLO instance-segmentation labels.

Reports:
- instances/image percentiles
- bbox-area percentiles
- aspect-ratio percentiles
- exact package-count frequency
- background/empty-image fraction

Outputs:
- distribution_percentiles.csv
- count_distribution.csv

Example:
python scripts/report_dataset_distribution.py \
  --synthetic-labels data/synthetic/randomized/train/labels \
  --real-labels data/real/package_seg/labels/train \
  --output-dir results/data_analysis_old
"""

from pathlib import Path
from collections import Counter
import argparse
import csv
import numpy as np


COUNT_PERCENTILES = [0, 5, 25, 50, 75, 90, 95, 99, 100]
AREA_PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]
ASPECT_PERCENTILES = [5, 25, 50, 75, 95]


def read_stats(label_dir):
    counts = []
    areas = []
    aspect_ratios = []

    files = sorted(Path(label_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt labels found in {label_dir}")

    for f in files:
        n = 0

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

            width = max(float(xs.max() - xs.min()), 0.0)
            height = max(float(ys.max() - ys.min()), 0.0)

            areas.append(width * height)
            aspect_ratios.append(width / max(height, 1e-8))
            n += 1

        # Empty .txt files contribute a count of 0.
        counts.append(n)

    return (
        np.asarray(counts, dtype=float),
        np.asarray(areas, dtype=float),
        np.asarray(aspect_ratios, dtype=float),
    )


def print_percentiles(name, counts, areas, aspect_ratios):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nINSTANCES / IMAGE")
    for p in COUNT_PERCENTILES:
        print(f"P{p:>3}: {np.percentile(counts, p):.2f}")

    print("\nBBOX AREA")
    for p in AREA_PERCENTILES:
        print(f"P{p:>3}: {np.percentile(areas, p):.5f}")

    print("\nASPECT RATIO")
    for p in ASPECT_PERCENTILES:
        print(f"P{p:>3}: {np.percentile(aspect_ratios, p):.2f}")

    counter = Counter(int(v) for v in counts)

    print("\nEXACT INSTANCE-COUNT DISTRIBUTION")
    for n in sorted(counter):
        pct = 100.0 * counter[n] / len(counts)
        print(
            f"{n:2d} instances : "
            f"{counter[n]:4d} images  "
            f"({pct:5.2f}%)"
        )

    print()
    print("Total images:", len(counts))
    print("Empty images:", counter.get(0, 0))
    print(
        "Empty fraction: "
        f"{100.0 * counter.get(0, 0) / len(counts):.2f}%"
    )


def percentile_rows(source, metric, array, percentiles):
    rows = []
    for p in percentiles:
        rows.append({
            "source": source,
            "metric": metric,
            "percentile": p,
            "value": float(np.percentile(array, p)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-labels", required=True)
    parser.add_argument("--real-labels", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/data_analysis_old",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        "synthetic": read_stats(args.synthetic_labels),
        "real": read_stats(args.real_labels),
    }

    for source, (counts, areas, ratios) in datasets.items():
        print_percentiles(
            source.upper(),
            counts,
            areas,
            ratios,
        )

    rows = []
    for source, (counts, areas, ratios) in datasets.items():
        rows += percentile_rows(
            source,
            "instances_per_image",
            counts,
            COUNT_PERCENTILES,
        )
        rows += percentile_rows(
            source,
            "bbox_area",
            areas,
            AREA_PERCENTILES,
        )
        rows += percentile_rows(
            source,
            "aspect_ratio",
            ratios,
            ASPECT_PERCENTILES,
        )

    with (out / "distribution_percentiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "metric",
                "percentile",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    all_counts = sorted(
        set(
            int(v)
            for counts, _, _ in datasets.values()
            for v in counts
        )
    )

    syn_counter = Counter(
        int(v) for v in datasets["synthetic"][0]
    )
    real_counter = Counter(
        int(v) for v in datasets["real"][0]
    )

    with (out / "count_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "instances_per_image",
            "synthetic_images",
            "synthetic_fraction",
            "real_images",
            "real_fraction",
        ])

        syn_total = len(datasets["synthetic"][0])
        real_total = len(datasets["real"][0])

        for n in all_counts:
            writer.writerow([
                n,
                syn_counter.get(n, 0),
                syn_counter.get(n, 0) / syn_total,
                real_counter.get(n, 0),
                real_counter.get(n, 0) / real_total,
            ])

    print("\nSaved:")
    print(out / "distribution_percentiles.csv")
    print(out / "count_distribution.csv")


if __name__ == "__main__":
    main()
