#!/usr/bin/env python3

from pathlib import Path
import csv

from ultralytics import YOLO


DATA = "configs/datasets/real_finetune.yaml"

MODELS = {
    "Model A - Baseline Synthetic":
        "results/model_a_baseline_v2/weights/best.pt",

    "Model B - Domain Randomized":
        "results/model_b_randomized_v2/weights/best.pt",

    "Model C - B + 20 Real":
        "results/model_c_finetune_v2/weights/best.pt",
}


output_dir = Path("results/final_test_v2")
output_dir.mkdir(parents=True, exist_ok=True)

rows = []


for model_name, model_path in MODELS.items():

    print("\n" + "=" * 80)
    print(model_name)
    print(model_path)
    print("=" * 80)

    model = YOLO(model_path)

    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=16,
        workers=8,
        project=str(output_dir),
        name=model_name
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_"),
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    row = {
        "model": model_name,
        "checkpoint": model_path,

        "box_precision": metrics.box.mp,
        "box_recall": metrics.box.mr,
        "box_map50": metrics.box.map50,
        "box_map50_95": metrics.box.map,

        "mask_precision": metrics.seg.mp,
        "mask_recall": metrics.seg.mr,
        "mask_map50": metrics.seg.map50,
        "mask_map50_95": metrics.seg.map,
    }

    rows.append(row)


csv_path = output_dir / "final_test_metrics.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

for row in rows:

    print(f"\n{row['model']}")

    print(
        f"  Box : "
        f"mAP50={row['box_map50']:.4f}, "
        f"mAP50-95={row['box_map50_95']:.4f}, "
        f"P={row['box_precision']:.4f}, "
        f"R={row['box_recall']:.4f}"
    )

    print(
        f"  Mask: "
        f"mAP50={row['mask_map50']:.4f}, "
        f"mAP50-95={row['mask_map50_95']:.4f}, "
        f"P={row['mask_precision']:.4f}, "
        f"R={row['mask_recall']:.4f}"
    )


print(f"\nSaved: {csv_path}")
