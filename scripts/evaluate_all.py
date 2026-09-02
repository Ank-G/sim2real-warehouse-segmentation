from pathlib import Path
import argparse
import csv
import json
import sys

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_model(
    model_key,
    model_config,
    dataset_yaml,
    evaluation_config,
    output_root,
):
    weights = resolve_path(model_config["weights"])

    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found for {model_key}:\n"
            f"{weights}"
        )

    model_name = model_config["name"]

    print()
    print("=" * 70)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 70)

    print(f"Weights: {weights}")
    print(f"Dataset: {dataset_yaml}")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=evaluation_config.get("imgsz", 640),
        batch=evaluation_config.get("batch", 16),
        conf=evaluation_config.get("conf", 0.001),
        iou=evaluation_config.get("iou", 0.7),
        plots=True,
        project=str(output_root),
        name=model_name,
        exist_ok=True,
    )

    result = {
        "model": model_key,
        "name": model_name,

        # Detection / bounding-box metrics
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),

        # Instance-segmentation mask metrics
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
    }

    # Timing reported by Ultralytics in ms/image
    speed = getattr(metrics, "speed", {})

    result["preprocess_ms"] = float(
        speed.get("preprocess", 0.0)
    )

    result["inference_ms"] = float(
        speed.get("inference", 0.0)
    )

    result["postprocess_ms"] = float(
        speed.get("postprocess", 0.0)
    )

    return result


def print_results(results):
    print()
    print("=" * 100)
    print("SIM2REAL REAL-WORLD TEST RESULTS")
    print("=" * 100)

    header = (
        f"{'Model':<25}"
        f"{'Mask P':>10}"
        f"{'Mask R':>10}"
        f"{'Mask mAP50':>13}"
        f"{'Mask mAP50-95':>16}"
        f"{'Inference ms':>15}"
    )

    print(header)
    print("-" * 100)

    for result in results:
        print(
            f"{result['name']:<25}"
            f"{result['mask_precision']:>10.4f}"
            f"{result['mask_recall']:>10.4f}"
            f"{result['mask_map50']:>13.4f}"
            f"{result['mask_map50_95']:>16.4f}"
            f"{result['inference_ms']:>15.2f}"
        )


def save_csv(results, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "name",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def save_json(results, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def main(config_path):
    config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found:\n{config_path}"
        )

    config = load_yaml(config_path)

    dataset_yaml = resolve_path(
        config["dataset"]["test"]
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Real test YAML not found:\n{dataset_yaml}"
        )

    evaluation_config = config["evaluation"]

    output_directory = resolve_path(
        config["output"]["directory"]
    )

    csv_output = (
        output_directory
        / config["output"]["filename"]
    )

    json_output = csv_output.with_suffix(".json")

    evaluation_runs = (
        PROJECT_ROOT
        / "results"
        / "evaluation"
    )

    results = []

    for model_key, model_config in config["models"].items():

        result = evaluate_model(
            model_key=model_key,
            model_config=model_config,
            dataset_yaml=dataset_yaml,
            evaluation_config=evaluation_config,
            output_root=evaluation_runs,
        )

        results.append(result)

    print_results(results)

    save_csv(
        results,
        csv_output,
    )

    save_json(
        results,
        json_output,
    )

    print()
    print(f"CSV saved to:\n{csv_output}")
    print()
    print(f"JSON saved to:\n{json_output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Models A, B and C on the "
            "locked real-world test dataset."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Evaluation configuration YAML.",
    )

    args = parser.parse_args()

    try:
        main(args.config)

    except Exception as error:
        print()
        print("ERROR")
        print("-" * 70)
        print(error)
        sys.exit(1)