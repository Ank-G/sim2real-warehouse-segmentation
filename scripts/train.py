from pathlib import Path
import argparse
import sys

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_string: str) -> Path:
    """
    Resolve paths relative to the repository root.
    """
    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_config(config: dict, config_path: Path):
    required_sections = [
        "experiment",
        "model",
        "data",
        "training",
        "output",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Missing '{section}' section in {config_path}"
            )

    dataset_yaml = resolve_project_path(
        config["data"]["dataset"]
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML does not exist:\n{dataset_yaml}"
        )

    return dataset_yaml


def get_model_path(config: dict):
    """
    Models A/B:
        architecture: yolo26n-seg.pt

    Model C:
        initialization: results/model_b_randomized/weights/best.pt
    """

    model_cfg = config["model"]

    if "architecture" in model_cfg:
        # Pretrained Ultralytics model name
        return model_cfg["architecture"]

    if "initialization" in model_cfg:
        checkpoint = resolve_project_path(
            model_cfg["initialization"]
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "\nModel initialization checkpoint not found:\n"
                f"{checkpoint}\n\n"
                "For Model C, Model B must be trained first."
            )

        return str(checkpoint)

    raise KeyError(
        "Model config must contain either "
        "'architecture' or 'initialization'."
    )


def print_experiment_summary(
    config: dict,
    config_path: Path,
    dataset_yaml: Path,
    model_path: str,
):
    training = config["training"]

    print("=" * 70)
    print("SIM2REAL INSTANCE SEGMENTATION TRAINING")
    print("=" * 70)

    print(f"Config:       {config_path}")
    print(f"Experiment:   {config['experiment']['name']}")
    print(f"Model:        {model_path}")
    print(f"Dataset:      {dataset_yaml}")

    print()
    print("TRAINING")
    print("-" * 70)

    print(f"Epochs:       {training.get('epochs')}")
    print(f"Image size:   {training.get('imgsz')}")
    print(f"Batch size:   {training.get('batch')}")
    print(f"Patience:     {training.get('patience')}")
    print(f"Workers:      {training.get('workers')}")
    print(f"Seed:         {training.get('seed')}")

    print("=" * 70)


def train(config_path: Path):
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config does not exist:\n{config_path}"
        )

    config = load_yaml(config_path)

    dataset_yaml = validate_config(
        config,
        config_path,
    )

    model_path = get_model_path(config)

    print_experiment_summary(
        config,
        config_path,
        dataset_yaml,
        model_path,
    )

    model = YOLO(model_path)

    training_cfg = config["training"]
    output_cfg = config["output"]

    output_project = resolve_project_path(
        output_cfg["project"]
    )

    train_args = {
        "data": str(dataset_yaml),
        "epochs": training_cfg.get("epochs", 100),
        "imgsz": training_cfg.get("imgsz", 640),
        "batch": training_cfg.get("batch", 16),
        "patience": training_cfg.get("patience", 20),
        "workers": training_cfg.get("workers", 8),
        "seed": training_cfg.get("seed", 42),
        "optimizer": training_cfg.get("optimizer", "auto"),
        "cache": training_cfg.get("cache", False),
        "amp": training_cfg.get("amp", True),
        "project": str(output_project),
        "name": output_cfg["name"],

        # Ensures an existing experiment is not silently overwritten.
        "exist_ok": False,
    }

    # Only include this if specified.
    if "pretrained" in training_cfg:
        train_args["pretrained"] = training_cfg["pretrained"]

    print("\nStarting training...\n")

    results = model.train(**train_args)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Results directory:\n"
        f"{output_project / output_cfg['name']}"
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO26 instance-segmentation model "
            "using a project experiment YAML."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        config_path = resolve_project_path(args.config)
        train(config_path)

    except Exception as error:
        print("\nERROR")
        print("-" * 70)
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()