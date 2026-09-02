#!/usr/bin/env python3
"""
Minimal Isaac Sim 6.0 warehouse synthetic-data generator.

Usage:
  ./python.sh generate_warehouse_synthetic.py --mode baseline --frames 5
  ./python.sh generate_warehouse_synthetic.py --mode randomized --frames 5

Outputs RGB + colorized instance-segmentation masks for objects labelled class=box.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "randomized"], required=True)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


args = parse_args()

simulation_app = SimulationApp(
    launch_config={
        "headless": True,
        "renderer": "RealTimePathTracing",
    }
)

# Isaac/Omniverse imports must come after SimulationApp.
import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.utils.semantics import remove_all_labels
from isaacsim.core.experimental.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path


ENV_URL = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
BOX_URL = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd"

RESOLUTION = (512, 512)
NUM_BOXES = 5

# Start near the warehouse origin. If the first smoke-test view is poor,
# these are the only values we need to adjust.
TARGET = np.array([0.0, 0.0, 0.25], dtype=float)
BASE_CAMERA = np.array([3.2, 4.0, 1.6], dtype=float)

BASE_OFFSETS = np.array(
    [
        [-1.00, -0.55],
        [ 0.00, -0.55],
        [ 1.00, -0.55],
        [-0.55,  0.45],
        [ 0.70,  0.50],
    ],
    dtype=float,
)


def main():
    rng = np.random.default_rng(args.seed)

    out_dir = args.output
    if out_dir is None:
        out_dir = f"/home/ubuntu/sim2real_synthetic/{args.mode}"
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Could not resolve Isaac Sim assets root.")

    print(f"[SDG] Loading warehouse: {assets_root + ENV_URL}")
    opened, _ = open_stage(assets_root + ENV_URL)
    if not opened:
        raise RuntimeError("Could not open warehouse stage.")

    # Let referenced assets/materials load.
    for _ in range(5):
        simulation_app.update()
    print("[SDG] Warehouse loaded.")

    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)

    # Remove pre-existing semantic labels from warehouse props so that the
    # instance-segmentation target class contains only our generated boxes.
    stage = omni.usd.get_context().get_stage()
    print("[SDG] Clearing pre-existing warehouse semantics...")
    for prim in stage.Traverse():
        remove_all_labels(prim, include_descendants=True)
    print("[SDG] Warehouse semantics cleared.")

    # Organize generated content separately.
    rep.functional.create.xform(name="SDG")
    rep.functional.create.scope(name="Boxes", parent="/SDG")

    # Supplemental light. Warehouse lights remain part of the environment.
    dome_light = rep.functional.create.dome_light(
        intensity=180,
        parent="/SDG",
        name="DomainLight",
    )

    box_url = assets_root + BOX_URL
    boxes = []
    for i in range(NUM_BOXES):
        box = rep.functional.create.reference(
            usd_path=box_url,
            parent="/SDG/Boxes",
            name=f"Box_{i:02d}",
            semantics={"class": "box"},
        )
        boxes.append(box)

    print(f"[SDG] Created {len(boxes)} labelled box references.")

    # Referenced cardboard-box asset origins are close to their bottom surface.
    # Put them just above z=0 and spread them around the target.
    for i, box in enumerate(boxes):
        pos = (
            float(TARGET[0] + BASE_OFFSETS[i, 0]),
            float(TARGET[1] + BASE_OFFSETS[i, 1]),
            0.02,
        )
        rep.functional.modify.pose(
            box,
            position_value=pos,
            rotation_value=(0.0, 0.0, 0.0),
            scale_value=(1.0, 1.0, 1.0),
            write_to_usd=True,
        )

    camera = rep.functional.create.camera(
        position=tuple(BASE_CAMERA),
        look_at=tuple(TARGET),
        focal_length=30.0,
        clipping_range=(0.1, 10000.0),
        parent="/SDG",
        name="SyntheticCamera",
    )

    render_product = rep.create.render_product(
        camera,
        RESOLUTION,
        name="WarehouseView",
    )

    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir)

    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        instance_segmentation=True,
        colorize_instance_segmentation=True,
        semantic_filter_predicate="class:box",
    )
    writer.attach(render_product)
    print("[SDG] Writer attached.")

    # Let meshes/materials settle before the first capture.
    for _ in range(10):
        simulation_app.update()

    print(f"[SDG] Mode: {args.mode}")
    print(f"[SDG] Frames: {args.frames}")
    print(f"[SDG] Output: {out_dir}")

    for frame in range(args.frames):
        if args.mode == "baseline":
            # Controlled variation: fixed camera/light/scale, small pose jitter.
            rep.functional.modify.pose(
                camera,
                position_value=tuple(BASE_CAMERA),
                look_at_value=tuple(TARGET),
                look_at_up_axis=(0, 0, 1),
                write_to_usd=True,
            )
            rep.functional.modify.attribute(dome_light, "inputs:intensity", 180.0)

            for i, box in enumerate(boxes):
                jitter_xy = rng.uniform(-0.10, 0.10, size=2)
                yaw = float(rng.uniform(-12.0, 12.0))
                pos = (
                    float(TARGET[0] + BASE_OFFSETS[i, 0] + jitter_xy[0]),
                    float(TARGET[1] + BASE_OFFSETS[i, 1] + jitter_xy[1]),
                    0.02,
                )
                rep.functional.modify.pose(
                    box,
                    position_value=pos,
                    rotation_value=(0.0, 0.0, yaw),
                    scale_value=(1.0, 1.0, 1.0),
                    write_to_usd=True,
                )

        else:
            # Domain randomization: more pose variation + scale + camera + light.
            # Conservative camera randomization around the verified baseline view.
            cam_jitter = np.array(
                [
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(-0.20, 0.25),
                ]
            )
            cam_pos = BASE_CAMERA + cam_jitter
            rep.functional.modify.pose(
                camera,
                position_value=tuple(float(x) for x in cam_pos),
                look_at_value=tuple(TARGET),
                look_at_up_axis=(0, 0, 1),
                write_to_usd=True,
            )

            rep.functional.modify.attribute(
                dome_light,
                "inputs:intensity",
                float(rng.uniform(140.0, 280.0)),
            )

            for i, box in enumerate(boxes):
                jitter_xy = rng.uniform(-0.40, 0.40, size=2)
                yaw = float(rng.uniform(-180.0, 180.0))
                scale = float(rng.uniform(0.78, 1.22))
                pos = (
                    float(TARGET[0] + BASE_OFFSETS[i, 0] + jitter_xy[0]),
                    float(TARGET[1] + BASE_OFFSETS[i, 1] + jitter_xy[1]),
                    0.02,
                )
                rep.functional.modify.pose(
                    box,
                    position_value=pos,
                    rotation_value=(0.0, 0.0, yaw),
                    scale_value=(scale, scale, scale),
                    write_to_usd=True,
                )

        simulation_app.update()
        print(f"[SDG] Capture {frame + 1}/{args.frames}")
        rep.orchestrator.step(rt_subframes=4, wait_for_render=True)

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()

    metadata = {
        "mode": args.mode,
        "frames": args.frames,
        "seed": args.seed,
        "resolution": list(RESOLUTION),
        "environment": ENV_URL,
        "box_asset": BOX_URL,
        "class_name": "box",
        "num_boxes": NUM_BOXES,
        "baseline_randomization": {
            "box_xy_jitter_m": 0.10,
            "box_yaw_deg": [-12, 12],
            "camera": "fixed",
            "box_scale": "fixed",
            "supplemental_light_intensity": 180,
        },
        "domain_randomization": {
            "box_xy_jitter_m": 0.40,
            "box_yaw_deg": [-180, 180],
            "box_scale": [0.78, 1.22],
            "camera_xyz_jitter_m": [[-0.35, 0.35], [-0.35, 0.35], [-0.20, 0.25]],
            "supplemental_light_intensity": [140, 280],
        },
    }

    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[SDG] Complete.")
    simulation_app.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        print("\n[SDG] FATAL ERROR:")
        traceback.print_exc()
        raise
    finally:
        if simulation_app.is_running():
            simulation_app.close()
