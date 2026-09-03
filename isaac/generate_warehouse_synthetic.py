#!/usr/bin/env python3
"""Isaac Sim 6.0 warehouse SDG for the V2 Sim2Real experiment.

Model A (baseline): one warehouse region, no stacking/material DR, small pose jitter.
Model B (randomized): multiple warehouse regions, spread/cluster/stacked layouts,
material/floor appearance randomization, wider camera/light/pose variation.

A and B still share the same composition RNG for package count, camera-distance
bucket/radius, and active box identities.
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
    launch_config={"headless": True, "renderer": "RealTimePathTracing"}
)

# Isaac/Omniverse imports must come after SimulationApp.
import carb.settings
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom

from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
from isaacsim.core.experimental.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path


ENV_URL = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
BOX_URLS = [
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_02.usd",
]

RESOLUTION = (512, 512)
MAX_BOXES = 16
BASE_TARGET = np.array([0.0, 0.0, 0.25], dtype=float)
BASE_CAMERA = np.array([3.2, 4.0, 1.6], dtype=float)
BASE_AZIMUTH_RAD = math.atan2(BASE_CAMERA[1], BASE_CAMERA[0])
HIDDEN_POSITION = (1000.0, 1000.0, -100.0)

# B-only scene anchors: moderate offsets so the camera sees different warehouse
# sections without moving aggressively to unknown parts of the scene.
WAREHOUSE_ANCHORS = np.array(
    [
        [0.0, 0.0],
        [2.4, 0.0],
        [-2.4, 0.0],
        [0.0, 2.4],
        [0.0, -2.4],
        [1.7, 1.7],
        [-1.7, -1.7],
    ],
    dtype=float,
)

GRID_COORDS = (-1.275, -0.425, 0.425, 1.275)
GRID_OFFSETS = np.array([(x, y) for y in GRID_COORDS for x in GRID_COORDS], dtype=float)
CLUSTER_COORDS = (-0.65, 0.0, 0.65)
CLUSTER_OFFSETS = np.array(
    [(x, y) for y in CLUSTER_COORDS for x in CLUSTER_COORDS], dtype=float
)

# Package-Seg TRAIN empirical package-count distribution.
COUNT_VALUES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16], dtype=int)
COUNT_PROBS = np.array(
    [
        117 / 1920,
        149 / 1920,
        418 / 1920,
        368 / 1920,
        370 / 1920,
        187 / 1920,
        165 / 1920,
        93 / 1920,
        33 / 1920,
        8 / 1920,
        6 / 1920,
        3 / 1920,
        3 / 1920,
    ],
    dtype=float,
)
COUNT_PROBS /= COUNT_PROBS.sum()

CAMERA_DISTANCE_BUCKETS = (
    ("close", 0.25, 1.20, 1.90),
    ("medium", 0.50, 1.90, 3.00),
    ("wide", 0.25, 3.00, 4.80),
)


def sample_package_count(rng):
    return int(rng.choice(COUNT_VALUES, p=COUNT_PROBS))


def sample_camera_distance(rng):
    names = [x[0] for x in CAMERA_DISTANCE_BUCKETS]
    probs = np.array([x[1] for x in CAMERA_DISTANCE_BUCKETS], dtype=float)
    probs /= probs.sum()
    chosen = str(rng.choice(names, p=probs))
    for name, _, lo, hi in CAMERA_DISTANCE_BUCKETS:
        if name == chosen:
            return name, float(rng.uniform(lo, hi))
    raise RuntimeError("Could not sample camera distance")


def make_camera_position(target, radius, mode, rng):
    if mode == "baseline":
        azimuth = BASE_AZIMUTH_RAD
        z = 1.45
    else:
        azimuth = BASE_AZIMUTH_RAD + math.radians(float(rng.uniform(-70.0, 70.0)))
        z = float(rng.uniform(0.95, 2.25))
    return np.array(
        [
            target[0] + radius * math.cos(azimuth),
            target[1] + radius * math.sin(azimuth),
            z,
        ],
        dtype=float,
    )


def get_local_z_extents(stage, prim_path):
    """Local bbox min/max Z for robust stacking of different box assets."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0.0, 0.50
    try:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        bound = cache.ComputeLocalBound(prim)
        r = bound.GetRange()
        mn, mx = float(r.GetMin()[2]), float(r.GetMax()[2])
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            raise ValueError("invalid bbox")
        return mn, mx
    except Exception:
        return 0.0, 0.50


def collect_mesh_paths(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    paths = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            paths.append(str(prim.GetPath()))
    return paths


def randomized_layout(rng, n):
    """Return layout name, XY offsets, stack IDs, stack layers."""
    if n == 0:
        return "empty", [], [], []

    mode = str(rng.choice(["spread", "cluster", "stacked"], p=[0.30, 0.35, 0.35]))

    if mode == "spread":
        ids = rng.choice(len(GRID_OFFSETS), size=n, replace=False)
        offsets = [GRID_OFFSETS[int(i)].copy() for i in ids]
        return mode, offsets, list(range(n)), [0] * n

    if mode == "cluster":
        order = rng.permutation(len(CLUSTER_OFFSETS)).tolist()
        offsets = []
        for i in range(n):
            base = CLUSTER_OFFSETS[order[i % len(order)]].copy()
            if i >= len(CLUSTER_OFFSETS):
                base += rng.uniform(-0.20, 0.20, size=2)
            offsets.append(base)
        return mode, offsets, list(range(n)), [0] * n

    # stacked: about 2 boxes/base, at most 3 layers/base
    n_bases = max(1, min(len(CLUSTER_OFFSETS), int(math.ceil(n / 2.2))))
    chosen = rng.choice(len(CLUSTER_OFFSETS), size=n_bases, replace=False)
    bases = [CLUSTER_OFFSETS[int(i)].copy() for i in chosen]
    heights = [0] * n_bases
    offsets, stack_ids, layers = [], [], []

    for _ in range(n):
        candidates = [j for j, h in enumerate(heights) if h < 3]
        if not candidates:
            candidates = list(range(n_bases))
        min_h = min(heights[j] for j in candidates)
        lowest = [j for j in candidates if heights[j] == min_h]
        sid = int(rng.choice(lowest))
        offsets.append(bases[sid].copy())
        stack_ids.append(sid)
        layers.append(int(heights[sid]))
        heights[sid] += 1

    return mode, offsets, stack_ids, layers


def main():
    composition_seed = args.seed
    variation_seed = args.seed + (1000 if args.mode == "baseline" else 2000)
    composition_rng = np.random.default_rng(composition_seed)
    variation_rng = np.random.default_rng(variation_seed)

    out_dir = args.output or f"/home/ubuntu/sim2real_synthetic_v2/{args.mode}"
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Could not resolve Isaac Sim assets root")

    print(f"[SDG-V2] Loading {assets_root + ENV_URL}")
    opened, _ = open_stage(assets_root + ENV_URL)
    if not opened:
        raise RuntimeError("Could not open warehouse stage")

    for _ in range(8):
        simulation_app.update()

    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
    stage = omni.usd.get_context().get_stage()

    for prim in stage.Traverse():
        remove_all_labels(prim, include_descendants=True)

    rep.functional.create.xform(name="SDG_V2")
    rep.functional.create.scope(name="Boxes", parent="/SDG_V2")
    dome_light = rep.functional.create.dome_light(
        intensity=180, parent="/SDG_V2", name="DomainLight"
    )

    boxes, box_asset_records = [], []
    for i in range(MAX_BOXES):
        rel = BOX_URLS[i % len(BOX_URLS)]
        # Create the referenced asset WITHOUT applying semantics yet.
        # Some Isaac warehouse box USDs contain semantic labels on internal
        # child prims. If those remain, instance segmentation treats the
        # labelled children as separate instances. We clear all inherited
        # labels after the references have loaded, then label only this
        # logical box root.
        box = rep.functional.create.reference(
            usd_path=assets_root + rel,
            parent="/SDG_V2/Boxes",
            name=f"Box_{i:02d}",
        )
        boxes.append(box)
        box_asset_records.append(rel)
        rep.functional.modify.pose(
            box,
            position_value=HIDDEN_POSITION,
            rotation_value=(0.0, 0.0, 0.0),
            scale_value=(1.0, 1.0, 1.0),
            write_to_usd=True,
        )

    for _ in range(10):
        simulation_app.update()

    # ------------------------------------------------------------------
    # IMPORTANT: semantic cleanup for referenced box assets
    # ------------------------------------------------------------------
    # Replicator instance segmentation resolves instances at the lowest
    # semantically-labelled prim in the hierarchy. The warehouse box USDs
    # can carry labels on internal child meshes, which can turn one logical
    # cardboard box into dozens of segmentation IDs. Remove ALL semantics
    # recursively from each referenced asset and then apply exactly one
    # class label to its logical root.
    for i in range(MAX_BOXES):
        root_path = f"/SDG_V2/Boxes/Box_{i:02d}"
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Invalid box root prim: {root_path}")

        remove_all_labels(
            root_prim,
            remove_taxonomies=True,
            include_descendants=True,
        )
        add_labels(
            root_prim,
            labels=["box"],
            taxonomy="class",
        )

    # Give USD/Replicator a few updates to rebuild the semantic hierarchy.
    for _ in range(5):
        simulation_app.update()

    print("[SDG-V2] Cleared descendant semantics; one class=box label per logical box root")

    box_z_extents = [
        get_local_z_extents(stage, f"/SDG_V2/Boxes/Box_{i:02d}")
        for i in range(MAX_BOXES)
    ]

    # B-only material/background DR. NVIDIA Replicator supports material
    # randomization via material_omnipbr + randomizer.materials.
    box_mesh_paths = []
    if args.mode == "randomized":
        for i in range(MAX_BOXES):
            box_mesh_paths.extend(
                collect_mesh_paths(stage, f"/SDG_V2/Boxes/Box_{i:02d}")
            )

        floor_overlay = rep.create.plane(
            position=(0.0, 0.0, 0.003),
            scale=(12.0, 12.0, 1.0),
            name="RandomizedFloorOverlay",
        )

        floor_palette = [
            (0.18, 0.18, 0.18),
            (0.28, 0.28, 0.28),
            (0.38, 0.38, 0.36),
            (0.46, 0.44, 0.40),
            (0.30, 0.32, 0.34),
        ]
        floor_materials = rep.create.material_omnipbr(
            diffuse=rep.distribution.choice(floor_palette),
            roughness=rep.distribution.uniform(0.45, 0.95),
            count=8,
        )

        box_palette = [
            (0.42, 0.25, 0.10),
            (0.55, 0.34, 0.16),
            (0.68, 0.47, 0.25),
            (0.75, 0.60, 0.40),
            (0.82, 0.72, 0.55),
            (0.58, 0.52, 0.42),
        ]
        box_materials = rep.create.material_omnipbr(
            diffuse=rep.distribution.choice(box_palette),
            roughness=rep.distribution.uniform(0.35, 0.90),
            count=14,
        )

        with rep.trigger.on_frame():
            with floor_overlay:
                rep.randomizer.materials(floor_materials)
            if box_mesh_paths:
                with rep.get.prim_at_path(box_mesh_paths):
                    rep.randomizer.materials(box_materials)

        print(
            f"[SDG-V2] Strong B randomization enabled; "
            f"{len(box_mesh_paths)} box mesh prims found"
        )

    camera = rep.functional.create.camera(
        position=tuple(BASE_CAMERA),
        look_at=tuple(BASE_TARGET),
        focal_length=30.0,
        clipping_range=(0.1, 10000.0),
        parent="/SDG_V2",
        name="SyntheticCamera",
    )
    render_product = rep.create.render_product(camera, RESOLUTION, name="WarehouseViewV2")

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

    for _ in range(10):
        simulation_app.update()

    frame_records = []

    for frame in range(args.frames):
        # Shared A/B composition.
        n_boxes = sample_package_count(composition_rng)
        camera_bucket, camera_radius = sample_camera_distance(composition_rng)
        if n_boxes:
            active_box_indices = composition_rng.choice(
                len(boxes), size=n_boxes, replace=False
            )
        else:
            active_box_indices = np.asarray([], dtype=int)

        # Scene region + layout.
        if args.mode == "baseline":
            anchor_xy = np.array([0.0, 0.0], dtype=float)
            layout_mode = "baseline_grid"
            if n_boxes:
                grid_ids = composition_rng.choice(
                    len(GRID_OFFSETS), size=n_boxes, replace=False
                )
                local_offsets = [GRID_OFFSETS[int(i)].copy() for i in grid_ids]
            else:
                local_offsets = []
            stack_ids, stack_layers = list(range(n_boxes)), [0] * n_boxes
        else:
            anchor_xy = WAREHOUSE_ANCHORS[
                int(variation_rng.integers(0, len(WAREHOUSE_ANCHORS)))
            ].copy()
            layout_mode, local_offsets, stack_ids, stack_layers = randomized_layout(
                variation_rng, n_boxes
            )

        target_z = 0.55 if args.mode == "randomized" and layout_mode == "stacked" else 0.25
        target = np.array([anchor_xy[0], anchor_xy[1], target_z], dtype=float)
        camera_pos = make_camera_position(
            target, camera_radius, args.mode, variation_rng
        )
        rep.functional.modify.pose(
            camera,
            position_value=tuple(float(x) for x in camera_pos),
            look_at_value=tuple(float(x) for x in target),
            look_at_up_axis=(0, 0, 1),
            write_to_usd=True,
        )

        if args.mode == "baseline":
            light_intensity = 180.0
            xy_jitter = 0.08
            yaw_min, yaw_max = -10.0, 10.0
            scale_min, scale_max = 1.0, 1.0
        else:
            light_intensity = float(variation_rng.uniform(70.0, 420.0))
            xy_jitter = 0.22
            yaw_min, yaw_max = -180.0, 180.0
            scale_min, scale_max = 0.88, 1.12

        rep.functional.modify.attribute(
            dome_light, "inputs:intensity", light_intensity
        )

        for box in boxes:
            rep.functional.modify.pose(
                box,
                position_value=HIDDEN_POSITION,
                rotation_value=(0.0, 0.0, 0.0),
                scale_value=(1.0, 1.0, 1.0),
                write_to_usd=True,
            )

        stack_top = {}
        positions, scales, yaws, assets = [], [], [], []

        for j, raw_idx in enumerate(active_box_indices):
            box_idx = int(raw_idx)
            box = boxes[box_idx]
            local_xy = np.asarray(local_offsets[j], dtype=float)
            jitter = variation_rng.uniform(-xy_jitter, xy_jitter, size=2)
            xy = anchor_xy + local_xy + jitter
            yaw = float(variation_rng.uniform(yaw_min, yaw_max))
            scale = float(variation_rng.uniform(scale_min, scale_max))

            min_z, max_z = box_z_extents[box_idx]
            sid = int(stack_ids[j])
            if args.mode == "randomized" and layout_mode == "stacked":
                bottom = float(stack_top.get(sid, 0.02))
            else:
                bottom = 0.02

            pivot_z = float(bottom - min_z * scale)
            next_top = float(pivot_z + max_z * scale + 0.012)
            if args.mode == "randomized" and layout_mode == "stacked":
                stack_top[sid] = next_top

            pos = (float(xy[0]), float(xy[1]), pivot_z)
            rep.functional.modify.pose(
                box,
                position_value=pos,
                rotation_value=(0.0, 0.0, yaw),
                scale_value=(scale, scale, scale),
                write_to_usd=True,
            )

            positions.append(list(pos))
            scales.append(scale)
            yaws.append(yaw)
            assets.append(box_asset_records[box_idx])

        simulation_app.update()

        print(
            f"[SDG-V2] {frame + 1}/{args.frames} | {args.mode} | "
            f"boxes={n_boxes} | layout={layout_mode} | "
            f"anchor=({anchor_xy[0]:.1f},{anchor_xy[1]:.1f}) | "
            f"camera={camera_bucket}:{camera_radius:.2f}m"
        )

        rep.orchestrator.step(rt_subframes=4, wait_for_render=True)

        frame_records.append(
            {
                "frame": frame,
                "num_boxes_requested": n_boxes,
                "camera_bucket": camera_bucket,
                "camera_horizontal_radius_m": camera_radius,
                "warehouse_anchor_xy": [float(x) for x in anchor_xy],
                "layout_mode": layout_mode,
                "camera_target": [float(x) for x in target],
                "camera_position": [float(x) for x in camera_pos],
                "active_box_indices": [int(x) for x in active_box_indices],
                "active_box_assets": assets,
                "stack_ids": [int(x) for x in stack_ids],
                "stack_layers": [int(x) for x in stack_layers],
                "box_positions": positions,
                "box_yaws_deg": yaws,
                "box_scales": scales,
                "light_intensity": light_intensity,
            }
        )

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()

    metadata = {
        "experiment_version": "improved-v2-strong-dr-semantics-fixed",
        "mode": args.mode,
        "frames": args.frames,
        "seed": args.seed,
        "composition_seed": composition_seed,
        "variation_seed": variation_seed,
        "resolution": list(RESOLUTION),
        "environment": ENV_URL,
        "box_assets": BOX_URLS,
        "semantic_class": "box",
        "training_class_id": 0,
        "training_class_name": "package",
        "shared_A_B": [
            "package count distribution",
            "camera distance distribution",
            "active box identities",
            "warehouse USD",
            "focal length",
        ],
        "baseline": {
            "warehouse_regions": 1,
            "layouts": ["grid"],
            "stacking": False,
            "box_material_randomization": False,
            "floor_material_randomization": False,
            "camera_azimuth_jitter_deg": 0,
            "camera_height_m": 1.45,
            "light_intensity": 180,
            "xy_jitter_m": 0.08,
            "yaw_deg": [-10, 10],
            "scale": 1.0,
        },
        "randomized": {
            "warehouse_anchors": WAREHOUSE_ANCHORS.tolist(),
            "layouts": ["spread", "cluster", "stacked"],
            "stacking": "up to 3 layers; bbox-aware",
            "box_material_randomization": True,
            "floor_material_randomization": True,
            "camera_azimuth_jitter_deg": [-70, 70],
            "camera_height_m": [0.95, 2.25],
            "light_intensity": [70, 420],
            "xy_jitter_m": 0.22,
            "yaw_deg": [-180, 180],
            "isotropic_scale": [0.88, 1.12],
        },
        "frame_records": frame_records,
    }

    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[SDG-V2] Complete")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        print("\n[SDG-V2] FATAL ERROR")
        traceback.print_exc()
        raise
    finally:
        if simulation_app.is_running():
            simulation_app.close()
