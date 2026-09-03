# Sim2Real Warehouse Package Instance Segmentation

A compact Sim2Real experiment for warehouse package instance segmentation using **Isaac Sim** synthetic data and **YOLO26n-seg**.

The project compares three training strategies:

- **Model A — Baseline synthetic:** trained only on a simple synthetic warehouse distribution.
- **Model B — Domain-randomized synthetic:** trained only on a more strongly randomized synthetic distribution.
- **Model C — B + 20 real images:** Model B fine-tuned on only 20 real Package-Seg training images.

The main finding is that **adding a very small amount of real data produced a large improvement.**

| Synthetic data | Instance segmentation masks | 
|:---:|:---:|
| <img width="512" alt="rgb_0000" src="https://github.com/user-attachments/assets/2a98c435-6857-4ef9-9a83-bd039bfebe26" /> | <img width="512" alt="instance_segmentation_0000" src="https://github.com/user-attachments/assets/5808a3d1-cc33-46a1-b1cc-7d3456099b37"/> |

<table>
  <tr>
    <th colspan="2">Real images test set</th>
  </tr>
  <tr>
    <td width="30%"><img alt="frame_5106" src="https://github.com/user-attachments/assets/8cad45cd-7e9e-443c-bb21-de02a6e148b2" /></td>
    <td width="30%"><img alt="572_zl20230718" src="https://github.com/user-attachments/assets/c38e881d-b119-425a-94c5-5d08dda79730" /></td>
  </tr>
</table>

## Motivation

Synthetic data is attractive in robotics because it is cheap to scale, automatically labeled, and easy to vary. The challenge is the **Sim2Real gap**: a model can perform very well in simulation while failing on real images.

This project explores three questions:

1. How well does a model trained on simple synthetic warehouse images transfer to real package images?
2. Does stronger domain randomization improve zero-shot transfer?
3. How much can a small amount of real data improve a synthetic-pretrained model?

---

## Experimental Design

```text
                         Isaac Sim
                            |
             +--------------+--------------+
             |                             |
             v                             v
      Baseline synthetic           Domain-randomized synthetic
             |                             |
             v                             v
          Model A                       Model B
                                           |
                                  + 20 real train images
                                           |
                                           v
                                        Model C

                 A, B and C evaluated on the same
                    official Package-Seg test split
```

### Model A — Baseline Synthetic

Model A uses a controlled synthetic warehouse setup with:

- warehouse environment from Isaac Sim
- cardboard-box assets
- natural box proportions
- limited position jitter
- limited yaw variation
- fixed lighting
- restricted camera viewpoint
- no aggressive appearance randomization

This serves as the synthetic-only baseline.

### Model B — Strong Domain Randomization

Model B uses a wider synthetic distribution, including:

- multiple cardboard-box assets
- multiple warehouse regions
- wider camera viewpoint variation
- lighting variation
- object-position variation
- random yaw
- moderate isotropic scale variation
- clustered layouts
- stacked layouts
- floor/material appearance variation

The goal was to expose the model to a broader range of synthetic conditions while keeping the task and model architecture unchanged.

### Model C — Synthetic Pretraining + 20 Real Images

Model C starts from the corrected Model B checkpoint and is fine-tuned on **exactly 20 real Package-Seg training images**.

The official Package-Seg validation split is used for checkpoint selection during fine-tuning.

---

## Real Dataset

The real target domain is **Package-Seg**, formatted for Ultralytics instance segmentation.

Dataset sizes used in this project:

| Split | Images |
|---|---:|
| Train | 1920 |
| Validation | 188 |
| Test | 89 |

Only **20 images from the official training split** are used for Model C fine-tuning.

The test split is not used for hyperparameter tuning.

---

## Synthetic Dataset

For each synthetic strategy:

- 500 synthetic images are generated
- 450 images are used for training
- 50 images are used for synthetic validation

The target package-count distribution was chosen to approximately match the real Package-Seg training distribution.

For the final corrected randomized dataset:

```text
Requested packages/image: 3.448
Visible YOLO instances/image: 2.95
Frames with labels > requested objects: 0
Maximum visible labels/image: 13
```

The lower visible count is expected because some generated boxes are occluded or outside the camera field of view.

---

## Annotation Pipeline Debugging

One of the most useful parts of the project was discovering and fixing a synthetic-labeling failure.

An early version of the instance-segmentation conversion pipeline occasionally produced impossible labels such as:

```text
5 requested boxes -> 118 YOLO instances
```

The cause was not the package-count sampler itself. The Isaac Sim warehouse already contained semantically labeled cardboard-box geometry, and the instance-segmentation annotator also returned those objects when the camera viewed certain warehouse regions.

The final converter therefore exports only instances belonging to the generated package hierarchy:

```text
/SDG_V2/Boxes/...
```

All child segmentation colors belonging to one logical generated `Box_XX` object are merged into a single package instance before conversion to YOLO segmentation format.

After the fix:

```text
Requested total packages: 1724
YOLO visible instances:    1475
Frames labels > requested: 0
```

The bug-affected B/C runs are not used as final experimental results.

---

## Training

All three models use **YOLO26n-seg**.

### Synthetic-only models

Common settings for A and B:

```text
epochs: 100
image size: 640
batch size: 16
seed: 42
pretrained: true
optimizer: auto
AMP: enabled
```

A and B use the same training configuration so that the main treatment difference is the synthetic-data distribution.

### Model C fine-tuning

```text
initialization: Model B best checkpoint
real training images: 20
epochs: 50
image size: 640
batch size: 4
optimizer: AdamW
learning rate: 1e-4
freeze: first 10 layers
patience: 15
seed: 42
```

---

## Final Real-Test Results

All reported values below are measured on the same **89-image Package-Seg test split**.

| Model | Box Precision | Box Recall | Box mAP50 | Box mAP50-95 | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A — Baseline Synthetic** | 0.0666 | 0.1785 | 0.0264 | 0.0126 | 0.0616 | 0.1569 | 0.0228 | 0.0113 |
| **B — Domain Randomized** | 0.0237 | 0.0215 | 0.0057 | 0.0048 | 0.0203 | 0.0185 | 0.0053 | 0.0021 |
| **C — B + 20 Real** | **0.2746** | **0.3051** | **0.1720** | **0.1065** | **0.2713** | **0.2862** | **0.1633** | **0.0963** |

### Main result

Model C clearly performs best on real data.

Compared with Model B:

- mask mAP50-95 improves from **0.0021 -> 0.0963**
- mask mAP50 improves from **0.0053 -> 0.1633**

Compared with Model A:

- mask mAP50-95 is about **8.5x higher**
- mask mAP50 is about **7.2x higher**

---

## Why Did Model B Perform Worse Than Model A?

The important observation is that Model B learned its own synthetic domain very well, but did not transfer well to Package-Seg.

This suggests a **distribution-alignment problem**, not simply a training failure.

Stronger domain randomization increased variation in:

- viewpoint
- lighting
- materials
- layout
- stacking
- object pose
- warehouse location

However, broader synthetic variation is not automatically closer to the real target distribution.

Possible reasons include:

- some randomized viewpoints or layouts may be poorly represented in Package-Seg
- aggressive appearance variation may suppress real package cues such as cardboard texture, tape, printing, edges, and shadows
- stacking and cluster distributions may differ from the real dataset
- the same 450-image training budget must cover a much larger synthetic state space
- synthetic validation only measures performance inside the generated distribution

The key lesson is:

> **Domain randomization should cover relevant real-world variation, not simply maximize variation.**

The very high synthetic validation performance of Model B together with its poor real-test result also demonstrates that **high synthetic mAP is not evidence of successful Sim2Real transfer**.

---

## What Model C Shows

Fine-tuning the domain-randomized model on only 20 real images produces a large improvement.

This suggests that a small amount of target-domain information can strongly help adaptation when a model has first learned the basic package-segmentation task in simulation.

A careful interpretation is important: this experiment shows that **B + 20 real images works much better than B alone**, but it does not prove that B is the best possible initialization for 20-image fine-tuning because this project does not include an `A + 20 real` or `pretrained YOLO + 20 real` control.

---

## Repository Structure

```text
sim2real-warehouse-segmentation/
|
+-- configs/
|   +-- baseline.yaml
|   +-- randomized.yaml
|   +-- finetune.yaml
|   +-- evaluation.yaml
|   +-- datasets/
|
+-- isaac/
|   +-- generate_warehouse_synthetic.py
|   +-- isaac_instance_to_yolo.py
|
+-- scripts/
|   +-- train.py
|   +-- evaluate_v2_test.py
|   +-- split_synthetic_dataset.py
|   +-- analyze_dataset_geometry.py
|   +-- report_dataset_distribution.py
|
+-- data/
|   +-- real/
|   +-- synthetic/
|
+-- results/
|
+-- README.md
```

Large datasets and model artifacts can be kept outside Git and linked locally as needed.

---

## Reproducing the Experiment

### 1. Install training dependencies

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python pillow numpy pyyaml
```

Synthetic generation requires a working Isaac Sim Python environment.

### 2. Generate synthetic data

Baseline:

```bash
/opt/IsaacSim/python.sh \
  isaac/generate_warehouse_synthetic.py \
  --mode baseline \
  --frames 500 \
  --seed 42 \
  --output /path/to/baseline_500
```

Domain-randomized:

```bash
/opt/IsaacSim/python.sh \
  isaac/generate_warehouse_synthetic.py \
  --mode randomized \
  --frames 500 \
  --seed 42 \
  --output /path/to/randomized_500
```

### 3. Convert Isaac masks to YOLO segmentation labels

```bash
python isaac/isaac_instance_to_yolo.py \
  --input /path/to/randomized_500 \
  --output /path/to/randomized_500_yolo \
  --class-id 0 \
  --instance-root /SDG_V2/Boxes
```

### 4. Split synthetic data

```bash
python scripts/split_synthetic_dataset.py \
  --input /path/to/randomized_500_yolo \
  --output /path/to/randomized_split \
  --val-count 50 \
  --seed 42
```

### 5. Train Model A and Model B

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/randomized.yaml
```

### 6. Fine-tune Model C

Initialize from Model B's best checkpoint and train on the 20-image real subset using the settings described above.

### 7. Evaluate A/B/C

```bash
python scripts/evaluate_v2_test.py
```

Final metrics are written to:

```text
results/final_test_v2/final_test_metrics.csv
```

---

## Limitations

This is a deliberately small, interview-scale Sim2Real study rather than a production benchmark.

Important limitations:

- only 500 synthetic images per synthetic strategy
- only one model size, YOLO26n-seg
- Model C uses only 20 real images
- no `A + 20 real` control
- no `pretrained YOLO + 20 real` control
- material randomization is relatively simple and does not model every real package texture
- synthetic warehouse geometry and camera statistics are not guaranteed to match Package-Seg
- the official test split was evaluated during an earlier bug-affected run before the annotation issue was discovered; the corrected results were rerun after fixing the data pipeline, with no test-driven hyperparameter tuning

---

## Next Steps

Useful follow-up experiments would be:

- train **A + 20 real** as a control
- train **pretrained YOLO + 20 real** as a control
- increase synthetic dataset size
- make domain randomization target-aware using measured real-data statistics
- add more realistic package textures, tape, labels, and printed surfaces
- compare randomization strengths rather than using only baseline vs strong DR
- perform failure mining on real deployment images and iteratively expand the training distribution

---

## Takeaway

This project produced three practical Sim2Real lessons:

1. **Synthetic validation performance alone is not enough.** A model can perform extremely well in simulation and still fail on real images.
2. **More domain randomization is not automatically better.** Randomization must overlap with the real target distribution.
3. **A small amount of real data can be extremely valuable.** Fine-tuning with only 20 real images produced the strongest model by a large margin.

The final experiment therefore supports a practical robotics workflow:

```text
simulation
    -> diverse synthetic pretraining
    -> real-world validation
    -> small real-data adaptation
    -> failure mining
    -> dataset improvement
    -> retraining
```

