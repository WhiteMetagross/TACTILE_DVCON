# TACTILE — DVCon India 2026 Design Contest: Stage 2A Full Prompt

## Context and Background

You are assisting the TACTILE team (IIIT Allahabad EDA/VLSI Wing) in building the Stage 2A submission for the **DVCon India 2026 Design Contest**. The contest requires a task-aware object detection pipeline deployed at the edge, ultimately targeting the Digilent Genesys-2 FPGA board with the VEGA AS1061 RV64GC soft processor.

### What is TACTILE?

TACTILE stands for **Task-Aware Cascaded Inference with Lightweight Edge Deployment**. It is a hardware-software co-designed edge AI inference system that:
- Accepts a **natural language task query** (one of 14 defined tasks, encoded as a 4-bit integer) and a **camera/dataset image** as paired inputs.
- Outputs the **bounding box and class label of the single most task-appropriate object** in the scene.
- Targets ≥50 FPS at 160×160 resolution within 50,000 LUT slices on Xilinx Kintex-7 XC7K325T, under 5W.

### The Core Problem

Standard object detectors are task-agnostic — given a kitchen scene, they report all detected objects equally. TACTILE must **selectively surface the most task-relevant object**: the wine glass when the task is "serve wine", the knife when the task is "spread butter", the screwdriver when the task is "tighten screw". This must work in real time on resource-constrained hardware.

The contest uses the **COCO dataset** with **14 task categories** as defined in the reference paper:
> Sawatzky et al., "What Object Should I Use? — Task Driven Object Detection", CVPR 2019. arXiv:1904.03000

### The 14 Tasks (from the reference paper / contest definition)

The pipeline must answer all 14 task queries. These tasks map onto COCO object classes, and the system must rank task-relevant objects above task-irrelevant ones:
1. Serve wine
2. Spread butter / jam
3. Drink coffee
4. Set the table
5. Cut vegetables
6. Serve food on a plate
7. Tighten a screw
8. Dig a hole
9. Hang a picture
10. Check the time
11. Make a phone call
12. Take a photo
13. Play music
14. Read a book

---

## Stage 2A Objective

Stage 2A is a **software-only, functionally correct pipeline** running on CPU (not GPU at inference time; GPU may be used for training). The goal is correctness first — speed and hardware acceleration come in Stage 2B/3.

### Deliverables Required (Deadline: 17 May 2026)

1. **Source code** — a functionally correct CPU inference pipeline that:
   - Accepts an image and a task query (from the 14 defined tasks)
   - Outputs the most appropriate object for that task with its bounding box and class label
   - Runs on CPU (Python/PyTorch/ONNX/etc. are all acceptable)

2. **Two-page report** — strictly two pages:
   - **Page 1**: Approach description (pipeline design, model choices, any deviations from Stage 1 architecture)
   - **Page 2**: Results snapshot (visual results on test images across multiple tasks, accuracy metrics if available)

3. **Video demonstration** — short video showing the working application

All three must be zipped into a single file for EasyChair submission under `DC_Stage2_Submission`.

---

## Stage 1 Architecture Reference (MUST READ BEFORE CODING)

The Stage 1 proposal document (`Architectural_guide.md`) defines the full TACTILE architecture that was submitted and signed off. Before writing inference code, thoroughly review it. Key architectural decisions that directly affect the software pipeline:

### Model: YOLOv5n INT8

- Backbone: YOLOv5n (nano variant, 140–217M MACs at 160×160)
- Quantization: INT8 post-training quantization (PTQ)
- Input resolution: **160×160** (primary, for FPS budget)
- Detection output: raw proposals → decoded bboxes + confidence + class_id

### Novel Contribution 1: SAM-IP (Spatial Task Attention Map)

A tiny CNN that generates a **40×40 binary spatial mask** identifying image regions likely to contain task-relevant objects. In software, implement this as:
- 3-layer tiny CNN: Conv3×3 (stride 2, 4ch) → Depthwise Conv3×3 (stride 2, 4ch) → Pointwise Conv1×1 (1ch) → Bilinear upsample to 40×40
- Trained separately with per-task threshold calibration (97% recall target per task)
- **Soft gating** (not hard-drop): if a proposal's center falls in a mask=0 cell, multiply its confidence by `SAM_ALPHA = 0.25`. Do NOT discard the proposal.

### Novel Contribution 2: NMS-IP (Task-Score-Fused NMS)

Replace confidence-only NMS with a **three-way fused score**:

```
fused_score = detection_confidence × task_score × class_task_prior[class_id][task_id]
```

Where:
- `detection_confidence`: YOLOv5n detector confidence
- `task_score`: dot product between 128-D proposal feature and 128-D task embedding
- `class_task_prior[c][t]`: fraction of class-c instances preferred for task t in COCO-Tasks training data (precomputed offline, stored as 80×14 table)

This correctly ranks wine glass over beer glass for "serve wine" task, even when beer glass has higher detector confidence.

### Task Scoring (VEGA software — implement in Python for Stage 2A)

```python
# 75 proposals × 128-D feature × 128-D task embedding = 9,600 MACs
# task_emb: shape [14, 128], INT8
# proposal_features: shape [N, 128], INT8

for i, proposal in enumerate(proposals):
    acc = np.dot(proposal.feature.astype(np.int32), task_emb[task_id].astype(np.int32))
    scores[i] = np.clip(acc >> 14, 0, 255).astype(np.uint8)
```

### Key Worked Example (Wine Glass vs Beer Glass)

For task "serve wine" (task_id=0):

| Object     | conf | task_score | class_prior | fused_score |
|------------|------|------------|-------------|-------------|
| Beer glass | 0.81 | 0.18       | 0.12        | **0.017**   |
| Wine glass | 0.72 | 0.94       | 0.94        | **0.636**   |

Standard NMS: beer glass wins (0.81 > 0.72) — **wrong**.
Task-NMS: wine glass wins (0.636 > 0.017) — **correct**.

---

## What to Build for Stage 2A

### Inference Pipeline Architecture

```
Input: (image, task_id)
       ↓
[1] Preprocessing
    - Resize to 160×160
    - Normalize: INT8 = clamp((uint8 - mean_c) × scale_c, -128, 127)

[2] SAM-IP (parallel/sequential in software, parallel on FPGA)
    - Forward pass of tiny 3-layer CNN on 40×40×3 thumbnail
    - Apply per-task threshold → binary 40×40 mask

[3] YOLOv5n Detection (INT8 quantized)
    - Forward pass → raw proposals (~300 bboxes)
    - Decode: sigmoid/exp for box coordinates, objectness, class probs

[4] SAM Soft Gating
    - For each proposal: check mask at center (cx/160×40, cy/160×40)
    - If mask[cy_scaled][cx_scaled] == 0: conf_effective = conf × 0.25
    - Else: conf_effective = conf (unchanged)

[5] RoI Feature Extraction
    - For each surviving proposal:
      - RoI Align → 7×7 feature grid from backbone feature map
      - Global Average Pool: 7×7×256 → 256-D
      - Linear projection: 256 → 128-D

[6] Task Scoring
    - score_i = dot(feature_i[128], task_emb[task_id][128])
    - Normalize to [0, 255]

[7] Task-Fused NMS
    - fused_i = conf_eff_i × score_i × class_task_prior[class_i][task_id]
    - Sort by fused score (descending)
    - Greedy IoU suppression (threshold = 0.5)
    - Keep top-5 results

Output: Top-1 result → (bbox, class_name, fused_score)
```

### Training Requirements

Before writing inference code, you need trained model weights. Here is what to train:

**A. YOLOv5n backbone (pre-trained weights available)**
- Use Ultralytics YOLOv5n pretrained on COCO (`yolov5n.pt`)
- Fine-tune on COCO-Tasks subset if needed for better task-relevant class detection
- Quantize to INT8 using PyTorch PTQ or ONNX quantization

**B. SAM-IP weights**
- Tiny CNN (199 INT8 parameters total; see architecture above)
- Train as binary classifier: does this spatial cell contain a task-relevant object?
- Positive: cells containing ground-truth bbox of task-preferred class
- Negative: all other cells
- Use pos_weight=25 in BCEWithLogitsLoss to handle severe class imbalance
- Target: ≥97% recall per task on COCO-Tasks validation set
- Calibrate per-task thresholds after training

**C. Task embeddings (14 × 128 INT8 matrix)**
- Train a contrastive/metric learning head on top of YOLOv5n features
- For each task, learn a 128-D embedding that scores high for task-preferred objects
- Alternatively: use mean pooled RoI features of all task-preferred instances per task as initial embeddings (no-train baseline acceptable for Stage 2A)

**D. Class-Task Prior Table (80 × 14)**
- Compute from COCO-Tasks annotation statistics:
  `prior[c][t] = count(class c preferred for task t) / count(class c in task-t images)`
- This is offline computation, not a trained model

### Suggested Code Structure

```
tactile/
├── config/
│   ├── tasks.py              # 14 task definitions and IDs
│   ├── class_task_prior.py   # 80×14 precomputed prior table
│   └── sam_thresholds.py     # 14 per-task calibrated thresholds
├── models/
│   ├── yolov5n_quant.py      # YOLOv5n INT8 inference wrapper
│   ├── sam_ip.py             # Tiny SAM CNN (3 layers)
│   └── task_emb.py           # Task embedding table (14×128)
├── pipeline/
│   ├── preprocess.py         # Image normalization
│   ├── soft_gate.py          # SAM mask application
│   ├── roi_extract.py        # RoI Align + GAP + projection
│   ├── task_score.py         # Dot-product scoring
│   └── fused_nms.py          # Task-score-fused NMS
├── inference.py              # Main entry point
├── eval.py                   # mAP evaluation on COCO-Tasks
└── demo.py                   # Video/image demo script
```

**Main entry point signature:**
```python
def infer(image_path: str, task_id: int) -> dict:
    """
    Args:
        image_path: path to input image
        task_id: integer 0–13 corresponding to one of 14 tasks
    Returns:
        {
          "bbox": [x1, y1, x2, y2],
          "class": "wine glass",
          "confidence": 0.636,
          "task": "serve wine"
        }
    """
```

---

## Architecture Modifications for FPGA-Friendliness (Stage 3 Preparation)

While Stage 2A is software-only, design the code with FPGA deployment in mind. The following modifications from the Stage 1 architecture should be evaluated and documented:

### 1. Keep 160×160 as primary resolution
The FPS budget closes at 160×160 with the 8×8 INT8 systolic array. Do NOT use 224×224 or 416×416 as primary resolution — the 50 FPS target cannot be met at those resolutions on hardware.

### 2. Quantize everything to INT8
Avoid FP32/FP16 in the main inference path. INT8 maps directly to the DSP48E1 packing trick (2 INT8 MACs per DSP). Use PyTorch's `torch.quantization` or ONNX Runtime quantization.

### 3. Replace depthwise conv with standard conv in SAM-IP if needed
The architecture uses depthwise convolution only in SAM-IP Layer 2 (4-channel depthwise). On FPGA, this is implemented as 4 FIR filters. In PyTorch this is `groups=4`. Verify this exports cleanly to ONNX. If it causes issues, replace with a standard 3×3 conv (acceptable for Stage 2A).

### 4. Avoid dynamic shapes
Fix all tensor shapes at compile time. The systolic array has fixed tiling logic for static input dimensions. Variable-size inputs require padding to 160×160.

### 5. Use division-free IoU in NMS
The Stage 1 NMS-IP uses `2×intersection > union` instead of dividing. Implement this in software NMS as well for consistency and to validate the hardware version.

### 6. Keep the projection head at 256→128
The 256→128 linear projection after GAP produces the 128-D features used in task scoring. This maps exactly to one systolic array pass. Do not change this dimensionality without re-evaluating BRAM usage.

### 7. Document any changes from Stage 1 architecture
If you simplify, replace, or skip any module for Stage 2A (e.g., using a simpler SAM or skipping RoI align), clearly document what was changed and why in the Page 1 report. The contest evaluates novelty and alignment with the original proposal.

---

## Two-Page Report Requirements

The report must be **exactly two pages**. Follow the Stage 2A report format (not the Stage 1 template).

### Page 1: Approach Description (~500 words + pipeline diagram)

Cover:
- Brief restatement of the problem (task-aware object detection)
- Your pipeline stages with a block diagram (image → preprocessing → SAM → YOLOv5n → gating → RoI → task scoring → fused NMS → output)
- Key design decisions:
  - Why YOLOv5n (not DETR, not MobileNetV3, not NanoDet)
  - Why soft gating (not hard-drop)
  - Why fused NMS score formula
- Any modifications from Stage 1 proposal and rationale
- Dataset used (COCO-Tasks subset from Sawatzky et al.)
- Training setup (if applicable)

### Page 2: Results Snapshot (~4–6 result images + accuracy table)

Cover:
- At least 4–6 result images showing different tasks, with bbox overlay and task label
- Ideally include the wine glass vs beer glass disambiguation example
- Accuracy table: per-task AP or success rate on a test subset
- Any failure cases and analysis
- Runtime on CPU (ms per image)

---

## Evaluation Criteria (from Contest)

The submission will be evaluated on:
1. **Novelty** of the detection pipeline
2. **Accuracy** — correct task-relevant object ranked #1 across 14 tasks
3. **Inference latency** (CPU for Stage 2A; FPGA for Stage 3)
4. **CPU/FPGA utilization** (hardware metrics deferred to Stage 3)
5. **Power/energy consumption** (deferred to Stage 3)

For Stage 2A, criteria 1 and 2 are primary. Demonstrate both the novel fused-score NMS and the SAM spatial attention in results.

---

## References

- Sawatzky et al., "What Object Should I Use? — Task Driven Object Detection", CVPR 2019. https://doi.org/10.48550/arXiv.1904.03000
- Ultralytics YOLOv5: https://github.com/ultralytics/yolov5
- COCO Dataset: https://cocodataset.org
- TACTILE Stage 1 Architectural Reference Manual (Architectural_guide.md) — authoritative source for all architectural decisions
- Xilinx Kintex-7 DSP48E1: UG479
- VEGA AS1061 RV64GC: CDAC Trivandrum documentation

---

## Summary Checklist for Stage 2A

Before submission, verify:

- [ ] `inference.py` runs on CPU and accepts (image_path, task_id) → returns top-1 result
- [ ] Pipeline covers all 14 tasks
- [ ] SAM soft gating implemented (not hard-drop)
- [ ] Fused NMS uses `conf × task_score × class_prior` formula
- [ ] INT8 quantized YOLOv5n weights used (or clearly documented why FP32 was used for Stage 2A)
- [ ] Two-page report: Page 1 = approach, Page 2 = results snapshots
- [ ] Video demo recorded showing ≥3 different task queries working correctly
- [ ] All three (code + report + video) zipped into single file
- [ ] Uploaded to EasyChair under `DC_Stage2_Submission` by 17 May 2026
