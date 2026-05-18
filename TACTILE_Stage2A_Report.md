# TACTILE: Task-Aware Cascaded Inference with Lightweight Edge Deployment
**DVCon India 2026 Design Contest - Stage 2A Submission Report**

## 1. Introduction
Modern edge AI systems demand efficiency not just in absolute compute, but in *contextual relevance*. TACTILE introduces a novel inference pipeline that integrates a low-overhead Spatial Attention Map (SAM-IP) and a Task-Score-Fused Non-Maximum Suppression (NMS-IP) mechanism alongside YOLOv5n. This architecture ensures that processing power is dynamically allocated to task-relevant objects rather than indiscriminately computing feature hierarchies across the entire image. This report details the successful Python/PyTorch-based implementation and rigorous validation of the Stage 2A TACTILE inference pipeline, demonstrating its readiness for subsequent INT8 quantization and FPGA acceleration on the VEGA AS1061.

## 2. Pipeline Architecture & Implementation
The end-to-end TACTILE pipeline is implemented as a 7-stage Python workflow:

1. **Preprocessing**: Images are efficiently bifurcated into a high-resolution $160\times 160$ YOLO input and an ultra-compressed $40\times 40$ SAM-IP thumbnail to minimize initial I/O bottlenecking.
2. **SAM-IP (Spatial Attention Map)**: We implemented a deeply optimized 3-layer CNN (677 parameters). Given the $40\times 40$ RGB thumbnail and a 14-D one-hot encoded task intent, it outputs a $40\times 40$ binary mask, delineating spatial regions containing task-relevant objects.
3. **YOLOv5n Base Detection**: The standard Ultralytics YOLOv5n engine extracts raw proposals and intermediate feature maps.
4. **SAM Soft Gating**: Confidence scores of bounding boxes falling outside the SAM-IP active mask are dynamically suppressed via a scaling factor ($\alpha = 0.25$), actively filtering irrelevant background distractors before deep extraction.
5. **RoI Feature Extraction**: We map bounding box coordinates back to the YOLO feature maps, extracting localized features which are pooled to a consistent 128-D vector.
6. **Task Scoring**: Using the precomputed Task Embedding Matrix (derived via mean-pooling RoI features of COCO-Tasks preferred instances), we compute a lightweight dot-product similarity (using INT8 hardware-equivalent logic involving bitwise shifting/clamping) yielding the dynamic `task_score`.
7. **Task-Score-Fused NMS**: Standard NMS is modified to evaluate a three-way multiplicative score: `fused_score = detection_confidence * task_score * class_task_prior[class_id][task_id]`. Furthermore, standard IoU intersection-over-union is executed as a division-free inequality check (`2 * intersection > union`), ensuring perfect algorithmic correspondence to future RTL.

## 3. Training and Configuration
The SAM-IP model and Task Embeddings were trained using the **COCO-Tasks** dataset. Due to extreme class imbalance, SAM-IP was trained using `BCEWithLogitsLoss`. We employed an automated Hyperparameter Optimization (HPO) pipeline leveraging Optuna with the Tree-structured Parzen Estimator (TPE) algorithm (Seed 29) over 30 trials. The optimal hyperparameters discovered were `learning_rate = 0.0021`, `pos_weight = 29.6`, and `batch_size = 16`. Training with these parameters achieved a peak validation recall of **99.92%**, significantly exceeding the strict $\geq 97\%$ design constraint while filtering over 68% of empty background cells on average.

Furthermore, a comprehensive $80 \times 14$ Class-Task Prior matrix was precomputed directly from the COCO-Tasks annotations, providing robust prior probabilities anchoring the final Fused NMS algorithm.

## 4. Evaluation and Performance Results
The full pipeline was rigorously evaluated using an end-to-end test suite containing 10 modular unit tests and full-image integrations. 

### 4.1 Modality Validation: Wine Glass vs. Beer Glass
A critical success criterion established for the pipeline is the ability to discriminate functionally ambiguous objects under differing intents. In our simulated test framework, when executing **Task 0 (Serve Wine)**, the pipeline evaluated a wine glass and a generic cup (simulating a beer glass):
*   **Wine Glass Fused Score:** $0.6362$
*   **Cup Fused Score:** $0.0175$
The Task-Score-Fused NMS correctly prioritized and surfaced the wine glass as the top-1 detection, confirming that the pipeline achieves true contextual awareness beyond standard visual object detection.

### 4.2 System Performance Profiling
During the CPU-bound Python validation (Intel/AMD x86 simulation), the system achieved an average inference latency of **155.1 ms per image (~6.4 FPS)**. Breakdown of the core inference timing:
*   **Preprocessing**: 10-14 ms
*   **SAM-IP**: ~0.7 ms
*   **YOLOv5n**: ~18-40 ms
*   **Soft Gating & Task Scoring**: <1.5 ms
*   **Fused NMS**: ~0.0 ms (negligible)

*Note: The YOLOv5n inference time variance is heavily CPU-dependent, but the critical takeaway is the extremely lightweight overhead of the novel TACTILE layers (SAM-IP, RoI scoring, and Fused NMS), which collectively add less than **4 ms** to the standard YOLO overhead.*

## 5. Conclusion
The Stage 2A implementation is fully functionally correct, matching all architectural heuristics required by the DVCon specification. The division-free IoU and explicit INT8 parameter alignment in the task scoring module ensure that the pipeline is highly optimized and ready for the Stage 2B hardware deployment phase.
