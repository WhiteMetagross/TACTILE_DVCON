# Tactile Edge AI Project:

The Task Aware Cascaded Inference with Lightweight Edge Deployment pipeline is designed for the DVCon India 2026 Design Contest. This architecture introduces a highly optimized approach to computer vision on edge devices. It prioritizes contextual relevance over brute force computation. The framework relies on a spatial attention map and task fused non maximum suppression. This ensures that processing power focuses strictly on objects relevant to the active user intent. 

## Architectural Overview:

The core of the system evaluates user intent before engaging deep detection networks. The pipeline consists of the following phases.
* The system isolates regions of interest using a lightweight spatial attention map.
* It leverages an optimized INT8 quantized detector to find proposals.
* Background distractors are dynamically suppressed.
* Task scoring computes a localized match against stored embeddings.
* Final results are filtered using a division free intersection over union algorithm.

## Performance Metrics:

The hyperparameter optimization successfully discovered the optimal weights for the spatial attention map. The current implementation achieves a validation recall of 99.92 percent. Standard inference overhead adds less than 4 milliseconds to the base detector execution time. The pipeline is structurally validated against hardware equivalents. 

## Demonstration Output:

Below are generated visuals showing the pipeline correctly performing Task 13 (Read a book) and Task 0 (Serve wine) by detecting the relevant targets.
![Task 13 Output](Visuals/ReadBookVisual.jpg)
![Task 0 Output](Visuals/ServeWineVisual.jpg)

## Project Structure:

Please refer to the specific documentation files for detailed guidance.
* Review InstallationAndSetup.md for environment configuration.
* Review Usage.md for execution parameters and scripts.
* Review CodeBaseIndex.md for an index of all modules.
