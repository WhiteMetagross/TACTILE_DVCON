# Code Base Index:

This document serves as a comprehensive directory mapping for the entire project. It outlines the purpose and function of each module within the repository.

## Root Directory Scripts:

These scripts handle top level execution and orchestration tasks.
* Demo.py manages the visualization and single image inference testing.
* Evaluate.py runs the full dataset validation to calculate the top success rate.
* HpoSam.py performs hyperparameter optimization for the spatial attention map.
* RunTests.py executes the unit testing suite to validate all individual components.
* SetupDataset.py prepares and organizes the image data and annotations.
* TrainSam.py initiates the standard training loop for the spatial attention model.
* TrainTaskEmb.py computes and saves the analytical task embedding vectors.

## Configuration Modules:

The Config folder contains static definitions and heuristic mappings.
* ClassTaskPrior.py computes the prior probability matrix linking object classes to tasks.
* SamThresholds.py defines the suppression alpha value and mask dimensions.
* Tasks.py stores the numerical identifiers and string names for all fourteen categories.

## Neural Network Models:

The Models directory houses the PyTorch definitions for the deep learning components.
* SamIp.py defines the three layer convolutional architecture for the attention mask.
* TaskEmb.py implements the lookup table for the computed embedding vectors.
* YoloV5nWrapper.py provides the interface to the Ultralytics base detector.

## Pipeline Processing Stages:

The Pipeline folder implements the core operational logic of the framework.
* FusedNms.py replaces standard filtering with the division free fused algorithm.
* Inference.py orchestrates the end to end forward pass of the system.
* Preprocess.py handles image resizing and normalization before network execution.
* RoiExtract.py maps bounding boxes to feature grids to extract localized data.
* SoftGate.py applies the spatial attention mask to suppress irrelevant proposals.
* TaskScore.py executes the bitwise dot product matching against the task embeddings.
