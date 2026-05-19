# Installation And Setup Guide:

This document outlines the procedures required to construct the execution environment. The dependencies have been carefully selected to ensure compatibility with both central processing unit simulation and eventual field programmable gate array deployment. 

## Environment Preparation:

The project utilizes Conda for environment isolation. You must prepare the virtual environment before installing packages.
* Install Miniconda or Anaconda on your system.
* Create a new environment using Python version 3.11.
* Activate the newly created environment.

## Dependency Installation:

The codebase requires specific machine learning and computer vision libraries. 
* Install the PyTorch library tailored for your computational hardware.
* Install the OpenCV library for image processing (`opencv-python`).
* Install the Ultralytics library to access the base detector model.
* Install Optuna for running hyperparameter optimization tasks.

*Note: For WSL users intending to use the interactive CLI (`Cli.py`), ensure WSLg is functioning properly for `cv2.imshow` windows. A fallback font directory config is provided in the script to suppress Qt errors.*

## Dataset Configuration:

The pipeline relies on the Coco Task Dataset for training and evaluation. 
* Download the dataset archive to your local storage.
* Extract the contents to the designated CocoTaskDataset directory.
* Run the SetupDataset.py script to organize annotations and images properly.

## Initialization Checklist:

Before running the main scripts, ensure that all components are verified.
* Ensure the Weights directory contains the precomputed models.
* Verify that the test suite passes all assertions.
* Validate that your hardware supports the required tensor operations.
