# Execution And Usage Guide:

This document details the operational instructions for the Task Aware Cascaded Inference pipeline. The system includes several scripts for training, evaluation, and demonstration purposes. All scripts must be executed from the project root directory.

## Running The Interactive CLI:

The interactive CLI provides a user-friendly interface to test the pipeline dynamically.
* Execute `python Cli.py`.
* Select an inference task from the displayed list.
* Input a corresponding Image ID (e.g., `8211`) or press `m` to manually enter a path.
* The script will visualize the object detection output directly via OpenCV and save it to the output directory.

## Running The Programmatic Demonstration:

The `Demo.py` script processes a single image programmatically without interactive prompts.
* Execute the `Demo.py` script.
* Provide the path to the target image using the image argument.
* Specify the numerical identifier for the desired task using the task argument.
* Define the output destination using the output argument.
* The script will process the image and save the result to the specified location.

*A recorded walkthrough is also provided in the repository as `CompressedDemo.mp4`.*

## Generating Task Embeddings:

The system requires task embeddings to score the detected bounding boxes. 
* Run the TrainTaskEmb.py script.
* The script will compute mean pooled features from the preferred objects.
* It will save the resulting array to the Weights folder.

## Training The Spatial Attention Map:

The spatial attention map can be trained from scratch or optimized using the provided tools.
* Execute the HpoSam.py script to begin the automated hyperparameter search.
* The script will run 30 optimization trials.
* Following the trials, it will train the final model using the best discovered parameters.
* The optimal weights will be automatically saved to the Weights directory.

## Evaluating Pipeline Performance:

The evaluation script measures the overall success rate of the pipeline against the validation dataset.
* Run the Evaluate.py script.
* The system will calculate the top success rate across all 14 predefined tasks.
* The script will output an accuracy table detailing the performance metrics.
* Use this script to confirm that the pipeline meets the required design constraints.

## Visual Output Reference:

The demonstration script generates comprehensive visuals for inspection.
![Task 13 Output](Visuals/ReadBookVisual.jpg)
![Task 0 Output](Visuals/ServeWineVisual.jpg)
