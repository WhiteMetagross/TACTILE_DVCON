# Architecture Justification And Model Selection:

This document details the precise rationale for utilizing the YOLOv5n architecture instead of more recent iterations such as YOLOv8, YOLOv11, or YOLOv26. The selection is driven by rigid hardware constraints and algorithmic requirements mandated by the Stage 2A design prompt.

## Hardware Synthesis Constraints:

The target hardware is the VEGA AS1061 soft processor on a Kintex 7 field programmable gate array. This severely restricts our power and logic footprint.
* The hardware budget is strictly capped at fifty thousand lookup table slices.
* Newer models rely on Distribution Focal Loss to calculate bounding boxes.
* This loss mechanism requires complex exponential calculations across continuous distributions.
* Translating these mathematical operations into INT8 hardware logic requires excessive logic gates and exceeds the architectural budget.
* The chosen model utilizes simple anchor based regressions that translate perfectly into the native digital signal processing blocks on the hardware.

## Algorithmic Integrity:

The primary novelty of the TACTILE pipeline is the Task Fused Non Maximum Suppression algorithm. This custom block requires intercepting thousands of overlapping object proposals.
* Recent architectures from YOLOv10 to YOLOv26 utilize end to end suppression free detection paradigms.
* These newer models internally delete secondary overlapping proposals without external visibility.
* Losing access to raw overlapping guesses destroys the ability to inject custom task embeddings.
* The chosen model natively outputs a dense grid of proposals that the custom suppression block requires to function.

## Contest Compliance:

The design contest enforces strict adherence to the previously submitted technical proposals.
* The Stage 1 document explicitly defined a 160 by 160 input resolution utilizing a nano scale model.
* Transitioning to a decoupled head or suppression free model invalidates the original power and memory bandwidth projections.
* Strict compliance ensures the hardware synthesis phase proceeds without disqualification.
