# Real-time Person Re-Identification with Body Part Segmentation

This repository contains a real-time person re-identification (Re-ID) system aimed for high-performance inference on actual service environments. The model is inspired by [**BPBReID** (Body-Part-Based Re-Identification)](https://github.com/vlsomers/bpbreid), utilizing a CNN-based architecture to segment body parts and extract localized embedding vectors for robust matching.

Along with the speed and accuracy, the model also considers the coverage. In real-world scenarios, people may be partially occluded or cropped out of the frame during the person detection. To address this, we do a **quality filtering**, which discards invalid bounding boxes. If the filtering is too strict, it may show a high accuracy but low coverage, and if it is too loose, it may show a high coverage but low accuracy. Therefore, we consider the **trade-off between speed, accuracy and coverage** as a key metric for evaluating the model's performance.

## Model Overview

The system identifies individuals by segmenting the body into predefined body parts, and the default is **Head**, **Torso**, **Arms**, **Upper Legs**, and **Lower Legs**. It also uses a **foreground mask**, which is a combination of all body parts, to extract global features.

- **Input**: Image tensor.
- **Output 1**: Visibility score for each body part (range [0, 1]).
- **Output 2**: Embedding vector for each body part.

### Architecture

#### Feature Extraction

In the original BPBReID model, **HRNet** was the most accurate backbone. In this model, we prioritize speed and efficiency, so we also experimented on **ResNet50** and **MobileNetV3** as the backbone, with a **Feature Pyramid Network (FPN)** structure to achieve high-resolution segmentation.

![Model Architecture](model-architecture.jpg)
_Visual representation of the feature extraction and segmentation process._

#### Quality Filtering

In order to ensure that the model only processes valid bounding boxes, we only keep the bounding boxes where all body parts are visible. We examined the coverages on two segmentation variants:

- **5pf**: 5-part segmentation (Head, Torso, Arms, Upper Legs, Lower Legs) with a foreground mask.
- **5paf**: Alternative 5-part segmentation (Head, Torso, Arms, Legs, Feet) with a foreground mask.

Since the feet are often occluded or cropped out of the frame, the 5paf variant shows a lower coverage than the 5pf variant. Therefore, we use the 5pf variant as the default segmentation.

#### Comparison & Metrics

In the original BPBReID model, comparison between two images is done by calculating L2 distances between the embedding vectors of each body part that is commonly visible on both images. In our model we assume all body parts are visible from the quality filtering. Therefore, we can safely concatenate all embedding vectors into a single vector and calculate the L2 distance between the two vectors. Handling a single embedding vector per image has a greater advantage in application since it can leverage **Approximate Nearest Neighbor (ANN)** search algorithms for faster retrieval.

![Comparison Diagram](comparison-diagram.jpg)
_Visual representation of the comparison process._

### Training

The model was trained using a multi-task approach:

- **Segmentation**: Trained on the **COCO DensePose** dataset using **Focal Loss**.
- **Feature Extraction**: Trained on Re-ID datasets from **Market1501**, **DukeMTMC-reID**, and **Korean Re-Identification Image Dataset** using a combination of **Classification Loss** and **Triplet Loss**.

When training feature extraction, all datasets were used simultaneously to improve the model's generalization. When doing so, we made sure that images from one dataset are not compared with images from another dataset. Images from different datasets are not mixed in the same batch, and different classification layers are used for each dataset when training.

Training on all datasets simultaneously may reduce performance on individual benchmarks, but it can improve the model's generalization across different domains.

---

## Evaluation on Market1501

| Backbone                | Segment Variant | mAP (%) | Rank-1 (%) | Coverage (%) | Speedup (vs HRNet) |
| ----------------------- | --------------- | ------- | ---------- | ------------ | ------------------ |
| HRNet                   | 5pf             | 76.7    | 91.0       | 91.4         | 1.0 x              |
| ResNet50 + FPN          | 5pf             | 68.9    | 87.0       | 91.4         | 2.34 x             |
| MobileNetV3 Large + FPN | 5pf             | 63.6    | 82.9       | 91.0         | 3.37 x             |
| MobileNetV3 Small + FPN | 5pf             | 51.6    | 73.6       | 80.2         | 4.21 x             |
| HRNet                   | 5paf            | 79.8    | 91.5       | 71.2         | 1.0 x              |
| ResNet50 + FPN          | 5paf            | 74.6    | 87.6       | 56.2         | 2.34 x             |
| MobileNetV3 Large + FPN | 5paf            | 66.7    | 81.7       | 56.7         | 3.37 x             |
| MobileNetV3 Small + FPN | 5paf            | 61.8    | 73.0       | 25.4         | 4.21 x             |

Coverage is defined as the percentage of valid bounding boxes that are not filtered out by the quality filtering.

---

## Live Demo

A live demo of this system is available here:
👉 **[https://juhonamnam.github.io/realtime-re-id/](https://juhonamnam.github.io/realtime-re-id/)**

The demo runs entirely in your browser using `onnxruntime-web` and your webcam.

### How to use:

1.  **Snapshot**: Capture a frame where people are visible.
2.  **Selection**: Click on one of the detected people to set them as the target.
3.  **Tracking**: The system will now track and identify people in the live feed:
    - **Red Bounding Box**: The model identifies the person as the target.
    - **Yellow Bounding Box**: The model identifies the person as a different individual.
    - **Gray Bounding Box**: Invalid bounding box (not enough body parts visible).
