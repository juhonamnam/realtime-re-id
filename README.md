# Real-time Person Re-Identification with Body Part Segmentation

This repository contains a real-time person re-identification (Re-ID) system that performs detection and identification directly in the browser. The model is inspired by **BPBReID** (Body-Part-Based Re-ID), utilizing a CNN-based architecture to segment body parts and extract localized embedding vectors for robust matching.

## Model Overview

The system identifies individuals by segmenting the body into five key parts: **Head**, **Torso**, **Arms**, **Upper Legs**, and **Lower Legs**.

- **Input**: Image tensor.
- **Output 1**: Visibility score for each body part (range [0, 1]).
- **Output 2**: Embedding vector for each body part.

### Architecture

#### Feature Extraction
The model uses **MobileNetV3** as its backbone for high-speed inference, coupled with a **Feature Pyramid Network (FPN)** structure for high-resolution segmentation.

- **Segmentation as Attention**: Feature maps from various levels of the backbone are combined to generate both an embedding feature map and a segmentation map. The segmentation map acts as an attention mechanism; the model calculates weighted averages of the embedding map against these segments to extract part-specific features.
- **Local Feature Focus**: To prioritize local details and minimize global noise, the embedding feature map is derived primarily from shallow feature maps, while the segmentation map utilizes a combination of deep, shallow, and intermediate maps.

![Model Architecture](model-architecture.jpg)
*Visual representation of the feature extraction and segmentation process.*

#### Comparison & Metrics
Matching between two images is performed using the **cosine similarity** of each body part's embedding. Visibility scores are used to selectively weight these parts.

To determine if two images represent the same person, multiple metrics are supported to combine part-level similarities into a single score:
- **Product**: Intuitively weights mismatches more heavily. If one part (e.g., shoes) differs significantly, the overall score drops sharply.
- **Weighted Mean**: The standard approach for score aggregation.
- **Weighted Geometric Mean**: Provides a balance between the mean and the product.
- **Minimum**: Focuses entirely on the "weakest link" or the least similar visible part.

The **Product** and **Minimum** metrics were introduced specifically to capture critical mismatches. Intuitively, when comparing two people, a single significant difference (e.g., different shoes on otherwise similar outfits) is often enough to determine they are different individuals. By considering the "weakest link" among part-level similarities rather than just an average, the system becomes more robust against false positives in crowded or similar-looking environments.

Furthermore, each part-level similarity score undergoes **logistic remapping**. This is necessary because different body parts (e.g., Torso vs. Legs) may have different optimal similarity thresholds due to their varying descriptive power or commonality. The remapping process aligns the chosen threshold for each body part to a uniform value. These thresholds are carefully selected to maximize true positives while maintaining clear separability between different individuals. The core philosophy is to decisively eliminate candidates that are "surely different" while retaining those that are "potentially the same."

![Comparison Diagram](comparison-diagram.jpg)
*Diagram showing how body part similarities are combined.*

### Training

The model was trained using a multi-task approach:
- **Segmentation**: Trained on the **COCO DensePose** dataset using **Focal Loss**.
- **Feature Extraction**: Trained on Re-ID datasets from **AI Hub** (a Korean government-funded platform) using a combination of **Classification Loss** and **Triplet Loss**.

---

## Live Demo

A live demo of this system is available here:  
👉 **[https://juhonamnam.github.io/realtime-re-id/](https://juhonamnam.github.io/realtime-re-id/)**

The demo runs entirely in your browser using `onnxruntime-web` and your webcam.

### How to use:
1.  **Snapshot**: Capture a frame where people are visible.
2.  **Selection**: Click on one of the detected people to set them as the target.
3.  **Tracking**: The system will now track and identify people in the live feed:
    -   **Red Bounding Box**: The model identifies the person as the target.
    -   **Yellow Bounding Box**: The model identifies the person as a different individual.

---

## Project Structure

- `/re-id`: Python source code for model training, evaluation, and ONNX export.
- `/web`: React-based web application for real-time inference.
