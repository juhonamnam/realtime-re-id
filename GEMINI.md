# Realtime Re-ID

A person re-identification system that performs real-time detection and identification in the browser using ONNX models.

## Project Overview

This project consists of two main components:
1.  **Re-ID Model Training (`/re-id`)**: A Python-based environment for training and exporting person re-identification models. It uses MobileNetV3 as a backbone and incorporates a segmentation-based attention mechanism.
2.  **Web Application (`/web`)**: A React-based frontend that uses `onnxruntime-web` to run person detection (YOLOv11) and Re-ID models directly in the user's browser via webcam.

## Architecture

### AI / Model Training (`/re-id`)
- **Backbone**: MobileNetV3 (Small/Large) with FPN.
- **Attention Mechanism**: Segmentation-based attention to focus on specific body parts.
- **Loss Functions**: Combination of Segmentation Loss, Classification Loss, and Triplet Loss.
- **Export**: Models are exported to `.onnx` format for web deployment.
- **Key Files**:
  - `re-id/src/models/reid.py`: Core model architecture.
  - `re-id/main.ipynb`: Primary notebook for training, evaluation, and ONNX export.
  - `re-id/src/data/`: Data loading and augmentation logic.

### Web Application (`/web`)
- **Framework**: React 19 with Vite and TypeScript.
- **Inference**: `onnxruntime-web` for client-side model execution.
- **Styling**: Bootstrap and Sass.
- **Key Components**:
  - `web/src/components/reIdentification.tsx`: Main logic for camera feed processing and Re-ID.
  - `web/src/components/cam/`: Camera management hooks and providers.
  - `web/src/components/reidConfig/`: Model and metric configuration.
  - `web/public/model/`: Contains the ONNX model files.

## Getting Started

### Web Application
1.  Navigate to the `web/` directory.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Build for production:
    ```bash
    npm run build
    ```

### Re-ID Model Training
1.  Navigate to the `re-id/` directory.
2.  It is recommended to use `uv` for environment management.
3.  Install dependencies:
    ```bash
    uv sync
    ```
4.  Open `main.ipynb` in a Jupyter environment to start training or exporting models.

## Development Conventions

### General
- **Monorepo Structure**: Keep AI training and web implementation separated in their respective directories.
- **Model Updates**: When updating a model, export it to `.onnx` and place it in `web/public/model/`, then update `web/src/components/reidConfig/modelInfo.ts` if necessary.

### Frontend (`/web`)
- Use TypeScript for all components and logic.
- Prefer functional components and hooks.
- Follow the existing folder structure in `src/components`.

### AI (`/re-id`)
- Use the `src/` directory for modularizing model, data, and utility code.
- Keep experimental code in Jupyter notebooks but move stable logic to `.py` files in `src/`.
