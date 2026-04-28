import type { FEModelInfo, PDModelInfo, Status } from "./type"

const baseUrl = import.meta.env.BASE_URL

export const OFFSET_REM = 13

export const PD_MODELS: { [key: string]: PDModelInfo } = {
  "YOLOv11 320x320": {
    path: `${baseUrl}/model/yolo11n_person-detection_320x320.onnx`,
    shape: [320, 320],
    threshold: 0.5,
  },
  "YOLOv11 640x640": {
    path: `${baseUrl}/model/yolo11n_person-detection_640x640.onnx`,
    shape: [640, 640],
    threshold: 0.5,
  },
}

export const FE_MODELS: { [key: string]: FEModelInfo } = {
  "ReID M3Small 5a24e 64x192": {
    path: `${baseUrl}/model/reid_m3small_5a24e_64x192.onnx`,
    shape: [192, 64],
    similarityThreshold: 0.64,
    partSimilarityThresholds: [0.5, 0.6, 0.5, 0.5, 0.5],
    visibilityThreshold: 0.7,
    segmentNames: ["Head", "Torso", "Arm", "Upper Leg", "Lower Leg"],
  },
}

export const DEFAULT_PD_MODEL = Object.keys(PD_MODELS)[0]
export const DEFAULT_FE_MODEL = Object.keys(FE_MODELS)[0]

export const LOCAL_STORAGE_PREFIX = "reid-"

export const INITIAL_STATUS: Status = "default"

export const COLOR_OF_UNMATCH = [255, 255, 0] as const
export const COLOR_OF_MATCH = [255, 0, 0] as const
export const LINE_WIDTH = 3
export const FONT_SIZE = 16
