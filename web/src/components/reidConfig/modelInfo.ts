import type { ExternalDataFileDescription } from "onnxruntime-web"
import type { MetricType } from "../metrics"

const baseUrl = import.meta.env.BASE_URL

export type PDModelInfo = {
  idx: number
  name: string
  path: string
  shape: readonly [number, number]
  threshold: number
}

export type FEModelInfo = {
  idx: number
  name: string
  path: string
  externalData: ExternalDataFileDescription[]
  shape: readonly [number, number]
  optimalThresholds: { [key in MetricType]: number }
  segmentNames: string[]
}

export const PD_MODELS: PDModelInfo[] = [
  {
    name: "YOLOv11 320x320",
    path: `${baseUrl}/model/yolo11n_person-detection_320x320.onnx`,
    shape: [320, 320] as const,
    threshold: 0.5,
  },
  {
    name: "YOLOv11 640x640",
    path: `${baseUrl}/model/yolo11n_person-detection_640x640.onnx`,
    shape: [640, 640] as const,
    threshold: 0.5,
  },
].map((model, idx) => ({
  ...model,
  idx,
}))

export const FE_MODELS: FEModelInfo[] = [
  {
    name: "ReID M3Large 5pf256e 64x192",
    path: `${baseUrl}/model/reid_m3large_5pf256e_64x192/model.onnx`,
    externalData: [],
    shape: [192, 64] as const,
    optimalThresholds: {
      ["Concat Distance"]: 3.3,
      ["Distance Mean"]: 1.3,
    },
    segmentNames: ["Head", "Torso", "Arm", "Leg", "Foot", "Full Body"],
  },
  {
    name: "ReID M3Small 5pf256e 64x192",
    path: `${baseUrl}/model/reid_m3small_5pf256e_64x192/model.onnx`,
    externalData: [],
    shape: [192, 64] as const,
    optimalThresholds: {
      ["Concat Distance"]: 2.4,
      ["Distance Mean"]: 1.0,
    },
    segmentNames: ["Head", "Torso", "Arm", "Leg", "Foot", "Full Body"],
  },
  {
    name: "ReID HRNet32 5pf256e 64x192",
    path: `${baseUrl}/model/reid_hrnet32_5pf256e_64x192/model.onnx`,
    externalData: [
      {
        path: "55c14cd0-81ac-11f1-935e-b52b2e712954.data",
        data: `${baseUrl}/model/reid_hrnet32_5pf256e_64x192/55c14cd0-81ac-11f1-935e-b52b2e712954.data`,
      },
    ],
    shape: [192, 64] as const,
    optimalThresholds: {
      ["Concat Distance"]: 3.7,
      ["Distance Mean"]: 1.5,
    },
    segmentNames: ["Head", "Torso", "Arm", "Leg", "Foot", "Full Body"],
  },
].map((model, idx) => ({
  ...model,
  idx,
}))
