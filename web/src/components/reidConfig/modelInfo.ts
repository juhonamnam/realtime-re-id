import {
  getProductSimilarity,
  getWeightedMeanSimilarity,
  getWeightedGeometricMeanSimilarity,
  getMinimumSimilarity,
  buildSimilarityFunc,
} from "../metrics"
import type { Feature } from "../type"

const baseUrl = import.meta.env.BASE_URL

export type PDModelInfo = {
  idx: number
  name: string
  path: string
  shape: readonly [number, number]
  threshold: number
}

type MetricName =
  | "Product"
  | "Weighted Mean"
  | "Weighted Geometric Mean"
  | "Minimum"

export type FEModelInfo = {
  idx: number
  name: string
  path: string
  shape: readonly [number, number]
  defaultSimilarityThresholds: { [key in MetricName]: number }
  partSimilarityThresholds: number[]
  segmentNames: string[]
  visibilityThreshold: number
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
    name: "ReID M3Small 5a24e 64x192",
    path: `${baseUrl}/model/reid_m3small_5a24e_64x192.onnx`,
    shape: [192, 64] as const,
    defaultSimilarityThresholds: {
      ["Product"]: 0.64,
      ["Weighted Mean"]: 0.66,
      ["Weighted Geometric Mean"]: 0.64,
      ["Minimum"]: 0.64,
    },
    partSimilarityThresholds: [0.5, 0.6, 0.5, 0.5, 0.5],
    visibilityThreshold: 0.7,
    segmentNames: ["Head", "Torso", "Arm", "Upper Leg", "Lower Leg"],
  },
].map((model, idx) => ({
  ...model,
  idx,
}))

export type MetricType = {
  idx: number
  name: MetricName
  function: (
    feature1: Feature,
    feature2: Feature,
    thresholds: number[],
  ) => [number, number[]]
}

export const METRIC_TYPES: MetricType[] = [
  {
    name: "Product" as const,
    function: buildSimilarityFunc(getProductSimilarity),
  },
  {
    name: "Weighted Mean" as const,
    function: buildSimilarityFunc(getWeightedMeanSimilarity),
  },
  {
    name: "Weighted Geometric Mean" as const,
    function: buildSimilarityFunc(getWeightedGeometricMeanSimilarity),
  },
  {
    name: "Minimum" as const,
    function: buildSimilarityFunc(getMinimumSimilarity),
  },
].map((metricType, idx) => ({ ...metricType, idx }))
