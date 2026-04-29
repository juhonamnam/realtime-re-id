import { getWeakestLinkSimilarity } from "../metrics/weakestLink"
import { getWeightedAvgSimilarity } from "../metrics/weightedAvg"
import type { Feature } from "../type"

const baseUrl = import.meta.env.BASE_URL

export type PDModelInfo = {
  idx: number
  name: string
  path: string
  shape: readonly [number, number]
  threshold: number
}

type MetricName = "Weakest Link" | "Weighted Average"

export type FEModelInfo = {
  idx: number
  name: string
  path: string
  shape: readonly [number, number]
  similarityThresholds: { [key in MetricName]: number }
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
    similarityThresholds: {
      ["Weakest Link"]: 0.64,
      ["Weighted Average"]: 0.74,
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
  { name: "Weakest Link" as const, function: getWeakestLinkSimilarity },
  { name: "Weighted Average" as const, function: getWeightedAvgSimilarity },
].map((metricType, idx) => ({ ...metricType, idx }))
