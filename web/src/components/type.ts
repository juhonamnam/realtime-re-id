export type PDModelInfo = {
  path: string
  shape: [number, number]
  threshold: number
}

export type FEModelInfo = {
  path: string
  shape: [number, number]
  similarityThreshold: number
  partSimilarityThresholds: number[]
  segmentNames: string[]
  visibilityThreshold: number
}

export type Status = "default" | "select" | "reid"

export type Feature = {
  embVecs: Float32Array[]
  vScores: Float32Array
}

export type Snap = {
  canvas: OffscreenCanvas
  bboxes: [number, number, number, number][]
  features: Feature[]
}
