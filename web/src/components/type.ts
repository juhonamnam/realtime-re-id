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
