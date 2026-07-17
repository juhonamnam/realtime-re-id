import type { Feature } from "../type"

import { getConcatDist } from "./concatDist"
import { getDistMean } from "./distMean"

export type MetricType = "Concat Distance" | "Distance Mean"

type DistanceResult = {
  finalDist: number
  vScores: number[]
  partDists: number[]
}

const MIN_V_SCORE = 0.1

const distFunctions: {
  [key in MetricType]: (
    embVecs1: Float32Array[],
    embVecs2: Float32Array[],
  ) => [number[], number]
} = {
  ["Concat Distance"]: getConcatDist,
  ["Distance Mean"]: getDistMean,
}

export const buildDistanceFunc = (metricType: MetricType) => {
  const distanceFunc = distFunctions[metricType]
  return (features1: Feature, features2: Feature): DistanceResult => {
    const { vScores: vScores1, embVecs: embVecs1 } = features1
    const { vScores: vScores2, embVecs: embVecs2 } = features2

    const vScores = []

    for (let i = 0; i < vScores1.length; i++) {
      vScores.push(Math.min(vScores1[i], vScores2[i]))
    }

    if (vScores.some((score) => score < MIN_V_SCORE)) {
      return {
        finalDist: -1,
        vScores,
        partDists: [],
      }
    }

    const [partDists, finalDist] = distanceFunc(embVecs1, embVecs2)

    return { finalDist, vScores, partDists }
  }
}

export type MetricOption = {
  idx: number
  type: MetricType
  function: (features1: Feature, features2: Feature) => DistanceResult
}

export const METRIC_OPTIONS: MetricOption[] = [
  {
    type: "Concat Distance" as const,
    function: buildDistanceFunc("Concat Distance"),
  },
  {
    type: "Distance Mean" as const,
    function: buildDistanceFunc("Distance Mean"),
  },
].map((metricType, idx) => ({ ...metricType, idx }))
