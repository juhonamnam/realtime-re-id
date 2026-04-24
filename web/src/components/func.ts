import { COLOR_OF_MATCH, COLOR_OF_UNMATCH } from "./const"
import type { Feature } from "./type"

const dot = (vec1: Float32Array, vec2: Float32Array): number => {
  let result = 0
  for (let i = 0; i < vec1.length; i++) {
    result += vec1[i] * vec2[i]
  }
  return result
}

const magnitude = (vec: Float32Array): number => {
  let result = 0
  for (let i = 0; i < vec.length; i++) {
    result += vec[i] ** 2
  }
  return Math.sqrt(result)
}

const getEmbSimilarityScore = (
  embVec1: Float32Array,
  embVec2: Float32Array,
): number => {
  const dotProduct = dot(embVec1, embVec2)
  const magnitude1 = magnitude(embVec1)
  const magnitude2 = magnitude(embVec2)

  const denominator = magnitude1 * magnitude2
  if (denominator === 0) {
    return -1
  }
  return dotProduct / denominator
}

const VALUE_AT_THRESHOLD = 0.7

export const getSimilarityScore = (
  feature1: Feature,
  feature2: Feature,
  thresholds: number[],
  exp: number = 2,
): number => {
  const combinedVScores: number[] = []
  for (let i = 0; i < feature1.vScores.length; i++) {
    combinedVScores.push(Math.min(feature1.vScores[i], feature2.vScores[i]))
  }

  let totalSScore = 1

  for (let i = 0; i < feature1.embVecs.length; i++) {
    const combinedVScore = combinedVScores[i]
    const unknownRatio = 1 - combinedVScore

    const embVec1 = feature1.embVecs[i]
    const embVec2 = feature2.embVecs[i]
    let sScore = getEmbSimilarityScore(embVec1, embVec2)

    const threshold = (thresholds[i] + 1) / 2
    sScore = (sScore + 1) / 2

    if (sScore < threshold) {
      sScore = (VALUE_AT_THRESHOLD / threshold ** exp) * sScore ** exp
    } else {
      sScore =
        1 -
        ((1 - VALUE_AT_THRESHOLD) / (1 - threshold) ** exp) *
          (1 - sScore) ** exp
    }

    totalSScore *= unknownRatio + combinedVScore * sScore
  }

  return totalSScore
}

export const getSimilarityColor = (
  similarityScore: number,
  threshold: number,
) => {
  const matchColorRatio =
    ((similarityScore + 1) / 2) **
    (Math.log(0.5) / Math.log((threshold + 1) / 2))
  const unmatchColorRatio = 1 - matchColorRatio

  return [
    COLOR_OF_MATCH[0] * matchColorRatio +
      COLOR_OF_UNMATCH[0] * unmatchColorRatio,
    COLOR_OF_MATCH[1] * matchColorRatio +
      COLOR_OF_UNMATCH[1] * unmatchColorRatio,
    COLOR_OF_MATCH[2] * matchColorRatio +
      COLOR_OF_UNMATCH[2] * unmatchColorRatio,
  ]
}

export const getVisibilityColor = (
  visibilityScore: number,
  threshold: number,
) => {
  const visibleColorRatio =
    visibilityScore ** (Math.log(0.5) / Math.log(threshold))
  const notVisibleColorRatio = 1 - visibleColorRatio

  return [
    COLOR_OF_MATCH[0] * visibleColorRatio +
      COLOR_OF_UNMATCH[0] * notVisibleColorRatio,
    COLOR_OF_MATCH[1] * visibleColorRatio +
      COLOR_OF_UNMATCH[1] * notVisibleColorRatio,
    COLOR_OF_MATCH[2] * visibleColorRatio +
      COLOR_OF_UNMATCH[2] * notVisibleColorRatio,
  ]
}

export const getTotalVisibilityColor = (
  feature: Feature,
  threshold: number,
) => {
  const visibleScore =
    feature.vScores.reduce((acc, cur) => acc + cur, 0) / feature.vScores.length
  return getVisibilityColor(visibleScore, threshold)
}

export const visible = (feature: Feature, threshold: number) => {
  return feature.vScores.every((score) => score >= threshold)
}

export const relativelyVisible = (
  featureToCompare: Feature,
  feature: Feature,
  threshold: number,
) => {
  return feature.vScores.every((score, i) =>
    featureToCompare.vScores[i] >= threshold ? score >= threshold : true,
  )
}
