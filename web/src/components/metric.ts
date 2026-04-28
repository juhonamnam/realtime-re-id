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

const logisticRemap = (x: number, threshold: number): number => {
  const logistic =
    1 /
    (1 +
      Math.exp(Math.log(threshold / (1 - threshold)) - Math.log(x / (1 - x))))
  return logistic
}

export const getSimilarityScore = (
  feature1: Feature,
  feature2: Feature,
  thresholds: number[],
) => {
  const combinedVScores: number[] = []
  for (let i = 0; i < feature1.vScores.length; i++) {
    combinedVScores.push(Math.min(feature1.vScores[i], feature2.vScores[i]))
  }

  const vScoreSum = combinedVScores.reduce((acc, score) => acc + score, 0)
  if (vScoreSum === 0) {
    return [0, thresholds.map(() => 0)] as const
  }

  let totalSScore = 0
  const partSScores = []

  for (let i = 0; i < feature1.embVecs.length; i++) {
    const combinedVScore = combinedVScores[i]
    const threshold = thresholds[i]

    const embVec1 = feature1.embVecs[i]
    const embVec2 = feature2.embVecs[i]
    let sScore = getEmbSimilarityScore(embVec1, embVec2)
    partSScores.push(sScore)

    sScore = Math.max(sScore, 0)
    if (threshold !== 0.5) {
      sScore = logisticRemap(sScore, threshold)
    }

    totalSScore += combinedVScore * sScore
  }

  totalSScore /= vScoreSum

  return [totalSScore, partSScores] as const
}

export const getTotalVisibilityScore = (feature: Feature) => {
  return (
    feature.vScores.reduce((acc, score) => acc + score, 0) /
    feature.vScores.length
  )
}

export const getComparability = (
  featureToCompare: Feature,
  feature: Feature,
) => {
  return (
    feature.vScores.reduce((acc, score, i) => {
      const compareScore = featureToCompare.vScores[i]
      const unknownRatio = 1 - compareScore

      return (
        acc +
        unknownRatio +
        compareScore * (score / Math.max(score, compareScore))
      )
    }) / feature.vScores.length
  )
}
