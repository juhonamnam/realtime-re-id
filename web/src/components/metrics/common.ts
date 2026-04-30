import type { Feature } from "../type"

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

export const getEmbSimilarityScore = (
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

export const prepareForSimilarityMetric = (
  feature1: Feature,
  feature2: Feature,
) => {
  const vScores = []
  const partSScores = []

  for (let i = 0; i < feature1.vScores.length; i++) {
    vScores.push(Math.min(feature1.vScores[i], feature2.vScores[i]))

    const embVec1 = feature1.embVecs[i]
    const embVec2 = feature2.embVecs[i]
    partSScores.push(getEmbSimilarityScore(embVec1, embVec2))
  }

  return { vScores, partSScores }
}
