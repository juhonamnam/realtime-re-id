import type { Feature } from "../type"
import { getEmbSimilarityScore } from "./common"

const logisticRemap = (x: number, threshold: number): number => {
  if (x === 0) {
    return 0
  } else if (x === 1) {
    return 1
  } else if (threshold === 0.5) {
    return x
  }

  const logistic =
    1 /
    (1 +
      Math.exp(Math.log(threshold / (1 - threshold)) - Math.log(x / (1 - x))))
  return logistic
}

export const getWeightedAvgSimilarity = (
  feature1: Feature,
  feature2: Feature,
  thresholds: number[],
): [number, number[]] => {
  const combinedVScores: number[] = []
  for (let i = 0; i < feature1.vScores.length; i++) {
    combinedVScores.push(Math.min(feature1.vScores[i], feature2.vScores[i]))
  }

  const combinedVScoreSum = combinedVScores.reduce(
    (sum, score) => sum + score,
    0,
  )

  if (combinedVScoreSum === 0) {
    return [0, combinedVScores.map(() => 0)] as const
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
    sScore = logisticRemap(sScore, threshold)

    totalSScore += combinedVScore * sScore
  }

  totalSScore /= combinedVScoreSum

  return [totalSScore, partSScores]
}
