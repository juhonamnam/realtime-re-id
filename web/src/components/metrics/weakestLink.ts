import type { Feature } from "../type"
import { getEmbSimilarityScore } from "./common"

const VALUE_AT_THRESHOLD = 0.7

const logisticRemap = (x: number, threshold: number, exp = 2): number => {
  if (x === 0) {
    return 0
  } else if (x === 1) {
    return 1
  }

  const logistic =
    1 /
    (1 +
      Math.exp(
        exp * Math.log(threshold / (1 - threshold)) +
          Math.log((1 - VALUE_AT_THRESHOLD) / VALUE_AT_THRESHOLD) -
          exp * Math.log(x / (1 - x)),
      ))
  return logistic
}

export const getWeakestLinkSimilarity = (
  feature1: Feature,
  feature2: Feature,
  thresholds: number[],
): [number, number[]] => {
  const combinedVScores: number[] = []
  for (let i = 0; i < feature1.vScores.length; i++) {
    combinedVScores.push(Math.min(feature1.vScores[i], feature2.vScores[i]))
  }

  let totalSScore = 1
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

    totalSScore *= 1 - combinedVScore + combinedVScore * sScore
  }

  return [totalSScore, partSScores]
}
