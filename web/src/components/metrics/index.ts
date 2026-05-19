import { prepareForSimilarityMetric } from "./common"
import type { Feature } from "../type"

export { getProductSimilarity } from "../metrics/product"
export { getWeightedMeanSimilarity } from "../metrics/weightedMean"
export { getWeightedGeometricMeanSimilarity } from "../metrics/weightedGeometricMean"
export { getMinimumSimilarity } from "../metrics/minimum"

export const buildSimilarityFunc = (
  similarityFunc: (
    vScores: number[],
    partSScores: number[],
    partThresholds: number[],
  ) => number,
) => {
  return (
    features1: Feature,
    features2: Feature,
    partThresholds: number[],
  ): [number, number[]] => {
    const { vScores, partSScores } = prepareForSimilarityMetric(
      features1,
      features2,
    )
    return [similarityFunc(vScores, partSScores, partThresholds), partSScores]
  }
}
