const logisticRemap = (x: number, threshold: number): number => {
  if (x === 0) {
    return 0
  } else if (x === 1) {
    return 1
  } else if (threshold === 0.5) {
    return x
  }

  const logistic = 1 / (1 + (threshold / (1 - threshold)) * ((1 - x) / x))
  return logistic
}

export const getWeightedGeometricMeanSimilarity = (
  vScores: number[],
  partSScores: number[],
  partThresholds: number[],
): number => {
  const vScoreSum = vScores.reduce((sum, score) => sum + score, 0)

  if (vScoreSum === 0) {
    return 0
  }

  let totalLnSScore = 0

  for (let i = 0; i < vScores.length; i++) {
    const vScore = vScores[i]
    let partSScore = partSScores[i]
    const partThreshold = partThresholds[i]

    partSScore = Math.max(partSScore, 0)
    partSScore = logisticRemap(partSScore, partThreshold)

    totalLnSScore += vScore * Math.log(Math.max(partSScore, 1e-6))
  }

  const totalSScore = Math.exp(totalLnSScore / vScoreSum)

  return totalSScore
}
