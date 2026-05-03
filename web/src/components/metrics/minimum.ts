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

export const getMinimumSimilarity = (
  vScores: number[],
  partSScores: number[],
  partThresholds: number[],
): number => {
  let totalSScore = 1

  for (let i = 0; i < vScores.length; i++) {
    const vScore = vScores[i]
    let partSScore = partSScores[i]
    const partThreshold = partThresholds[i]

    partSScore = Math.max(partSScore, 0)
    partSScore = logisticRemap(partSScore, partThreshold)

    const partMinScore = 1 - vScore + vScore * partSScore
    totalSScore = Math.min(totalSScore, partMinScore)
  }

  return totalSScore
}
