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
      (threshold / (1 - threshold)) ** exp *
        ((1 - VALUE_AT_THRESHOLD) / VALUE_AT_THRESHOLD) *
        ((1 - x) / x) ** exp)
  return logistic
}

export const getProductSimilarity = (
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

    totalSScore *= 1 - vScore + vScore * partSScore
  }

  return totalSScore
}
