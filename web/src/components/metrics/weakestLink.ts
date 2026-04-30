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
  vScores: number[],
  partSScores: number[],
  thresholds: number[],
): number => {
  let totalSScore = 1

  for (let i = 0; i < vScores.length; i++) {
    const vScore = vScores[i]
    let partSScore = partSScores[i]
    const threshold = thresholds[i]

    partSScore = Math.max(partSScore, 0)
    partSScore = logisticRemap(partSScore, threshold)

    totalSScore *= 1 - vScore + vScore * partSScore
  }

  return totalSScore
}
