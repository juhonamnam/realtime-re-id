export const COLOR_OF_UNMATCH = [255, 255, 0] as const
export const COLOR_OF_MATCH = [255, 0, 0] as const

export const getSimilarityColor = (
  similarityScore: number,
  threshold: number,
): [number, number, number] => {
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
): [number, number, number] => {
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

export const mergeColors = (
  ...colors: [number, number, number][]
): [number, number, number] => {
  const totalColors = colors.length
  const mergedColor = [0, 0, 0] as [number, number, number]

  for (const color of colors) {
    mergedColor[0] += color[0] / totalColors
    mergedColor[1] += color[1] / totalColors
    mergedColor[2] += color[2] / totalColors
  }

  return mergedColor
}
