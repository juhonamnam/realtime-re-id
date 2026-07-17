export const getDistMean = (
  embVecs1: Float32Array[],
  embVecs2: Float32Array[],
): [number[], number] => {
  let finalDist = 0
  const partDists = []
  for (let i = 0; i < embVecs1.length; i++) {
    let partDist = 0
    for (let j = 0; j < embVecs1[i].length; j++) {
      partDist += (embVecs1[i][j] - embVecs2[i][j]) ** 2
    }

    partDist = Math.sqrt(partDist)
    partDists.push(partDist)

    finalDist += partDist
  }

  return [partDists, finalDist / embVecs1.length]
}
