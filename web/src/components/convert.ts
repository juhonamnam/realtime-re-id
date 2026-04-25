import * as ort from "onnxruntime-web"

// Keep ratio with padding
export const videoToTensor = async (
  video: HTMLVideoElement,
  resize: readonly [number, number], // [height, width]
) => {
  const [resizeHeight, resizeWidth] = resize
  const [videoWidth, videoHeight] = [video.videoWidth, video.videoHeight]

  const canvas = new OffscreenCanvas(resizeWidth, resizeHeight)
  const ctx = canvas.getContext("2d")
  if (!ctx) {
    throw new Error("Failed to get 2d context")
  }

  const yResizeRatio = resizeHeight / videoHeight
  const xResizeRatio = resizeWidth / videoWidth

  const scale = Math.min(yResizeRatio, xResizeRatio)

  let padding

  if (yResizeRatio < xResizeRatio) {
    padding = {
      y: 0,
      x: 1 - yResizeRatio / xResizeRatio,
    }
  } else {
    padding = {
      y: 1 - xResizeRatio / yResizeRatio,
      x: 0,
    }
  }

  ctx.drawImage(video, 0, 0, videoWidth * scale, videoHeight * scale)

  const { data, width, height } = ctx.getImageData(
    0,
    0,
    resizeWidth,
    resizeHeight,
  )

  const pixelCount = width * height
  const float32Data = new Float32Array(pixelCount * 3)

  for (let i = 0; i < pixelCount; i++) {
    const offset = i * 4
    float32Data[i] = data[offset] / 255
    float32Data[i + pixelCount] = data[offset + 1] / 255
    float32Data[i + pixelCount * 2] = data[offset + 2] / 255
  }

  const tensor = new ort.Tensor("float32", float32Data, [1, 3, height, width])

  return { tensor, padding, canvas }
}

// Keep ratio with letterbox padding
export const cropCanvasToTensor = async (
  canvas: OffscreenCanvas,
  bboxes: [number, number, number, number][],
  resize: readonly [number, number], // [height, width]
) => {
  const [resizeHeight, resizeWidth] = resize

  const canvasForCrop = new OffscreenCanvas(resizeWidth, resizeHeight)
  const ctx = canvasForCrop.getContext("2d")
  if (!ctx) {
    throw new Error("Failed to get 2d context")
  }

  const pixelCount = resizeWidth * resizeHeight
  const float32Data = new Float32Array(bboxes.length * 3 * pixelCount)

  for (let batch = 0; batch < bboxes.length; batch++) {
    const [x1, y1, x2, y2] = bboxes[batch]
    const width = x2 - x1
    const height = y2 - y1

    const yResizeRatio = resizeHeight / height
    const xResizeRatio = resizeWidth / width

    const scale = Math.min(yResizeRatio, xResizeRatio)

    let topPadding, leftPadding

    // ctx.drawImage(video, 0, 0, videoWidth * scale, videoHeight * scale)

    if (yResizeRatio < xResizeRatio) {
      topPadding = 0
      leftPadding = (1 - yResizeRatio / xResizeRatio) * (resizeWidth / 2)
    } else {
      leftPadding = 0
      topPadding = (1 - xResizeRatio / yResizeRatio) * (resizeHeight / 2)
    }

    ctx.drawImage(
      canvas,
      x1,
      y1,
      width,
      height,
      leftPadding,
      topPadding,
      width * scale,
      height * scale,
    )

    const { data } = ctx.getImageData(0, 0, resizeWidth, resizeHeight)

    for (let i = 0; i < pixelCount; i++) {
      const offset = i * 4
      float32Data[i + pixelCount * 3 * batch] = data[offset] / 255
      float32Data[i + pixelCount * (3 * batch + 1)] = data[offset + 1] / 255
      float32Data[i + pixelCount * (3 * batch + 2)] = data[offset + 2] / 255
    }
  }

  const tensor = new ort.Tensor("float32", float32Data, [
    bboxes.length,
    3,
    resizeHeight,
    resizeWidth,
  ])

  return tensor
}
