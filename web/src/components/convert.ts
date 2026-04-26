import * as ort from "onnxruntime-web"

// Keep ratio with padding
export const captureVideo = async (
  video: HTMLVideoElement,
  targetSize: readonly [number, number], // [height, width]
) => {
  const [targetHeight, targetWidth] = targetSize
  const [videoWidth, videoHeight] = [video.videoWidth, video.videoHeight]

  const capture = new OffscreenCanvas(videoWidth, videoHeight)
  const captureCtx = capture.getContext("2d")
  if (!captureCtx) {
    throw new Error("Failed to get 2d context")
  }
  captureCtx.drawImage(video, 0, 0, videoWidth, videoHeight)

  const canvForConvert = new OffscreenCanvas(targetWidth, targetHeight)
  const ctx = canvForConvert.getContext("2d")
  if (!ctx) {
    throw new Error("Failed to get 2d context")
  }

  const yResizeScale = targetHeight / videoHeight
  const xResizeScale = targetWidth / videoWidth

  const resizeScale = Math.min(yResizeScale, xResizeScale)

  ctx.drawImage(
    capture,
    0,
    0,
    videoWidth * resizeScale,
    videoHeight * resizeScale,
  )

  const { data, width, height } = ctx.getImageData(
    0,
    0,
    targetWidth,
    targetHeight,
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

  return { tensor, canvas: capture, resizeScale }
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
