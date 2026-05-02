/**
 * Captures the current frame from a video element and converts it into a Float32Array
 * suitable for ONNX model input. It maintains the aspect ratio by applying letterbox padding.
 *
 * @param video - The source HTMLVideoElement to capture from.
 * @param targetSize - The target dimensions [height, width] for the model input.
 * @returns An object containing the processed data, dimensions, the capture canvas, and the resize scale used.
 */
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

  return {
    data: float32Data,
    dims: [1, 3, height, width],
    canvas: capture,
    resizeScale,
  }
}

/**
 * Crops specific regions (bounding boxes) from a canvas and converts them into a
 * single Float32Array buffer for batch processing by an ONNX model.
 * It maintains the aspect ratio of each crop by applying letterbox padding.
 *
 * @param canvas - The source OffscreenCanvas to crop from.
 * @param bboxes - An array of bounding boxes, each defined as [x1, y1, x2, y2].
 * @param targetSize - The target dimensions [height, width] for each cropped image.
 * @returns An object containing the batch-processed data and its dimensions.
 */
export const cropCanvasToBuffer = async (
  canvas: OffscreenCanvas,
  bboxes: [number, number, number, number][],
  targetSize: readonly [number, number], // [height, width]
) => {
  const [targetHeight, targetWidth] = targetSize

  const canvasForCrop = new OffscreenCanvas(targetWidth, targetHeight)
  const ctx = canvasForCrop.getContext("2d")
  if (!ctx) {
    throw new Error("Failed to get 2d context")
  }

  const pixelCount = targetWidth * targetHeight
  const float32Data = new Float32Array(bboxes.length * 3 * pixelCount)

  for (let batch = 0; batch < bboxes.length; batch++) {
    const [x1, y1, x2, y2] = bboxes[batch]
    const width = x2 - x1
    const height = y2 - y1

    const yResizeScale = targetHeight / height
    const xResizeScale = targetWidth / width

    const scale = Math.min(yResizeScale, xResizeScale)

    let topPadding, leftPadding

    if (yResizeScale < xResizeScale) {
      topPadding = 0
      leftPadding = (1 - yResizeScale / xResizeScale) * (targetWidth / 2)
    } else {
      leftPadding = 0
      topPadding = (1 - xResizeScale / yResizeScale) * (targetHeight / 2)
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

    const { data } = ctx.getImageData(0, 0, targetWidth, targetHeight)

    // Debug
    // const blob = await canvasForCrop.convertToBlob()
    // const url = URL.createObjectURL(blob)
    // console.log(url)

    for (let i = 0; i < pixelCount; i++) {
      const offset = i * 4
      float32Data[i + pixelCount * 3 * batch] = data[offset] / 255
      float32Data[i + pixelCount * (3 * batch + 1)] = data[offset + 1] / 255
      float32Data[i + pixelCount * (3 * batch + 2)] = data[offset + 2] / 255
    }
  }

  return {
    data: float32Data,
    dims: [bboxes.length, 3, targetHeight, targetWidth],
  }
}
