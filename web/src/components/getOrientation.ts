import { OFFSET_REM } from "./const"

export const getOrientation = (
  pixelsInRem: number,
  windowWidth: number,
  windowHeight: number,
  camWidth: number,
  camHeight: number,
) => {
  const landscapeRatio = (windowWidth - pixelsInRem * OFFSET_REM) / windowHeight
  const portraitRatio = windowWidth / (windowHeight - pixelsInRem * OFFSET_REM)

  const camRatio = camWidth / camHeight

  const [landDiff, portDiff] = [
    Math.abs(landscapeRatio - camRatio),
    Math.abs(portraitRatio - camRatio),
  ]

  if (portDiff > landDiff) {
    return "landscape"
  } else {
    return "portrait"
  }
}
