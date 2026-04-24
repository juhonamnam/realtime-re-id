export type CamDataHandler = (input: HTMLVideoElement) => Promise<void>
export type SetCamDataHandler = (cameraHandler: CamDataHandler) => void
export type Clear = () => void
export type CamStatusType = {
  show: boolean
  predictCount: number | null
  errorMessage: string | null
  resolution: {
    width: number
    height: number
  } | null
}
export type FlipChange = {
  prev: boolean
  new: boolean
}
export type CamChange = {
  prev: string
  new: string
}
