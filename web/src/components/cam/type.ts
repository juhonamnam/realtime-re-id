export type CamDataHandler = (input: HTMLVideoElement) => Promise<void>
export type CamStatusType = {
  predictCount: number | null
  errorMessage: string | null
  resolution: {
    width: number
    height: number
  } | null
}
export type DeviceInfo = {
  label: string
  value: string
}
