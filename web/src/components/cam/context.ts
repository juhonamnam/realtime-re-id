import { createContext, type RefObject } from "react"
import type { CamDataHandler, CamStatusType, DeviceInfo } from "./type"

export const CamContext = createContext<{
  flip: boolean
  setFlip: (flip: boolean) => void
  camDataHandler: CamDataHandler | null
  setCamDataHandler: (cameraHandler: CamDataHandler | null) => void
  showCamStatus: boolean
  setShowCamStatus: (show: boolean) => void
  devices: DeviceInfo[]
  setDevices: (devices: DeviceInfo[]) => void
  selectedDeviceId: string
  setSelectedDeviceId: (deviceId: string) => void
  camRef: RefObject<HTMLVideoElement>
  camStatusRef: RefObject<CamStatusType>
}>({
  flip: false,
  setFlip: () => {},
  camDataHandler: null,
  setCamDataHandler: () => {},
  showCamStatus: false,
  setShowCamStatus: () => {},
  devices: [],
  setDevices: () => {},
  selectedDeviceId: "",
  setSelectedDeviceId: () => {},
  camRef: { current: {} as HTMLVideoElement },
  camStatusRef: {
    current: {
      predictCount: null,
      errorMessage: null,
      resolution: null,
    },
  },
})
