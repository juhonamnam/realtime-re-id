import { createContext, type RefObject } from "react"
import type {
  CamDataHandler,
  CamChange,
  CamStatusType,
  Clear,
  SetCamDataHandler,
  FlipChange,
} from "./type"

export const CamContext = createContext<{
  flipRef: RefObject<boolean>
  camDataHandlerRef: RefObject<CamDataHandler | null>
  setCamDataHandler: SetCamDataHandler
  clear: Clear
  camRef: RefObject<HTMLVideoElement>
  camStatusRef: RefObject<CamStatusType>
  flipChangeRef: RefObject<FlipChange>
  camChangeRef: RefObject<CamChange>
}>({
  flipRef: { current: false },
  camDataHandlerRef: { current: null },
  setCamDataHandler: () => {},
  clear: () => {},
  camRef: { current: {} as HTMLVideoElement },
  camStatusRef: {
    current: {
      show: false,
      predictCount: null,
      errorMessage: null,
      resolution: null,
    },
  },
  flipChangeRef: { current: { prev: false, new: false } },
  camChangeRef: { current: { prev: "", new: "" } },
})
