import {
  type RefObject,
  type PropsWithChildren,
  useRef,
  useMemo,
  useState,
  useCallback,
  useEffect,
} from "react"
import { CamContext } from "./context"
import { camLocalStorage } from "./localStorage"
import type { CamDataHandler, CamStatusType } from "./type"
import { EMPTY_DEVICE, getAllDevices } from "./devices"

export const CamProvider = ({ children }: PropsWithChildren) => {
  const initialFlip = useMemo(() => camLocalStorage.getFlip(), [])
  const initialShowCamStatus = useMemo(
    () => camLocalStorage.getShowCamStatus(),
    [],
  )

  const [flip, setFlip_] = useState(initialFlip)
  const [showCamStatus, setShowCamStatus_] = useState(initialShowCamStatus)
  const [devices, setDevices] = useState([EMPTY_DEVICE])
  const [selectedDeviceId, setSelectedDeviceId_] = useState("")
  const [camDataHandler, setCamDataHandler_] = useState<CamDataHandler | null>(
    null,
  )

  const setShowCamStatus = (s: boolean) => {
    setShowCamStatus_(s)
    camLocalStorage.setShowCamStatus(s)
  }

  const setFilp = (f: boolean) => {
    setFlip_(f)
    camLocalStorage.setFlip(f)
  }

  const setSelectedDeviceId = (deviceId: string) => {
    setSelectedDeviceId_(deviceId)
    camLocalStorage.setSelectedDeviceId(deviceId)
  }

  const setCamDataHandler = useCallback((handler: CamDataHandler | null) => {
    setCamDataHandler_(() => handler)
  }, [])

  const camRef = useRef<HTMLVideoElement>(null) as RefObject<HTMLVideoElement>

  const camStatusRef = useRef<CamStatusType>({
    predictCount: null,
    errorMessage: null,
    resolution: null,
  })

  useEffect(() => {
    ;(async () => {
      const allDevices = await getAllDevices()
      setDevices(allDevices)

      const initialSelectedDeviceId = camLocalStorage.getSelectedDeviceId()
      if (
        initialSelectedDeviceId &&
        allDevices.some((d) => d.value === initialSelectedDeviceId)
      ) {
        setSelectedDeviceId(initialSelectedDeviceId)
      }
    })()
  }, [])

  return (
    <CamContext.Provider
      value={{
        flip,
        setFlip: setFilp,
        camDataHandler,
        setCamDataHandler,
        showCamStatus,
        setShowCamStatus,
        devices,
        setDevices,
        selectedDeviceId,
        setSelectedDeviceId,
        camRef,
        camStatusRef,
      }}
    >
      {children}
    </CamContext.Provider>
  )
}
