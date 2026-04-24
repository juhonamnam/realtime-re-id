import {
  type RefObject,
  type PropsWithChildren,
  useEffect,
  useRef,
  useMemo,
} from "react"
import { CamContext } from "./context"
import { camLocalStorage } from "./localStorage"
import type {
  CamDataHandler,
  SetCamDataHandler,
  Clear,
  CamStatusType,
  CamChange,
} from "./type"

const wait = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms))

export const CamProvider = ({ children }: PropsWithChildren) => {
  const camDataHandlerRef = useRef<CamDataHandler | null>(null)
  const camRef = useRef<HTMLVideoElement>(null) as RefObject<HTMLVideoElement>
  const initialFlip = useMemo(() => camLocalStorage.getFlip(), [])
  const flipRef = useRef(initialFlip)
  const flipChangeRef = useRef({
    prev: initialFlip,
    new: initialFlip,
  })
  const setCamDataHandler: SetCamDataHandler = (p) => {
    camDataHandlerRef.current = p
  }

  const clear: Clear = () => {
    camDataHandlerRef.current = null
  }

  const camStatusRef = useRef<CamStatusType>({
    show: camLocalStorage.getShowCamStatus(),
    predictCount: null,
    errorMessage: null,
    resolution: null,
  })

  const camChangeRef = useRef<CamChange>({
    prev: "",
    new: "",
  })

  useEffect(() => {
    let unmounted = false

    let frameId = 0

    if (flipRef.current) {
      camRef.current.style.transform = "scaleX(-1)"
    }

    ;(async () => {
      const storedDeviceId = camLocalStorage.getSelectedDeviceId()

      try {
        await navigator.mediaDevices.getUserMedia({ video: true })

        let devices = await navigator.mediaDevices.enumerateDevices()

        devices = devices.filter((device) => device.kind === "videoinput")

        if (devices.length) {
          if (
            storedDeviceId &&
            devices.some((d) => d.deviceId === storedDeviceId)
          ) {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: storedDeviceId },
            })
            camRef.current.srcObject = stream
            camChangeRef.current.prev = storedDeviceId
            camChangeRef.current.new = storedDeviceId
            camLocalStorage.setSelectedDeviceId(storedDeviceId)
          } else {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: devices[0].deviceId },
            })
            camRef.current.srcObject = stream
            camChangeRef.current.prev = devices[0].deviceId
            camChangeRef.current.new = devices[0].deviceId
            camLocalStorage.setSelectedDeviceId(devices[0].deviceId)
          }
        } else {
          camRef.current.srcObject = null
        }
      } catch {
        camRef.current.srcObject = null
      }
    })()

    const handle = async () => {
      if (unmounted) return

      if (flipChangeRef.current.prev !== flipChangeRef.current.new) {
        flipChangeRef.current.prev = flipChangeRef.current.new
        flipRef.current = flipChangeRef.current.new
        if (flipRef.current) {
          camRef.current.style.transform = "scaleX(-1)"
        } else {
          camRef.current.style.transform = "none"
        }
      }

      if (camChangeRef.current.prev !== camChangeRef.current.new) {
        camChangeRef.current.prev = camChangeRef.current.new
        try {
          if (camChangeRef.current.new) {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: camChangeRef.current.new },
            })
            camRef.current.srcObject = stream
          } else {
            camRef.current.srcObject = null
          }
        } catch {
          camRef.current.srcObject = null
        }
      }

      const handler = camDataHandlerRef.current
      const readyState = camRef.current.readyState

      if (readyState === 4) {
        camStatusRef.current.resolution = {
          width: camRef.current.videoWidth,
          height: camRef.current.videoHeight,
        }
        if (handler) {
          try {
            await handler(camRef.current)
            if (camStatusRef.current.predictCount === null)
              camStatusRef.current.predictCount = 1
            else camStatusRef.current.predictCount++
            camStatusRef.current.errorMessage = null
          } catch (e) {
            if (handler === camDataHandlerRef.current) {
              camStatusRef.current.predictCount = null
              console.error(e)
              if (e instanceof Error) {
                camStatusRef.current.errorMessage = e.message
              } else {
                camStatusRef.current.errorMessage = String(e)
              }
              await wait(5000)
            }
          }
        } else {
          camStatusRef.current.errorMessage = null
          camStatusRef.current.predictCount = null
        }
      } else {
        camStatusRef.current.resolution = null
        camStatusRef.current.errorMessage = null
        camStatusRef.current.predictCount = null
      }
      frameId = requestAnimationFrame(handle)
    }
    frameId = requestAnimationFrame(handle)

    return () => {
      unmounted = true
      cancelAnimationFrame(frameId)
    }
  }, [])

  return (
    <CamContext.Provider
      value={{
        flipRef,
        camDataHandlerRef,
        setCamDataHandler,
        clear,
        camRef,
        camStatusRef,
        flipChangeRef,
        camChangeRef,
      }}
    >
      {children}
    </CamContext.Provider>
  )
}
