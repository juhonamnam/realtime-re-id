import { useContext, useEffect, useRef } from "react"
import { CamStatus } from "./camStatus"
import { CamContext } from "./context"
import type { CamDataHandler } from "./type"

const wait = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms))

export const Cam = () => {
  const { camRef, flip, selectedDeviceId, camDataHandler, camStatusRef } =
    useContext(CamContext)

  const flipRef = useRef(false)
  const flipChangedRef = useRef(false)

  const selectedDeviceIdRef = useRef("")
  const deviceChangedRef = useRef(false)

  const camDataHandlerRef = useRef<CamDataHandler | null>(null)

  useEffect(() => {
    flipRef.current = flip
    flipChangedRef.current = true
  }, [flip])

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceId
    deviceChangedRef.current = true
  }, [selectedDeviceId])

  useEffect(() => {
    camDataHandlerRef.current = camDataHandler
  }, [camDataHandler])

  useEffect(() => {
    let unmounted = false

    let frameId = 0

    if (flipRef.current) {
      camRef.current.style.transform = "scaleX(-1)"
    }

    const handle = async () => {
      if (unmounted) return

      if (flipChangedRef.current) {
        flipChangedRef.current = false
        if (flipRef.current) {
          camRef.current.style.transform = "scaleX(-1)"
        } else {
          camRef.current.style.transform = "none"
        }
      }

      if (deviceChangedRef.current) {
        deviceChangedRef.current = false
        try {
          if (selectedDeviceIdRef.current) {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: selectedDeviceIdRef.current },
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
  }, [camRef, camStatusRef])

  return (
    <>
      <video width="100%" height="100%" autoPlay ref={camRef} />
      <CamStatus />
    </>
  )
}
