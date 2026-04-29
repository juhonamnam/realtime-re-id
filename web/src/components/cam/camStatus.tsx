import { useContext, useEffect, useState } from "react"
import { CamContext } from "./context"

export type CamStatusComponent = typeof CamStatus

export const CamStatus = () => {
  const [fps, setFps] = useState<number | null>(null)
  const [resolution, setResolution] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const { showCamStatus, camStatusRef } = useContext(CamContext)

  useEffect(() => {
    const update = () => {
      setErrorMessage(camStatusRef.current.errorMessage)
      if (camStatusRef.current.resolution) {
        setResolution(
          `${camStatusRef.current.resolution.width}x${camStatusRef.current.resolution.height}`,
        )
      } else {
        setResolution(null)
      }

      if (camStatusRef.current.predictCount === null) {
        setFps(null)
      } else {
        setFps(camStatusRef.current.predictCount)
        camStatusRef.current.predictCount = 0
      }
    }
    update()
    const interval = setInterval(update, 1000)

    return () => {
      clearInterval(interval)
    }
  }, [camStatusRef])

  return (
    <div
      className={
        showCamStatus
          ? "text-light bg-dark position-absolute end-0 top-0 z-1 p-2"
          : "d-none"
      }
      style={{ "--bs-bg-opacity": 0.5 } as React.CSSProperties}
    >
      {resolution !== null && <div>Resolution: {resolution}</div>}
      {fps !== null && <div>Prediction Speed: {fps} FPS</div>}
      {errorMessage !== null && (
        <div className="text-danger">Error: {errorMessage}</div>
      )}
    </div>
  )
}
