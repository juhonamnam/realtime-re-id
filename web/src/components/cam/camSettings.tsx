import { useCallback, useContext, useEffect, useState } from "react"
import { CamContext } from "./context"
import { camLocalStorage } from "./localStorage"

const EMPTY_DEVICE = { label: "Not Selected", value: "" }

export const CamSettings = () => {
  const { flipChangeRef, camStatusRef, camChangeRef } = useContext(CamContext)
  const [devices, setDevices] = useState<{ label: string; value: string }[]>([
    EMPTY_DEVICE,
  ])

  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("")
  const [flip, setFlip] = useState<boolean>(false)
  const [showCamStatus, setShowCamStatus] = useState<boolean>(false)

  const refreshDevices = useCallback(async () => {
    const d: { label: string; value: string }[] = [EMPTY_DEVICE]

    try {
      await navigator.mediaDevices.getUserMedia({ video: true })

      const devices = await navigator.mediaDevices.enumerateDevices()

      devices.forEach((device) => {
        if (device.kind === "videoinput") {
          d.push({ value: device.deviceId, label: device.label })
        }
      })

      setDevices(d)
    } catch {
      setDevices(d)
    }
  }, [])

  useEffect(() => {
    refreshDevices()
    setSelectedDeviceId(camChangeRef.current.new)
    setFlip(flipChangeRef.current.new)
    setShowCamStatus(camStatusRef.current.show)
  }, [refreshDevices, camChangeRef, flipChangeRef, camStatusRef])

  return (
    <>
      <div className="d-flex align-items-center">
        Select Camera
        <button
          className="btn bi bi-arrow-clockwise"
          onClick={() => refreshDevices()}
        ></button>
      </div>
      <select
        className="form-select"
        onChange={(e) => {
          camChangeRef.current.new = e.currentTarget.value
          camLocalStorage.setSelectedDeviceId(e.currentTarget.value)
          setSelectedDeviceId(e.currentTarget.value)
        }}
        value={selectedDeviceId}
      >
        {devices.map((d) => (
          <option key={d.value} value={d.value}>
            {d.label}
          </option>
        ))}
      </select>
      <div className="form-check">
        <label className="form-check-label">
          <input
            className="form-check-input"
            type="checkbox"
            checked={flip}
            onChange={(e) => {
              flipChangeRef.current.new = e.currentTarget.checked
              camLocalStorage.setFlip(e.currentTarget.checked)
              setFlip(e.currentTarget.checked)
            }}
          />
          Horizontal Flip
        </label>
      </div>
      <div className="form-check">
        <label className="form-check-label">
          <input
            className="form-check-input"
            type="checkbox"
            checked={showCamStatus}
            onChange={(e) => {
              camStatusRef.current.show = e.currentTarget.checked
              camLocalStorage.setShowCamStatus(e.currentTarget.checked)
              setShowCamStatus(e.currentTarget.checked)
            }}
          />
          Show Status
        </label>
      </div>
    </>
  )
}
