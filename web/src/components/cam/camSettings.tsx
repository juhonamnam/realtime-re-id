import { useCallback, useContext, useEffect } from "react"
import { CamContext } from "./context"
import { getAllDevices } from "./devices"

export const CamSettings = () => {
  const {
    flip,
    setFlip,
    showCamStatus,
    setShowCamStatus,
    devices,
    setDevices,
    selectedDeviceId,
    setSelectedDeviceId,
  } = useContext(CamContext)

  const refreshDevices = useCallback(async () => {
    const allDevices = await getAllDevices()
    setDevices(allDevices)
  }, [setDevices])

  useEffect(() => {
    refreshDevices()
  }, [refreshDevices])

  return (
    <>
      <div className="d-flex align-items-center">
        Select Camera
        <button
          className="btn btn-sm bi bi-arrow-clockwise"
          onClick={() => refreshDevices()}
        ></button>
      </div>
      <select
        className="form-select"
        onChange={(e) => {
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
      <div className="form-check form-switch pt-2">
        <label className="form-check-label">
          <input
            className="form-check-input"
            type="checkbox"
            checked={flip}
            onChange={(e) => {
              setFlip(e.currentTarget.checked)
            }}
          />
          Horizontal Flip
        </label>
      </div>
      <div className="form-check form-switch pt-2">
        <label className="form-check-label">
          <input
            className="form-check-input"
            type="checkbox"
            checked={showCamStatus}
            onChange={(e) => {
              setShowCamStatus(e.currentTarget.checked)
            }}
          />
          Show Status
        </label>
      </div>
    </>
  )
}
