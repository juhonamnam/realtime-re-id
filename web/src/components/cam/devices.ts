import { logger } from "../logger"

export const EMPTY_DEVICE = { label: "Not Selected", value: "" }

export const getAllDevices = async () => {
  const d: { label: string; value: string }[] = [EMPTY_DEVICE]

  try {
    await navigator.mediaDevices.getUserMedia({ video: true })

    const devices = await navigator.mediaDevices.enumerateDevices()

    devices.forEach((device) => {
      if (device.kind === "videoinput") {
        d.push({ value: device.deviceId, label: device.label })
      }
    })

    return d
  } catch {
    logger("Error while fetching devices")
    return d
  }
}
