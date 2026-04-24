const PREFIX = "cam-"

class CamLocalStorage {
  storage: Storage
  constructor() {
    this.storage = window.localStorage
  }

  setSelectedDeviceId(deviceId: string) {
    this.storage.setItem(`${PREFIX}selectedDeviceId`, deviceId)
  }

  getSelectedDeviceId() {
    return this.storage.getItem(`${PREFIX}selectedDeviceId`)
  }

  setShowCamStatus(show: boolean) {
    this.storage.setItem(`${PREFIX}showCamStatus`, String(show))
  }

  getShowCamStatus() {
    return this.storage.getItem(`${PREFIX}showCamStatus`) === "true"
  }

  setFlip(flip: boolean) {
    this.storage.setItem(`${PREFIX}flip`, String(flip))
  }

  getFlip() {
    return this.storage.getItem(`${PREFIX}flip`) === "true"
  }
}

export const camLocalStorage = new CamLocalStorage()
