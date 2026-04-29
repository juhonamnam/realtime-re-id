export const PREFIX = "reid-"

class ReIDConfigLocalStorage {
  storage: Storage
  constructor() {
    this.storage = window.localStorage
  }

  setPDModel(modelName: string) {
    this.storage.setItem(`${PREFIX}pdModel`, modelName)
  }

  getPDModel() {
    return this.storage.getItem(`${PREFIX}pdModel`)
  }

  setFEModel(modelName: string) {
    this.storage.setItem(`${PREFIX}feModel`, modelName)
  }

  getFEModel() {
    return this.storage.getItem(`${PREFIX}feModel`)
  }

  setMetricType(metricTypeName: string) {
    this.storage.setItem(`${PREFIX}metricType`, metricTypeName)
  }

  getMetricType() {
    return this.storage.getItem(`${PREFIX}metricType`)
  }

  setShowComparsionDetail(show: boolean) {
    this.storage.setItem(
      `${PREFIX}showComparsionDetail`,
      show ? "true" : "false",
    )
  }

  getShowComparsionDetail() {
    const value = this.storage.getItem(`${PREFIX}showComparsionDetail`)
    return value === "true"
  }
}

export const reIdConfigLocalStorage = new ReIDConfigLocalStorage()
