import { LOCAL_STORAGE_PREFIX } from "./const"

class ReIDLocalStorage {
  storage: Storage
  constructor() {
    this.storage = window.localStorage
  }

  setPDModel(modelName: string) {
    this.storage.setItem(`${LOCAL_STORAGE_PREFIX}pdModel`, modelName)
  }

  getPDModel() {
    return this.storage.getItem(`${LOCAL_STORAGE_PREFIX}pdModel`)
  }

  setFEModel(modelName: string) {
    this.storage.setItem(`${LOCAL_STORAGE_PREFIX}feModel`, modelName)
  }

  getFEModel() {
    return this.storage.getItem(`${LOCAL_STORAGE_PREFIX}feModel`)
  }
}

export const reIdLocalStorage = new ReIDLocalStorage()
