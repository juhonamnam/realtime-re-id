import type { Feature } from "../type"

export type PredictResult = {
  bboxes: [number, number, number, number][]
  features: Feature[]
  error?: string
}

class ReidWorker {
  private worker: Worker
  private responseMap: Map<string, (data: any) => void> = new Map()

  constructor() {
    this.worker = new Worker(new URL("./reid.worker.ts", import.meta.url), {
      type: "module",
    })

    this.worker.onmessage = (e) => {
      const { type, payload } = e.data
      const resolve = this.responseMap.get(type)
      if (resolve) {
        resolve(payload)
        this.responseMap.delete(type)
      }
    }
  }

  private call(
    type: string,
    payload?: any,
    transfer?: Transferable[],
  ): Promise<any> {
    return new Promise((resolve) => {
      this.responseMap.set(type, resolve)
      this.worker.postMessage({ type, payload }, transfer || [])
    })
  }

  async loadModels(pdModelPath: string, feModelPath: string) {
    const result = await this.call("load", { pdModelPath, feModelPath })
    if (!result.success) throw new Error(result.error)
  }

  async detect(params: {
    input: { data: Float32Array; dims: number[]; resizeScale: number }
    threshold: number
  }): Promise<[number, number, number, number][]> {
    const transfer: Transferable[] = [params.input.data.buffer]
    const result = await this.call("detect", params, transfer)
    if (result.error) throw new Error(result.error)
    return result.bboxes
  }

  async extractFeatures(params: {
    input: { data: Float32Array; dims: number[] }
  }): Promise<Feature[]> {
    const transfer: Transferable[] = [params.input.data.buffer]

    const result = await this.call("extract_features", params, transfer)
    if (result.error) throw new Error(result.error)
    return result.features
  }

  terminate() {
    this.worker.terminate()
  }
}

export const reidWorker = new ReidWorker()
