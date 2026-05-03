import type { Feature } from "../type"

export type PredictResult = {
  bboxes: [number, number, number, number][]
  features: Feature[]
  error?: string
}

/**
 * A wrapper class for managing the Re-ID Web Worker.
 * It provides a promise-based API for model loading, person detection,
 * and feature extraction.
 */
class ReidWorker {
  private worker: Worker
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

  private call(type: string, payload?: unknown, transfer?: Transferable[]) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return new Promise<any>((resolve) => {
      this.responseMap.set(type, resolve)
      this.worker.postMessage({ type, payload }, transfer || [])
    })
  }

  /**
   * Loads the person detection and feature extraction models into the worker.
   *
   * @param pdModelPath - The path to the person detection ONNX model.
   * @param feModelPath - The path to the feature extraction ONNX model.
   * @returns A promise that resolves when both models are loaded.
   * @throws Error if model loading fails.
   */
  async loadModels(pdModelPath: string, feModelPath: string) {
    const result = await this.call("load", { pdModelPath, feModelPath })
    if (!result.success) throw new Error(result.error)
  }

  /**
   * Performs person detection on the input image data.
   *
   * @param params - The detection parameters.
   * @param params.input - The input image data, its dimensions, and resize scale.
   * @param params.threshold - The confidence threshold for detections.
   * @returns A promise that resolves to an array of bounding boxes [x1, y1, x2, y2].
   * @throws Error if detection fails.
   */
  async detect(params: {
    input: { data: Float32Array; dims: number[]; resizeScale: number }
    threshold: number
  }): Promise<[number, number, number, number][]> {
    const transfer: Transferable[] = [params.input.data.buffer]
    const result = await this.call("detect", params, transfer)
    if (result.error) throw new Error(result.error)
    return result.bboxes
  }

  /**
   * Extracts Re-ID features from cropped person images.
   *
   * @param params - The extraction parameters.
   * @param params.input - The batch of cropped person images and its dimensions.
   * @returns A promise that resolves to an array of Feature objects.
   * @throws Error if feature extraction fails.
   */
  async extractFeatures(params: {
    input: { data: Float32Array; dims: number[] }
  }): Promise<Feature[]> {
    const transfer: Transferable[] = [params.input.data.buffer]

    const result = await this.call("extract_features", params, transfer)
    if (result.error) throw new Error(result.error)
    return result.features
  }

  /**
   * Terminates the underlying Web Worker.
   */
  terminate() {
    this.worker.terminate()
  }
}

export const reidWorker = new ReidWorker()
