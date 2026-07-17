/**
 * @file reid.worker.ts
 * @description Web Worker for running ONNX models in the background.
 * It handles model loading, person detection (PD), and feature extraction (FE).
 */
import * as ort from "onnxruntime-web"

let pdSession: ort.InferenceSession | null = null
let feSession: ort.InferenceSession | null = null

let currentPdModelPath: string | null = null
let currentFeModelPath: string | null = null

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data

  switch (type) {
    case "load": {
      const { pdModelPath, feModelPath, feModelExternalData } = payload
      try {
        if (currentPdModelPath !== pdModelPath) {
          try {
            if (pdSession) await pdSession.release()
          } catch (error) {
            console.warn("Failed to release PD session:", error)
          }

          pdSession = null
          currentPdModelPath = pdModelPath

          try {
            pdSession = await ort.InferenceSession.create(pdModelPath)
          } catch (error) {
            currentPdModelPath = null
            throw error
          }
        }

        if (currentFeModelPath !== feModelPath) {
          try {
            if (feSession) await feSession.release()
          } catch (error) {
            console.warn("Failed to release FE session:", error)
          }

          feSession = null
          currentFeModelPath = feModelPath

          try {
            feSession = await ort.InferenceSession.create(feModelPath, {
              externalData: feModelExternalData,
            })
          } catch (error) {
            currentFeModelPath = null
            throw error
          }
        }

        self.postMessage({ type: "load", payload: { success: true } })
      } catch (error) {
        self.postMessage({
          type: "load",
          payload: { success: false, error: (error as Error).message },
        })
      }
      break
    }
    case "detect": {
      const { input, threshold } = payload

      try {
        if (!pdSession) throw new Error("PD Session not initialized")

        const { data, dims, resizeScale } = input
        const tensor = new ort.Tensor("float32", data, dims)
        const feeds = { [pdSession.inputNames[0]]: tensor }
        const result = await pdSession.run(feeds)
        const output = result[pdSession.outputNames[0]]
        const outputData = output.data as Float32Array
        const stride = output.dims[2]

        const bboxes: [number, number, number, number][] = []

        for (let i = 0; i < outputData.length; i += stride) {
          if (outputData[i + 4] < threshold) continue
          bboxes.push([
            outputData[i] / resizeScale,
            outputData[i + 1] / resizeScale,
            outputData[i + 2] / resizeScale,
            outputData[i + 3] / resizeScale,
          ])
        }
        output.dispose()
        tensor.dispose()

        self.postMessage({ type: "detect", payload: { bboxes } })
      } catch (error) {
        self.postMessage({
          type: "detect",
          payload: { error: (error as Error).message },
        })
      }

      break
    }

    case "extract_features": {
      const { input } = payload

      try {
        if (!feSession) throw new Error("FE Session not initialized")
        let features = []

        const { data, dims } = input
        const tensor = new ort.Tensor("float32", data, dims)
        const feeds = { [feSession.inputNames[0]]: tensor }
        const result = await feSession.run(feeds)

        const vScore = result[feSession.outputNames[0]]
        const embVec = result[feSession.outputNames[1]]

        const vScoreData = vScore.data as Float32Array
        const embVecData = embVec.data as Float32Array

        const batch = embVec.dims[0]
        const segments = embVec.dims[1]
        const featureNum = embVec.dims[2]

        features = []
        for (let i = 0; i < batch; i++) {
          const embVecs = []
          const vScores = vScoreData.slice(i * segments, (i + 1) * segments)
          for (let j = 0; j < segments; j++) {
            embVecs.push(
              embVecData.slice(
                (i * segments + j) * featureNum,
                (i * segments + j + 1) * featureNum,
              ),
            )
          }
          features.push({ embVecs, vScores })
        }

        vScore.dispose()
        embVec.dispose()
        tensor.dispose()

        self.postMessage({ type: "extract_features", payload: { features } })
      } catch (error) {
        self.postMessage({
          type: "extract_features",
          payload: { error: (error as Error).message },
        })
      }
      break
    }
  }
}
