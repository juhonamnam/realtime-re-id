import { createContext } from "react"
import {
  FE_MODELS,
  METRIC_TYPES,
  PD_MODELS,
  type FEModelInfo,
  type MetricType,
  type PDModelInfo,
} from "./modelInfo"

export const ReIDConfigContext = createContext<{
  feModel: FEModelInfo
  setFeModel: (model: FEModelInfo) => void
  pdModel: PDModelInfo
  setPdModel: (model: PDModelInfo) => void
  metricType: MetricType
  setMetricType: (metricType: MetricType) => void
  similarityThreshold: number
  setSimilarityThreshold: (threshold: number) => void
  showComparsionDetail: boolean
  setShowComparsionDetail: (show: boolean) => void
}>({
  feModel: FE_MODELS[0],
  setFeModel: () => {},
  pdModel: PD_MODELS[0],
  setPdModel: () => {},
  metricType: METRIC_TYPES[0],
  setMetricType: () => {},
  similarityThreshold: 0.64,
  setSimilarityThreshold: () => {},
  showComparsionDetail: false,
  setShowComparsionDetail: () => {},
})
