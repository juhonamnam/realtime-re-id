import { createContext } from "react"
import {
  FE_MODELS,
  PD_MODELS,
  type FEModelInfo,
  type PDModelInfo,
} from "./modelInfo"

import { METRIC_OPTIONS, type MetricOption } from "../metrics"

export const ReIDConfigContext = createContext<{
  feModel: FEModelInfo
  setFeModel: (model: FEModelInfo) => void
  pdModel: PDModelInfo
  setPdModel: (model: PDModelInfo) => void
  metric: MetricOption
  setMetric: (metricOption: MetricOption) => void
  threshold: number
  setThreshold: (threshold: number) => void
  showComparsionDetail: boolean
  setShowComparsionDetail: (show: boolean) => void
}>({
  feModel: FE_MODELS[0],
  setFeModel: () => {},
  pdModel: PD_MODELS[0],
  setPdModel: () => {},
  metric: METRIC_OPTIONS[0],
  setMetric: () => {},
  threshold: 1,
  setThreshold: () => {},
  showComparsionDetail: false,
  setShowComparsionDetail: () => {},
})
