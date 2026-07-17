import { useMemo, useState, type PropsWithChildren } from "react"
import { ReIDConfigContext } from "./context"
import { reIdConfigLocalStorage } from "./localStorage"
import {
  FE_MODELS,
  PD_MODELS,
  type FEModelInfo,
  type PDModelInfo,
} from "./modelInfo"

import { METRIC_OPTIONS, type MetricOption } from "../metrics"

/**
 * Provider component for the Re-ID configuration context.
 * It manages the state for person detection models, feature extraction models,
 * distance metrics, and visualization settings.
 *
 * @param props - The component props.
 * @param props.children - The child components that will have access to the context.
 * @returns A React functional component.
 */
export const ReIDConfigProvider = ({ children }: PropsWithChildren) => {
  const initialPdModel = useMemo(() => {
    const pdModelName = reIdConfigLocalStorage.getPDModel()
    const pdModel = PD_MODELS.find((model) => model.name === pdModelName)
    if (pdModel) {
      return pdModel
    } else {
      return PD_MODELS[0]
    }
  }, [])
  const initialFeModel = useMemo(() => {
    const feModelName = reIdConfigLocalStorage.getFEModel()
    const feModel = FE_MODELS.find((model) => model.name === feModelName)
    if (feModel) {
      return feModel
    } else {
      return FE_MODELS[0]
    }
  }, [])
  const initialMetric = useMemo(() => {
    const metricType = reIdConfigLocalStorage.getMetricType()

    const metric = METRIC_OPTIONS.find((option) => option.type === metricType)

    if (metric) {
      return metric
    } else {
      return METRIC_OPTIONS[0]
    }
  }, [])
  const initialThreshold = useMemo(() => {
    return initialFeModel.optimalThresholds[initialMetric.type]
  }, [initialFeModel, initialMetric])
  const initialShowComparsionDetail = useMemo(() => {
    return reIdConfigLocalStorage.getShowComparsionDetail()
  }, [])

  const [pdModel, setPdModel_] = useState(initialPdModel)
  const [feModel, setFeModel_] = useState(initialFeModel)
  const [metric, setMetric_] = useState(initialMetric)
  const [threshold, setThreshold] = useState(initialThreshold)
  const [showComparsionDetail, setShowComparsionDetail_] = useState(
    initialShowComparsionDetail,
  )

  const setPdModel = (model: PDModelInfo) => {
    reIdConfigLocalStorage.setPDModel(model.name)
    setPdModel_(model)
  }

  const setFeModel = (model: FEModelInfo) => {
    reIdConfigLocalStorage.setFEModel(model.name)
    setFeModel_(model)
    setThreshold(model.optimalThresholds[metric.type])
  }

  const setMetric = (metric: MetricOption) => {
    reIdConfigLocalStorage.setMetricType(metric.type)
    setMetric_(metric)
    setThreshold(feModel.optimalThresholds[metric.type])
  }

  const setShowComparsionDetail = (show: boolean) => {
    reIdConfigLocalStorage.setShowComparsionDetail(show)
    setShowComparsionDetail_(show)
  }

  return (
    <ReIDConfigContext.Provider
      value={{
        pdModel,
        setPdModel,
        feModel,
        setFeModel,
        metric,
        setMetric,
        threshold,
        setThreshold,
        showComparsionDetail,
        setShowComparsionDetail,
      }}
    >
      {children}
    </ReIDConfigContext.Provider>
  )
}
