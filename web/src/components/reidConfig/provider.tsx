import { useMemo, useState, type PropsWithChildren } from "react"
import { ReIDConfigContext } from "./context"
import { reIdConfigLocalStorage } from "./localStorage"
import {
  FE_MODELS,
  METRIC_TYPES,
  PD_MODELS,
  type FEModelInfo,
  type MetricType,
  type PDModelInfo,
} from "./modelInfo"

/**
 * Provider component for the Re-ID configuration context.
 * It manages the state for person detection models, feature extraction models,
 * similarity metrics, and visualization settings.
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
  const initialMetricType = useMemo(() => {
    const metricTypeName = reIdConfigLocalStorage.getMetricType()

    const metricType = METRIC_TYPES.find((type) => type.name === metricTypeName)

    if (metricType) {
      return metricType
    } else {
      return METRIC_TYPES[0]
    }
  }, [])
  const initialSimilarityThreshold = useMemo(() => {
    return initialFeModel.defaultSimilarityThresholds[initialMetricType.name]
  }, [initialFeModel, initialMetricType])
  const initialShowComparsionDetail = useMemo(() => {
    return reIdConfigLocalStorage.getShowComparsionDetail()
  }, [])

  const [pdModel, setPdModel_] = useState(initialPdModel)
  const [feModel, setFeModel_] = useState(initialFeModel)
  const [metricType, setMetricType_] = useState(initialMetricType)
  const [similarityThreshold, setSimilarityThreshold] = useState(
    initialSimilarityThreshold,
  )
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
    setSimilarityThreshold(model.defaultSimilarityThresholds[metricType.name])
  }

  const setMetricType = (metricType: MetricType) => {
    reIdConfigLocalStorage.setMetricType(metricType.name)
    setMetricType_(metricType)
    setSimilarityThreshold(feModel.defaultSimilarityThresholds[metricType.name])
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
        metricType,
        setMetricType,
        similarityThreshold,
        setSimilarityThreshold,
        showComparsionDetail,
        setShowComparsionDetail,
      }}
    >
      {children}
    </ReIDConfigContext.Provider>
  )
}
