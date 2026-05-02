import { useContext } from "react"
import { ReIDConfigContext } from "./context"

export const useReIDConfig = () => {
  const {
    pdModel,
    feModel,
    metricType,
    similarityThreshold,
    showComparsionDetail,
  } = useContext(ReIDConfigContext)
  return {
    pdModel,
    feModel,
    metricType,
    similarityThreshold,
    showComparsionDetail,
  }
}
