import { useContext } from "react"
import { ReIDConfigContext } from "./context"

export const useReIDConfig = () => {
  const { pdModel, feModel, metricType, showComparsionDetail } =
    useContext(ReIDConfigContext)
  return { pdModel, feModel, metricType, showComparsionDetail }
}
