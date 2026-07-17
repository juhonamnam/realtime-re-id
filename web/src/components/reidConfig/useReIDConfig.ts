import { useContext } from "react"
import { ReIDConfigContext } from "./context"

/**
 * Custom hook to access the Re-ID configuration state.
 *
 * @returns An object containing the current person detection model,
 * feature extraction model, metric, threshold, and
 * a boolean indicating if comparison details should be shown.
 */
export const useReIDConfig = () => {
  const { pdModel, feModel, metric, threshold, showComparsionDetail } =
    useContext(ReIDConfigContext)
  return {
    pdModel,
    feModel,
    metric,
    threshold,
    showComparsionDetail,
  }
}
