import { useContext } from "react"
import { ReIDConfigContext } from "./context"
import { FE_MODELS, PD_MODELS, METRIC_TYPES } from "./modelInfo"

export const ReIDConfig = () => {
  const {
    setPdModel,
    setFeModel,
    setMetricType,
    setShowComparsionDetail,
    pdModel,
    feModel,
    metricType,
    showComparsionDetail,
  } = useContext(ReIDConfigContext)

  return (
    <>
      <div className="pt-1 pb-1">Person Detection Model</div>
      <select
        className="form-select"
        onChange={(e) => {
          setPdModel(PD_MODELS[parseInt(e.currentTarget.value)])
        }}
        value={pdModel.idx}
      >
        {PD_MODELS.map((model) => (
          <option key={model.name} value={model.idx}>
            {model.name}
          </option>
        ))}
      </select>
      <div className="pt-1 pb-1">Feature Extraction Model</div>
      <select
        className="form-select"
        onChange={(e) => {
          setFeModel(FE_MODELS[parseInt(e.currentTarget.value)])
        }}
        value={feModel.idx}
      >
        {FE_MODELS.map((model) => (
          <option key={model.name} value={model.idx}>
            {model.name}
          </option>
        ))}
      </select>
      <div className="pt-1 pb-1">Metric Type</div>
      <select
        className="form-select"
        onChange={(e) => {
          setMetricType(METRIC_TYPES[parseInt(e.currentTarget.value)])
        }}
        value={metricType.idx}
      >
        {METRIC_TYPES.map((type) => (
          <option key={type.name} value={type.idx}>
            {type.name}
          </option>
        ))}
      </select>
      <div className="form-check form-switch pt-2">
        <label className="form-check-label">
          <input
            className="form-check-input"
            type="checkbox"
            role="switch"
            checked={showComparsionDetail}
            onChange={(e) => setShowComparsionDetail(e.currentTarget.checked)}
          />
          Show Comparison Detail
        </label>
      </div>
    </>
  )
}
