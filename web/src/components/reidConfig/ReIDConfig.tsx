import { useContext } from "react"
import { ReIDConfigContext } from "./context"
import { FE_MODELS, PD_MODELS } from "./modelInfo"
import { METRIC_OPTIONS } from "../metrics"

export const ReIDConfig = () => {
  const {
    setPdModel,
    setFeModel,
    setMetric,
    setThreshold,
    setShowComparsionDetail,
    pdModel,
    feModel,
    metric,
    threshold,
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
      <div className="pt-1 pb-1">Distance Metric</div>
      <select
        className="form-select"
        onChange={(e) => {
          setMetric(METRIC_OPTIONS[parseInt(e.currentTarget.value)])
        }}
        value={metric.idx}
      >
        {METRIC_OPTIONS.map((metricOption) => (
          <option key={metricOption.type} value={metricOption.idx}>
            {metricOption.type}
          </option>
        ))}
      </select>
      <div className="pt-1 pb-1">
        <span>
          Threshold: {threshold.toFixed(2)} (Optimal:{" "}
          {feModel.optimalThresholds[metric.type].toFixed(2)})
        </span>
        <button
          className="btn btn-sm bi bi-arrow-clockwise"
          onClick={() => setThreshold(feModel.optimalThresholds[metric.type])}
          title="Reset to Default"
        ></button>
      </div>
      <input
        type="range"
        className="form-range"
        min="0"
        max={feModel.optimalThresholds[metric.type] * 2}
        step="0.01"
        value={threshold}
        onChange={(e) => setThreshold(parseFloat(e.target.value))}
      />
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
