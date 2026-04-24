import { FE_MODELS, PD_MODELS } from "./const"

export const ModelSelect = ({
  pdModel,
  feModel,
  setPdModel,
  setFeModel,
}: {
  pdModel?: string
  feModel?: string
  setPdModel: (model: string) => void
  setFeModel: (model: string) => void
}) => {
  return (
    <>
      <div className="pt-1 pb-1">Person Detection Model</div>
      <select
        className="form-select"
        onChange={(e) => {
          setPdModel(e.currentTarget.value)
        }}
        value={pdModel}
      >
        {Object.keys(PD_MODELS).map((type) => (
          <option key={type} value={type}>
            {type}
          </option>
        ))}
      </select>
      <div className="pt-1 pb-1">Feature Extraction Model</div>
      <select
        className="form-select"
        onChange={(e) => {
          setFeModel(e.currentTarget.value)
        }}
        value={feModel}
      >
        {Object.keys(FE_MODELS).map((type) => (
          <option key={type} value={type}>
            {type}
          </option>
        ))}
      </select>
    </>
  )
}
