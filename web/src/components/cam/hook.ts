import { useContext } from "react"
import { CamContext } from "./context"

export const useCamData = () => {
  const { setCamDataHandler, flip, camRef } = useContext(CamContext)
  return { setCamDataHandler, flip, camRef }
}
