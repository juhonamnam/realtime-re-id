import { useContext } from "react"
import { CamStatus } from "./camStatus"
import { CamContext } from "./context"

export const Cam = () => {
  const { camRef } = useContext(CamContext)

  return (
    <>
      <video width="100%" height="100%" autoPlay ref={camRef} />
      <CamStatus />
    </>
  )
}
