import { Cam } from "./cam"
import { CamSettings } from "./camSettings"
import { useCamData } from "./hook"
import { CamWrapper } from "./wrapper"
import { CamProvider } from "./provider"

const exportDefault = {
  useCamData,
  CamProvider,
  CamWrapper,
  Cam,
  CamSettings,
}

export default exportDefault

export { useCamData, CamProvider, CamWrapper, Cam, CamSettings }
