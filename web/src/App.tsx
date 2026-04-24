import { useEffect, useMemo, useRef, useState } from "react"
import Cam from "./components/cam"
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap-icons/font/bootstrap-icons.css"
import { ReIdentification } from "./components/reIdentification"
import "./app.scss"
import { ModelSelect } from "./components/modelSelect"
import { reIdLocalStorage } from "./components/localStorage"
import { Navbar, Offcanvas } from "react-bootstrap"
import { getOrientation } from "./components/getOrientation"
import {
  DEFAULT_FE_MODEL,
  DEFAULT_PD_MODEL,
  FE_MODELS,
  PD_MODELS,
} from "./components/const"

function App() {
  const initialPdModel = useMemo(() => {
    const pdModel = reIdLocalStorage.getPDModel()
    if (pdModel && Object.keys(PD_MODELS).includes(pdModel)) {
      return pdModel
    } else {
      return DEFAULT_PD_MODEL
    }
  }, [])
  const initialFeModel = useMemo(() => {
    const feModel = reIdLocalStorage.getFEModel()
    if (feModel && Object.keys(FE_MODELS).includes(feModel)) {
      return feModel
    } else {
      return DEFAULT_FE_MODEL
    }
  }, [])

  const [pdModel, setPdModel_] = useState<string>(initialPdModel)
  const [feModel, setFeModel_] = useState<string>(initialFeModel)

  const setPdModel = (model: string) => {
    reIdLocalStorage.setPDModel(model)
    setPdModel_(model)
  }

  const setFeModel = (model: string) => {
    reIdLocalStorage.setFEModel(model)
    setFeModel_(model)
  }

  const windowRef = useRef<HTMLDivElement>(null)
  const { camRef } = Cam.useCamData()

  const resizeThrottleOccupied = useRef(false)

  useEffect(() => {
    if (!windowRef.current) return
    if (!camRef.current) return

    const window = windowRef.current
    const cam = camRef.current

    const setOrientation = () => {
      if (resizeThrottleOccupied.current) return

      resizeThrottleOccupied.current = true
      setTimeout(() => {
        const pixelsInRem = parseFloat(
          getComputedStyle(document.documentElement).fontSize,
        )

        const orientation = getOrientation(
          pixelsInRem,
          window.clientWidth,
          window.clientHeight,
          cam.videoWidth,
          cam.videoHeight,
        )

        window.setAttribute("data-orientation", orientation)
        resizeThrottleOccupied.current = false
      }, 100)
    }

    const resizeObserver = new ResizeObserver(setOrientation)
    resizeObserver.observe(window)
    resizeObserver.observe(cam)
  }, [camRef])

  return (
    <div className="window" ref={windowRef} data-orientation="portrait">
      <Navbar expand={false} className="nav-container">
        <Navbar.Toggle />
        <Navbar.Offcanvas>
          <Offcanvas.Header closeButton>
            <Offcanvas.Title>Setting</Offcanvas.Title>
          </Offcanvas.Header>
          <Offcanvas.Body>
            <ModelSelect
              pdModel={pdModel}
              feModel={feModel}
              setPdModel={setPdModel}
              setFeModel={setFeModel}
            />
            <Cam.CamSettings />
          </Offcanvas.Body>
        </Navbar.Offcanvas>
      </Navbar>
      <ReIdentification
        pdModel={PD_MODELS[pdModel]}
        feModel={FE_MODELS[feModel]}
      />
    </div>
  )
}

export default App
