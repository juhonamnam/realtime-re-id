import { useEffect, useRef } from "react"
import Cam from "./components/cam"
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap-icons/font/bootstrap-icons.css"
import { ReIdentification } from "./components/reIdentification"
import "./app.scss"
import { ReIDConfig } from "./components/reidConfig"
import { Navbar, Offcanvas } from "react-bootstrap"
import { getOrientation } from "./components/getOrientation"

function App() {
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
            <ReIDConfig />
            <Cam.CamSettings />
          </Offcanvas.Body>
        </Navbar.Offcanvas>
      </Navbar>
      <ReIdentification />
    </div>
  )
}

export default App
