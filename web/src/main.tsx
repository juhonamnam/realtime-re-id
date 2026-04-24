// import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { CamProvider } from "./components/cam"
import { LoadingWrapper } from "./components/loading"
import App from "./App.tsx"

createRoot(document.getElementById("root")!).render(
  // <StrictMode>
    <CamProvider>
      <LoadingWrapper>
        <App />
      </LoadingWrapper>
    </CamProvider>
  // </StrictMode>,
)
