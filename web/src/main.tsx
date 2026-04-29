import { createRoot } from "react-dom/client"
import { CamProvider } from "./components/cam"
import { LoadingProvider } from "./components/loading"
import { ReIDConfigProvider } from "./components/reidConfig"
import App from "./App.tsx"

createRoot(document.getElementById("root")!).render(
  // <StrictMode>
  <CamProvider>
    <ReIDConfigProvider>
      <LoadingProvider>
        <App />
      </LoadingProvider>
    </ReIDConfigProvider>
  </CamProvider>,
  // </StrictMode>,
)
