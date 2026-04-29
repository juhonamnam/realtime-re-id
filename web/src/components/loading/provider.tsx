import { type PropsWithChildren, useState } from "react"
import { Modal } from "react-bootstrap"
import { LoadingContext } from "./context"

export const LoadingProvider = ({ children }: PropsWithChildren) => {
  const [loading, setLoading] = useState(false)

  return (
    <LoadingContext.Provider value={{ setLoading }}>
      {children}
      <Modal
        className="d-flex justify-content-center align-items-center"
        show={loading}
        dialogAs={() => (
          <div
            className="spinner-border"
            style={{ width: "3rem", height: "3rem" }}
          >
            <span className="visually-hidden">Loading...</span>
          </div>
        )}
      />
    </LoadingContext.Provider>
  )
}
