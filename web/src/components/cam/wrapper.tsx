import type { PropsWithChildren } from "react"

export const CamWrapper = ({ children }: PropsWithChildren) => {
  return <div className="position-relative">{children}</div>
}
