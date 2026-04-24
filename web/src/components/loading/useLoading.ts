import { useContext } from "react"
import { LoadingContext } from "./context"

export const useLoading = () => {
  const { setLoading } = useContext(LoadingContext)
  return { setLoading }
}
