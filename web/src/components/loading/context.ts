import { createContext } from "react"

export const LoadingContext = createContext<{
  setLoading: (value: boolean) => void
}>({ setLoading: () => {} })
