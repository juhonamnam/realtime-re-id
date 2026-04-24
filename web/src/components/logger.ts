export const logger = (...messages: unknown[]) => {
  if (import.meta.env.PROD) return
  console.log(...messages)
}
