export function getApiBase() {
  return process.env.NEXT_PUBLIC_AFRAME_API_BASE || "http://localhost:8000"
}
