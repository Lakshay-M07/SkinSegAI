const BASE = (import.meta.env.VITE_API_BASE_URL || '') + '/api/v2'
const KEY = 'skinseg.v2.session'

export class APIError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail))
    this.status = status
  }
}
export async function api(path, {session, method = 'GET', json, form, binary = false} = {}) {
  const headers = {}
  if (session) headers['X-Session-ID'] = session
  if (json !== undefined) headers['Content-Type'] = 'application/json'
  const response = await fetch(BASE + path, {
    method, headers, body: form ?? (json === undefined ? undefined : JSON.stringify(json)),
    cache: 'no-store',
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new APIError(response.status, body.detail || `HTTP ${response.status}`)
  }
  return binary ? response.blob() : response.json()
}
let opening = null
export function openSession() {
  if (opening) return opening
  opening = (async () => {
    const old = sessionStorage.getItem(KEY)
    if (old) {
      try { return {...await api('/context', {session: old}), session_id: old} }
      catch (err) { if (err.status !== 404) throw err; sessionStorage.removeItem(KEY) }
    }
    let created
    try { created = await api('/session?provider=ollama', {method: 'POST'}) }
    catch (err) {
      if (err.status !== 503) throw err
      created = await api('/session?provider=offline', {method: 'POST'})
    }
    sessionStorage.setItem(KEY, created.session_id)
    return created
  })().finally(() => { opening = null })
  return opening
}
export function forgetSession() { sessionStorage.removeItem(KEY) }
export function asBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error)
    reader.onload = () => resolve(String(reader.result).split(',')[1])
    reader.readAsDataURL(blob)
  })
}
export function download(blob, name) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url; link.download = name
  document.body.appendChild(link); link.click(); link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
