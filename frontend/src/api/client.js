const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

async function requestJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const err = await res.json()
      if (err?.detail) detail = err.detail
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail)
  }
  return res.json()
}

export function fetchHealth() {
  return requestJson('/api/health')
}

export function fetchSamples() {
  return requestJson('/api/samples')
}

export function fetchSampleDetail(name) {
  return requestJson(`/api/samples/${encodeURIComponent(name)}`)
}

export function fetchBenchmark() {
  return requestJson('/api/benchmark')
}

export function segmentImage({
  file,
  sampleId,
  groundTruthFile,
  threshold = 0.5,
  overlayStyle = 'overlay',
  opacity = 0.35,
  includeUncertainty = true,
  signal,
}) {
  const form = new FormData()
  if (file) {
    form.append('image', file)
  } else if (sampleId) {
    form.append('sample_id', sampleId)
  }
  if (groundTruthFile) {
    form.append('ground_truth', groundTruthFile)
  }
  form.append('threshold', String(threshold))
  form.append('overlay_style', overlayStyle)
  form.append('opacity', String(opacity))
  form.append('include_uncertainty', String(includeUncertainty))

  return requestJson('/api/segment', {
    method: 'POST',
    body: form,
    signal,
  })
}
