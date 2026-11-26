import React, { useEffect, useRef, useState } from 'react'
import {
  fetchBenchmark,
  fetchHealth,
  fetchSampleDetail,
  fetchSamples,
  segmentImage,
} from './api/client'
import { TopHeader } from './components/TopHeader'
import { WorkstationView } from './components/WorkstationView'
import { ComparisonView } from './components/ComparisonView'
import { AboutProjectView } from './components/AboutProjectView'

export default function App() {
  const [activeTab, setActiveTab] = useState('workstation')
  const [health, setHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)

  const [samples, setSamples] = useState([])
  const [samplesLoading, setSamplesLoading] = useState(true)
  const [selectedSample, setSelectedSample] = useState('')
  const [benchmark, setBenchmark] = useState(null)
  const [benchmarkLoading, setBenchmarkLoading] = useState(true)

  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadedGtFile, setUploadedGtFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [gtPreviewUrl, setGtPreviewUrl] = useState('')
  const [sourceLabel, setSourceLabel] = useState('')

  // Live segmentation controls
  const [threshold, setThreshold] = useState(0.5)
  const [overlayStyle, setOverlayStyle] = useState('overlay')
  const [opacity, setOpacity] = useState(0.35)
  const [includeUncertainty, setIncludeUncertainty] = useState(true)

  // Inference state
  const [segmentResult, setSegmentResult] = useState(null)
  const [segmentLoading, setSegmentLoading] = useState(false)
  const [segmentError, setSegmentError] = useState('')
  const [resultRevision, setResultRevision] = useState(0)

  const abortRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      try {
        const [healthData, samplesData, benchmarkData] = await Promise.all([
          fetchHealth(),
          fetchSamples(),
          fetchBenchmark(),
        ])
        if (!cancelled) {
          setHealth(healthData)
          setBenchmark(benchmarkData)
          const list = samplesData.samples || []
          setSamples(list)
          // Auto-select the first curated sample for immediate inspection
          if (list.length > 0) {
            handleSelectSample(list[0].name)
          }
        }
      } catch {
        // Keep graceful offline state when FastAPI is not running
      } finally {
        if (!cancelled) {
          setHealthLoading(false)
          setSamplesLoading(false)
          setBenchmarkLoading(false)
        }
      }
    }

    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  // Debounced POST /api/segment whenever image source or control params change
  useEffect(() => {
    if (!uploadedFile && !selectedSample) {
      setSegmentResult(null)
      setSegmentError('')
      return
    }

    const timer = setTimeout(async () => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      setSegmentLoading(true)
      setSegmentError('')

      try {
        const data = await segmentImage({
          file: uploadedFile,
          sampleId: uploadedFile ? null : selectedSample,
          groundTruthFile: uploadedGtFile,
          threshold,
          overlayStyle,
          opacity,
          includeUncertainty,
          signal: controller.signal,
        })
        setSegmentResult(data)
        setResultRevision((prev) => prev + 1)
      } catch (err) {
        if (err.name !== 'AbortError') {
          setSegmentError(err.message || 'Segmentation request failed.')
        }
      } finally {
        if (abortRef.current === controller) {
          setSegmentLoading(false)
        }
      }
    }, 220)

    return () => {
      clearTimeout(timer)
    }
  }, [
    uploadedFile,
    selectedSample,
    uploadedGtFile,
    threshold,
    overlayStyle,
    opacity,
    includeUncertainty,
  ])

  async function handleSelectSample(name) {
    setSelectedSample(name)
    setUploadedFile(null)
    setUploadedGtFile(null)

    if (!name) {
      setPreviewUrl('')
      setGtPreviewUrl('')
      setSourceLabel('')
      return
    }
    try {
      const detail = await fetchSampleDetail(name)
      setPreviewUrl(`data:image/png;base64,${detail.image_base64}`)
      setGtPreviewUrl(
        detail.ground_truth_base64
          ? `data:image/png;base64,${detail.ground_truth_base64}`
          : ''
      )
      setSourceLabel(detail.name)
    } catch {
      // ignore preview error if backend unavailable
    }
  }

  function handleUploadImage(file) {
    setSelectedSample('')
    setUploadedFile(file)
    setSourceLabel(file.name)
    setPreviewUrl(URL.createObjectURL(file))
  }

  function handleUploadGroundTruth(file) {
    setUploadedGtFile(file)
    setGtPreviewUrl(URL.createObjectURL(file))
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#0A1426] text-[#CFD8E3]">
      <TopHeader
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        healthLoading={healthLoading}
      />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-5">
        <div key={activeTab} className="tab-panel-enter">
          {activeTab === 'workstation' && (
            <WorkstationView
              samples={samples}
              samplesLoading={samplesLoading}
              selectedSample={selectedSample}
              onSelectSample={handleSelectSample}
              previewUrl={previewUrl}
              gtPreviewUrl={gtPreviewUrl}
              sourceLabel={sourceLabel}
              onUploadImage={handleUploadImage}
              onUploadGroundTruth={handleUploadGroundTruth}
              healthLoading={healthLoading}
              threshold={threshold}
              onThresholdChange={setThreshold}
              overlayStyle={overlayStyle}
              onOverlayStyleChange={setOverlayStyle}
              opacity={opacity}
              onOpacityChange={setOpacity}
              includeUncertainty={includeUncertainty}
              onIncludeUncertaintyChange={setIncludeUncertainty}
              segmentResult={segmentResult}
              segmentLoading={segmentLoading}
              segmentError={segmentError}
              resultRevision={resultRevision}
            />
          )}
          {activeTab === 'comparison' && (
            <ComparisonView
              benchmark={benchmark}
              benchmarkLoading={benchmarkLoading}
              segmentResult={segmentResult}
              sourceLabel={sourceLabel}
            />
          )}
          {activeTab === 'about' && <AboutProjectView health={health} />}
        </div>
      </main>

      <footer className="border-t border-[#475569]/60 py-3 px-4 sm:px-6">
        <p className="max-w-[1440px] mx-auto text-center text-[11px] text-[#94A3B8]">
          Research use only — SkinSegAI is an academic segmentation workstation and is not a
          medical diagnostic tool.
        </p>
      </footer>
    </div>
  )
}
