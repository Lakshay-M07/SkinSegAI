import React from 'react'
import {
  Activity,
  AlertCircle,
  Eye,
  Flame,
  ImageIcon,
  Layers,
  Loader2,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from 'lucide-react'
import { BentoCard, SkeletonCard } from './BentoCard'

const THRESHOLD_PRESETS = [
  { label: 'Sensitive 0.35', value: 0.35 },
  { label: 'Balanced 0.50', value: 0.5 },
  { label: 'Strict 0.65', value: 0.65 },
]

const OVERLAY_STYLES = [
  { id: 'overlay', label: 'Translucent Fill' },
  { id: 'contour', label: 'Contour Outline' },
  { id: 'mask', label: 'Binary Mask' },
]

function ImageFrame({ src, alt, emptyIcon: EmptyIcon, emptyText, revealKey }) {
  return (
    <div className="h-[280px] rounded-lg bg-[#0A1426] border border-[#475569]/60 flex items-center justify-center overflow-hidden p-2">
      {src ? (
        <img
          key={revealKey}
          src={src}
          alt={alt}
          className="max-h-[260px] w-full object-contain rounded img-reveal"
        />
      ) : (
        <div className="text-center space-y-2 px-4">
          <EmptyIcon className="h-7 w-7 text-[#94A3B8] mx-auto" />
          <p className="text-xs text-[#94A3B8]">{emptyText}</p>
        </div>
      )}
    </div>
  )
}

export function WorkstationView({
  samples,
  samplesLoading,
  selectedSample,
  onSelectSample,
  previewUrl,
  gtPreviewUrl,
  sourceLabel,
  onUploadImage,
  onUploadGroundTruth,
  healthLoading,
  threshold,
  onThresholdChange,
  overlayStyle,
  onOverlayStyleChange,
  opacity,
  onOpacityChange,
  includeUncertainty,
  onIncludeUncertaintyChange,
  segmentResult,
  segmentLoading,
  segmentError,
  resultRevision,
}) {
  const unetSrc = segmentResult?.unet?.overlay_base64
    ? `data:image/png;base64,${segmentResult.unet.overlay_base64}`
    : ''
  const grabcutSrc = segmentResult?.grabcut?.overlay_base64
    ? `data:image/png;base64,${segmentResult.grabcut.overlay_base64}`
    : ''
  const origSrc = segmentResult?.original_base64
    ? `data:image/png;base64,${segmentResult.original_base64}`
    : previewUrl

  const unetLatency = segmentResult?.unet?.inference_ms
  const grabcutLatency = segmentResult?.grabcut?.inference_ms
  const borderUncertainty = segmentResult?.unet?.border_uncertainty
  const unetMetrics = segmentResult?.unet?.metrics

  return (
    <div className="space-y-4">
      {/* Top Row: Image Source + Live Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Image Source Card */}
        <BentoCard
          title="Image Source"
          subtitle="Choose a curated ISIC 2018 sample or upload a dermoscopic image"
          badge={sourceLabel || 'No image'}
          badgeTone={origSrc ? 'aqua' : 'neutral'}
          className="lg:col-span-5"
        >
          <div className="space-y-3">
            <div className="space-y-2">
              <label
                htmlFor="sample-select"
                className="block text-xs font-medium text-[#94A3B8]"
              >
                Curated Validation Samples
              </label>
              {samplesLoading ? (
                <div className="h-9 w-full rounded-lg skeleton-shimmer border border-[#475569]/60" />
              ) : (
                <>
                  <select
                    id="sample-select"
                    value={selectedSample}
                    onChange={(e) => onSelectSample(e.target.value)}
                    className="w-full h-9 px-3 rounded-lg bg-[#0A1426] border border-[#475569]/70 text-xs text-[#CFD8E3] focus:outline-none focus:border-[#22D3EE] transition-colors duration-200"
                  >
                    <option value="">
                      {samples.length > 0
                        ? 'Choose a sample image…'
                        : 'Backend offline or no samples found'}
                    </option>
                    {samples.map((item) => (
                      <option key={item.name} value={item.name}>
                        {item.name} {item.has_ground_truth ? '(+ GT mask)' : ''}
                      </option>
                    ))}
                  </select>

                  {samples.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-0.5">
                      {samples.map((item) => {
                        const active = selectedSample === item.name
                        return (
                          <button
                            key={item.name}
                            type="button"
                            onClick={() => onSelectSample(item.name)}
                            className={`px-2 py-1 rounded-md text-[11px] font-mono-num border transition-all duration-150 cursor-pointer ${
                              active
                                ? 'bg-[#22D3EE]/15 text-[#22D3EE] border-[#22D3EE]/50'
                                : 'bg-[#0A1426]/80 text-[#94A3B8] border-[#475569]/60 hover:border-[#22D3EE]/50 hover:text-[#CFD8E3]'
                            }`}
                          >
                            {item.name.replace('ISIC_', '#')}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2.5 pt-1">
              <label className="flex flex-col items-center justify-center gap-1 px-2.5 py-2.5 rounded-lg border border-dashed border-[#475569] bg-[#0A1426]/65 hover:border-[#22D3EE]/60 hover:bg-[#22D3EE]/[0.05] transition-all duration-150 cursor-pointer text-center">
                <Upload className="h-3.5 w-3.5 text-[#22D3EE]" />
                <span className="text-xs font-medium text-[#CFD8E3]">Upload Lesion</span>
                <span className="text-[10px] text-[#94A3B8]">PNG / JPEG</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) onUploadImage(file)
                  }}
                />
              </label>

              <label className="flex flex-col items-center justify-center gap-1 px-2.5 py-2.5 rounded-lg border border-dashed border-[#475569] bg-[#0A1426]/65 hover:border-[#FB923C]/60 hover:bg-[#FB923C]/[0.05] transition-all duration-150 cursor-pointer text-center">
                <Layers className="h-3.5 w-3.5 text-[#FB923C]" />
                <span className="text-xs font-medium text-[#CFD8E3]">Ground Truth</span>
                <span className="text-[10px] text-[#94A3B8]">Optional PNG</span>
                <input
                  type="file"
                  accept="image/png"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) onUploadGroundTruth(file)
                  }}
                />
              </label>
            </div>
          </div>
        </BentoCard>

        {/* Live Controls Bar */}
        <BentoCard
          title="Live Inference Controls"
          subtitle="Debounced real-time /api/segment parameters"
          badge={segmentLoading ? 'Running inference…' : 'Ready'}
          badgeTone={segmentLoading ? 'coral' : 'aqua'}
          className="lg:col-span-7"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Threshold + Presets */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#94A3B8] flex items-center gap-1.5 font-medium">
                  <SlidersHorizontal className="h-3.5 w-3.5 text-[#22D3EE]" />
                  Probability Threshold
                </span>
                <span className="font-mono-num text-[#22D3EE] font-semibold">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.01"
                value={threshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                className="w-full accent-[#22D3EE] cursor-pointer"
              />
              <div className="flex flex-wrap gap-1.5">
                {THRESHOLD_PRESETS.map((preset) => {
                  const active = Math.abs(threshold - preset.value) < 0.005
                  return (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => onThresholdChange(preset.value)}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium font-mono-num border transition-all duration-150 cursor-pointer ${
                        active
                          ? 'bg-[#22D3EE]/15 text-[#22D3EE] border-[#22D3EE]/50'
                          : 'bg-[#0A1426]/80 text-[#94A3B8] border-[#475569]/60 hover:border-[#475569] hover:text-[#CFD8E3]'
                      }`}
                    >
                      {preset.label}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Overlay Style + Opacity + Uncertainty Toggle */}
            <div className="space-y-3">
              <div className="space-y-1.5">
                <span className="block text-xs font-medium text-[#94A3B8]">
                  Overlay Style
                </span>
                <div className="grid grid-cols-3 gap-1 p-1 rounded-lg bg-[#0A1426] border border-[#475569]/60">
                  {OVERLAY_STYLES.map((style) => {
                    const active = overlayStyle === style.id
                    return (
                      <button
                        key={style.id}
                        type="button"
                        onClick={() => onOverlayStyleChange(style.id)}
                        className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all duration-150 cursor-pointer truncate ${
                          active
                            ? 'bg-[#22D3EE]/15 text-[#22D3EE] border border-[#22D3EE]/40'
                            : 'text-[#94A3B8] hover:text-[#CFD8E3] border border-transparent'
                        }`}
                      >
                        {style.label}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 items-center">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#94A3B8]">Opacity</span>
                    <span className="font-mono-num text-[#CFD8E3]">
                      {opacity.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="1.00"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
                    className="w-full accent-[#22D3EE] cursor-pointer"
                  />
                </div>

                <button
                  type="button"
                  onClick={() => onIncludeUncertaintyChange(!includeUncertainty)}
                  className={`h-9 px-3 rounded-lg border text-xs font-medium flex items-center justify-center gap-1.5 transition-all duration-150 cursor-pointer ${
                    includeUncertainty
                      ? 'bg-[#22D3EE]/15 text-[#22D3EE] border-[#22D3EE]/50'
                      : 'bg-[#0A1426]/80 text-[#94A3B8] border-[#475569]/60 hover:text-[#CFD8E3]'
                  }`}
                >
                  <Flame className="h-3.5 w-3.5" />
                  Uncertainty {includeUncertainty ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>
          </div>

          {/* Status strip */}
          <div className="mt-3 pt-2.5 border-t border-[#475569]/50 flex items-center justify-between text-[11px]">
            {segmentError ? (
              <span className="text-[#FB923C] flex items-center gap-1.5">
                <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                {segmentError}
              </span>
            ) : segmentLoading ? (
              <span className="text-[#22D3EE] flex items-center gap-1.5 font-mono-num">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-[#22D3EE]" />
                Running U-Net &amp; GrabCut inference…
              </span>
            ) : (
              <span className="text-[#94A3B8]">
                Adjusting sliders automatically updates overlays via debounced{' '}
                <code className="font-mono-num text-[#CFD8E3]">POST /api/segment</code>.
              </span>
            )}

            {segmentResult && (
              <span className="font-mono-num text-[#94A3B8]">
                {segmentResult.width}×{segmentResult.height} px
              </span>
            )}
          </div>
        </BentoCard>
      </div>

      {/* Multi-Panel Visual Inspection */}
      {healthLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <SkeletonCard heightClass="h-[280px]" />
          <SkeletonCard heightClass="h-[280px]" />
          <SkeletonCard heightClass="h-[280px]" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <BentoCard title="ORIGINAL INPUT" badge={sourceLabel || 'Empty'}>
            <ImageFrame
              src={origSrc}
              alt="Original dermoscopy"
              emptyIcon={ImageIcon}
              emptyText="Select a sample or upload a lesion image to begin."
              revealKey={`orig-${sourceLabel}`}
            />
            {gtPreviewUrl && (
              <div className="mt-3 pt-2.5 border-t border-[#475569]/50 flex items-center justify-between text-xs text-[#94A3B8]">
                <span>Ground-truth mask attached</span>
                <img
                  src={gtPreviewUrl}
                  alt="Ground-truth mask preview"
                  className="h-7 w-7 rounded border border-[#475569] object-cover"
                />
              </div>
            )}
          </BentoCard>

          <BentoCard
            title="U-NET SEGMENTATION"
            badge={unetLatency != null ? `${unetLatency.toFixed(1)} ms` : 'Pending'}
            badgeTone="aqua"
          >
            <ImageFrame
              src={unetSrc}
              alt="U-Net segmentation"
              emptyIcon={Sparkles}
              emptyText="U-Net prediction will appear once an image is selected."
              revealKey={`unet-${resultRevision}`}
            />
            {segmentResult?.unet?.measurements && (
              <div className="mt-3 pt-2.5 border-t border-[#475569]/50 flex items-center justify-between text-xs">
                <span className="text-[#94A3B8]">Lesion coverage</span>
                <span className="font-mono-num text-[#22D3EE] font-medium">
                  {segmentResult.unet.measurements.area_percent.toFixed(2)}%
                </span>
              </div>
            )}
          </BentoCard>

          <BentoCard
            title="GRABCUT BASELINE"
            badge={grabcutLatency != null ? `${grabcutLatency.toFixed(1)} ms` : 'Pending'}
            badgeTone="coral"
          >
            <ImageFrame
              src={grabcutSrc}
              alt="GrabCut baseline"
              emptyIcon={Layers}
              emptyText="GrabCut comparison will appear once an image is selected."
              revealKey={`gc-${resultRevision}`}
            />
            {segmentResult?.grabcut?.measurements && (
              <div className="mt-3 pt-2.5 border-t border-[#475569]/50 flex items-center justify-between text-xs">
                <span className="text-[#94A3B8]">Lesion coverage</span>
                <span className="font-mono-num text-[#FB923C] font-medium">
                  {segmentResult.grabcut.measurements.area_percent.toFixed(2)}%
                </span>
              </div>
            )}
          </BentoCard>
        </div>
      )}

      {/* Uncertainty & Ground-Truth Error Inspection Row */}
      {segmentResult && (includeUncertainty || segmentResult.error_map_base64) && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {includeUncertainty && segmentResult.uncertainty_heatmap_base64 && (
            <BentoCard
              title="SINGLE-PASS UNCERTAINTY HEATMAP"
              subtitle="Binary predictive entropy H(p) — peaks at ambiguous borders (p ≈ 0.5)"
              badge={
                borderUncertainty != null
                  ? `Border H: ${borderUncertainty.toFixed(4)}`
                  : 'Inferno'
              }
              badgeTone="aqua"
              className={
                segmentResult.error_map_base64 ? 'lg:col-span-7' : 'lg:col-span-12'
              }
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <span className="text-[11px] font-medium text-[#94A3B8] uppercase tracking-wider">
                    Entropy Overlay
                  </span>
                  <ImageFrame
                    src={`data:image/png;base64,${segmentResult.uncertainty_overlay_base64}`}
                    alt="Uncertainty overlay"
                    emptyIcon={Flame}
                    emptyText="No overlay"
                    revealKey={`uov-${resultRevision}`}
                  />
                </div>
                <div className="space-y-1.5">
                  <span className="text-[11px] font-medium text-[#94A3B8] uppercase tracking-wider">
                    Entropy Map [0 → 1]
                  </span>
                  <ImageFrame
                    src={`data:image/png;base64,${segmentResult.uncertainty_heatmap_base64}`}
                    alt="Uncertainty heatmap"
                    emptyIcon={Flame}
                    emptyText="No heatmap"
                    revealKey={`uhm-${resultRevision}`}
                  />
                </div>
              </div>

              {borderUncertainty != null && (
                <div className="mt-3 pt-3 border-t border-[#475569]/50 flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="text-[#94A3B8] flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5 text-[#22D3EE]" />
                    Mean Border Uncertainty (dilated boundary ring)
                  </span>
                  <span className="font-mono-num text-sm font-semibold text-[#22D3EE]">
                    {borderUncertainty.toFixed(4)}
                  </span>
                </div>
              )}
            </BentoCard>
          )}

          {segmentResult.error_map_base64 && (
            <BentoCard
              title="TP / FP / FN COMPARISON"
              subtitle="U-Net prediction vs ground-truth mask"
              badge={
                unetMetrics ? `Dice ${unetMetrics.dice.toFixed(4)}` : 'Ground Truth'
              }
              badgeTone="coral"
              className={
                includeUncertainty && segmentResult.uncertainty_heatmap_base64
                  ? 'lg:col-span-5'
                  : 'lg:col-span-12'
              }
            >
              <ImageFrame
                src={`data:image/png;base64,${segmentResult.error_map_base64}`}
                alt="TP FP FN error map"
                emptyIcon={Eye}
                emptyText="Error map"
                revealKey={`err-${resultRevision}`}
              />

              <div className="mt-3 pt-2.5 border-t border-[#475569]/50 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                <div className="flex items-center gap-3 text-[#94A3B8]">
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-emerald-500" /> TP
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-red-500" /> FP
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-blue-500" /> FN
                  </span>
                </div>

                {unetMetrics && (
                  <div className="flex items-center gap-3 font-mono-num text-[#CFD8E3]">
                    <span>IoU {unetMetrics.iou.toFixed(3)}</span>
                    <span>P {unetMetrics.precision.toFixed(3)}</span>
                    <span>R {unetMetrics.recall.toFixed(3)}</span>
                  </div>
                )}
              </div>
            </BentoCard>
          )}
        </div>
      )}
    </div>
  )
}
