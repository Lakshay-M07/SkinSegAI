import React from 'react'
import { Download, FileJson, FileSpreadsheet, Image as ImageIcon } from 'lucide-react'
import { BentoCard, SkeletonCard } from './BentoCard'

const METRIC_ROWS = [
  { key: 'dice', label: 'Dice Coefficient', format: (v) => v.toFixed(4) },
  { key: 'iou', label: 'IoU (Jaccard)', format: (v) => v.toFixed(4) },
  { key: 'precision', label: 'Precision', format: (v) => v.toFixed(4) },
  { key: 'recall', label: 'Recall', format: (v) => v.toFixed(4) },
  {
    key: 'inference_ms',
    label: 'Inference Latency',
    format: (v) => `${v.toFixed(1)} ms`,
    lowerIsBetter: true,
  },
]

function triggerBlobDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function GroupedOverlapChart({ unet, grabcut }) {
  const items = [
    { label: 'Dice', u: unet?.dice ?? 0, g: grabcut?.dice ?? 0 },
    { label: 'IoU', u: unet?.iou ?? 0, g: grabcut?.iou ?? 0 },
    { label: 'Precision', u: unet?.precision ?? 0, g: grabcut?.precision ?? 0 },
    { label: 'Recall', u: unet?.recall ?? 0, g: grabcut?.recall ?? 0 },
  ]

  const width = 520
  const height = 220
  const padL = 40
  const padR = 16
  const padT = 18
  const padB = 34
  const plotW = width - padL - padR
  const plotH = height - padT - padB
  const groupW = plotW / items.length
  const barW = Math.min(28, groupW * 0.28)

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto overflow-visible"
      role="img"
      aria-label="Overlap metrics comparison bar chart"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
        const y = padT + plotH * (1 - tick)
        return (
          <g key={tick}>
            <line
              x1={padL}
              x2={width - padR}
              y1={y}
              y2={y}
              stroke="rgba(71, 85, 105, 0.45)"
              strokeDasharray="3 3"
            />
            <text
              x={padL - 8}
              y={y + 3}
              textAnchor="end"
              fill="#94A3B8"
              fontSize="10"
              className="font-mono-num"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        )
      })}

      {items.map((item, idx) => {
        const cx = padL + groupW * idx + groupW / 2
        const uH = Math.max(0, Math.min(1, item.u)) * plotH
        const gH = Math.max(0, Math.min(1, item.g)) * plotH
        return (
          <g key={item.label}>
            <rect
              x={cx - barW - 3}
              y={padT + plotH - uH}
              width={barW}
              height={uH}
              rx="4"
              fill="#22D3EE"
            />
            <text
              x={cx - barW / 2 - 3}
              y={padT + plotH - uH - 5}
              textAnchor="middle"
              fill="#22D3EE"
              fontSize="9.5"
              className="font-mono-num"
            >
              {item.u.toFixed(2)}
            </text>

            <rect
              x={cx + 3}
              y={padT + plotH - gH}
              width={barW}
              height={gH}
              rx="4"
              fill="#FB923C"
            />
            <text
              x={cx + barW / 2 + 3}
              y={padT + plotH - gH - 5}
              textAnchor="middle"
              fill="#FB923C"
              fontSize="9.5"
              className="font-mono-num"
            >
              {item.g.toFixed(2)}
            </text>

            <text
              x={cx}
              y={height - 10}
              textAnchor="middle"
              fill="#94A3B8"
              fontSize="11"
            >
              {item.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function HorizontalRatioBars({ rows }) {
  return (
    <div className="space-y-3.5">
      {rows.map((row) => {
        const maxVal = Math.max(row.unet || 0, row.grabcut || 0, 1e-6)
        const uPct = ((row.unet || 0) / maxVal) * 100
        const gPct = ((row.grabcut || 0) / maxVal) * 100
        return (
          <div key={row.label} className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#94A3B8]">{row.label}</span>
              <span className="font-mono-num text-[11px] text-[#CFD8E3]">
                <span className="text-[#22D3EE]">{row.format(row.unet)}</span>
                {' · '}
                <span className="text-[#FB923C]">{row.format(row.grabcut)}</span>
              </span>
            </div>
            <div className="space-y-1">
              <div className="h-2 rounded-full bg-[#0A1426] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#22D3EE] transition-all duration-300"
                  style={{ width: `${Math.max(2, uPct)}%` }}
                />
              </div>
              <div className="h-2 rounded-full bg-[#0A1426] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#FB923C] transition-all duration-300"
                  style={{ width: `${Math.max(2, gPct)}%` }}
                />
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function ComparisonView({
  benchmark,
  benchmarkLoading,
  segmentResult,
  sourceLabel,
}) {
  if (benchmarkLoading) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <SkeletonCard key={i} lines={1} heightClass="h-14" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <SkeletonCard heightClass="h-56" />
          <SkeletonCard heightClass="h-56" />
        </div>
      </div>
    )
  }

  const summary = benchmark?.summary || []
  const unetBench = summary.find((r) => r.algorithm === 'U-Net') || null
  const grabcutBench = summary.find((r) => r.algorithm === 'GrabCut') || null

  const unetMorph = segmentResult?.unet?.measurements
  const grabcutMorph = segmentResult?.grabcut?.measurements

  function handleExportJson() {
    const payload = {
      exported_at: new Date().toISOString(),
      source: sourceLabel || null,
      benchmark: benchmark?.summary || [],
      current_segmentation: segmentResult
        ? {
            width: segmentResult.width,
            height: segmentResult.height,
            threshold: segmentResult.threshold,
            overlay_style: segmentResult.overlay_style,
            opacity: segmentResult.opacity,
            unet: {
              inference_ms: segmentResult.unet.inference_ms,
              border_uncertainty: segmentResult.unet.border_uncertainty,
              measurements: segmentResult.unet.measurements,
              metrics: segmentResult.unet.metrics,
            },
            grabcut: {
              inference_ms: segmentResult.grabcut.inference_ms,
              measurements: segmentResult.grabcut.measurements,
              metrics: segmentResult.grabcut.metrics,
            },
          }
        : null,
    }
    triggerBlobDownload(
      JSON.stringify(payload, null, 2),
      'SkinSegAI_summary.json',
      'application/json'
    )
  }

  function handleExportCsv() {
    const headers = [
      'scope',
      'algorithm',
      'dice',
      'iou',
      'precision',
      'recall',
      'inference_ms',
      'area_pixels',
      'area_percent',
      'perimeter_pixels',
      'circularity',
      'asymmetry',
      'border_uncertainty',
    ]
    const rows = []

    for (const entry of summary) {
      rows.push([
        'benchmark_test_split',
        entry.algorithm,
        entry.dice,
        entry.iou,
        entry.precision,
        entry.recall,
        entry.inference_ms,
        '',
        '',
        '',
        '',
        '',
        '',
      ])
    }

    if (segmentResult) {
      for (const algoKey of ['unet', 'grabcut']) {
        const item = segmentResult[algoKey]
        const m = item.measurements || {}
        const met = item.metrics || {}
        rows.push([
          `image:${sourceLabel || 'uploaded'}`,
          item.algorithm,
          met.dice ?? '',
          met.iou ?? '',
          met.precision ?? '',
          met.recall ?? '',
          item.inference_ms,
          m.area_pixels ?? '',
          m.area_percent ?? '',
          m.perimeter_pixels ?? '',
          m.circularity ?? '',
          m.asymmetry ?? '',
          item.border_uncertainty ?? '',
        ])
      }
    }

    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
    triggerBlobDownload(csv, 'SkinSegAI_metrics.csv', 'text/csv;charset=utf-8;')
  }

  const imageDownloads = segmentResult
    ? [
        {
          label: 'U-Net Overlay PNG',
          b64: segmentResult.unet.overlay_base64,
          filename: 'unet_overlay.png',
        },
        {
          label: 'U-Net Mask PNG',
          b64: segmentResult.unet.mask_base64,
          filename: 'unet_mask.png',
        },
        {
          label: 'GrabCut Overlay PNG',
          b64: segmentResult.grabcut.overlay_base64,
          filename: 'grabcut_overlay.png',
        },
        ...(segmentResult.uncertainty_heatmap_base64
          ? [
              {
                label: 'Uncertainty Heatmap PNG',
                b64: segmentResult.uncertainty_heatmap_base64,
                filename: 'uncertainty_heatmap.png',
              },
            ]
          : []),
      ]
    : []

  return (
    <div className="space-y-4">
      {/* Legend + Export Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-[#1E293B] border border-[#475569]/70 px-4 py-3">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <span className="font-semibold text-[#CFD8E3]">
            Held-Out Test Benchmark
            {unetBench?.samples ? (
              <span className="ml-2 font-mono-num text-[#94A3B8] font-normal">
                (n = {unetBench.samples})
              </span>
            ) : null}
          </span>
          <span className="inline-flex items-center gap-1.5 text-[#22D3EE]">
            <span className="h-2.5 w-2.5 rounded-sm bg-[#22D3EE]" /> U-Net
          </span>
          <span className="inline-flex items-center gap-1.5 text-[#FB923C]">
            <span className="h-2.5 w-2.5 rounded-sm bg-[#FB923C]" /> GrabCut
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleExportJson}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#22D3EE]/15 hover:bg-[#22D3EE]/25 text-[#22D3EE] border border-[#22D3EE]/40 text-xs font-medium transition-colors duration-150 cursor-pointer"
          >
            <FileJson className="h-3.5 w-3.5 text-[#22D3EE]" />
            Export JSON
          </button>
          <button
            type="button"
            onClick={handleExportCsv}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0A1426] hover:border-[#475569] text-[#CFD8E3] border border-[#475569]/70 text-xs font-medium transition-colors duration-150 cursor-pointer"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-[#FB923C]" />
            Export CSV
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {METRIC_ROWS.map((metric) => {
          const uVal = unetBench?.[metric.key] ?? 0
          const gVal = grabcutBench?.[metric.key] ?? 0
          const delta = uVal - gVal
          const deltaText = metric.lowerIsBetter
            ? gVal > 0 && uVal > 0
              ? `${(gVal / uVal).toFixed(1)}× faster`
              : '—'
            : `${delta >= 0 ? '+' : ''}${delta.toFixed(4)} vs GrabCut`

          return (
            <BentoCard
              key={metric.key}
              title={metric.label}
              badge={deltaText}
              badgeTone="aqua"
            >
              <div className="space-y-2 pt-1">
                <div className="flex items-baseline justify-between">
                  <span className="text-[11px] text-[#94A3B8]">U-Net</span>
                  <span className="text-xl font-semibold font-mono-num text-[#22D3EE]">
                    {metric.format(uVal)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between border-t border-[#475569]/50 pt-2">
                  <span className="text-[11px] text-[#94A3B8]">GrabCut</span>
                  <span className="text-sm font-medium font-mono-num text-[#FB923C]">
                    {metric.format(gVal)}
                  </span>
                </div>
              </div>
            </BentoCard>
          )
        })}
      </div>

      {/* Charts Row: Overlap Metrics + Morphology Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <BentoCard
          title="Overlap & Latency Comparison"
          subtitle="Benchmark summary from artifacts/metrics/comparison_summary.csv"
          badge={benchmark?.available ? 'Live CSV' : 'Fallback'}
          badgeTone={benchmark?.available ? 'aqua' : 'neutral'}
          className="lg:col-span-7"
        >
          <GroupedOverlapChart unet={unetBench} grabcut={grabcutBench} />

          {unetBench && grabcutBench && (
            <div className="mt-4 pt-3 border-t border-[#475569]/50">
              <HorizontalRatioBars
                rows={[
                  {
                    label: 'Mean Inference Latency (ms)',
                    unet: unetBench.inference_ms,
                    grabcut: grabcutBench.inference_ms,
                    format: (v) => `${v.toFixed(1)} ms`,
                  },
                ]}
              />
            </div>
          )}
        </BentoCard>

        <BentoCard
          title="Current Lesion Morphology"
          subtitle={
            sourceLabel
              ? `Shape metrics for ${sourceLabel}`
              : 'Run segmentation in Workstation to compare morphology'
          }
          badge={segmentResult ? 'Active Image' : 'No Image'}
          badgeTone={segmentResult ? 'coral' : 'neutral'}
          className="lg:col-span-5"
        >
          {unetMorph && grabcutMorph ? (
            <div className="space-y-4">
              <HorizontalRatioBars
                rows={[
                  {
                    label: 'Area (px²)',
                    unet: unetMorph.area_pixels,
                    grabcut: grabcutMorph.area_pixels,
                    format: (v) => `${Math.round(v).toLocaleString()} px²`,
                  },
                  {
                    label: 'Coverage (%)',
                    unet: unetMorph.area_percent,
                    grabcut: grabcutMorph.area_percent,
                    format: (v) => `${v.toFixed(2)}%`,
                  },
                  {
                    label: 'Perimeter (px)',
                    unet: unetMorph.perimeter_pixels,
                    grabcut: grabcutMorph.perimeter_pixels,
                    format: (v) => `${v.toFixed(1)} px`,
                  },
                  {
                    label: 'Circularity / Compactness',
                    unet: unetMorph.circularity ?? 0,
                    grabcut: grabcutMorph.circularity ?? 0,
                    format: (v) => v.toFixed(4),
                  },
                  {
                    label: 'Asymmetry Index',
                    unet: unetMorph.asymmetry ?? 0,
                    grabcut: grabcutMorph.asymmetry ?? 0,
                    format: (v) => v.toFixed(4),
                  },
                ]}
              />
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-10 text-center space-y-2">
              <ImageIcon className="h-7 w-7 text-[#94A3B8]" />
              <p className="text-xs text-[#94A3B8]">
                Select or upload an image in the Workstation tab to inspect live morphology
                metrics.
              </p>
            </div>
          )}
        </BentoCard>
      </div>

      {/* Downloadable Mask & Overlay Image Links */}
      {imageDownloads.length > 0 && (
        <BentoCard
          title="Downloadable Visual Artifacts"
          subtitle="Export rendered PNG overlays, binary masks, and entropy maps"
          badge={`${imageDownloads.length} PNGs`}
        >
          <div className="flex flex-wrap gap-2.5">
            {imageDownloads.map((item) => (
              <a
                key={item.filename}
                href={`data:image/png;base64,${item.b64}`}
                download={item.filename}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#0A1426] border border-[#475569]/70 hover:border-[#22D3EE]/60 text-xs text-[#CFD8E3] transition-colors duration-150"
              >
                <Download className="h-3.5 w-3.5 text-[#22D3EE]" />
                {item.label}
              </a>
            ))}
          </div>
        </BentoCard>
      )}
    </div>
  )
}
