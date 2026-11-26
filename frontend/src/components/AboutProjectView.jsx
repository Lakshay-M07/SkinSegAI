import React from 'react'
import { ArrowRight, ShieldAlert } from 'lucide-react'
import { BentoCard } from './BentoCard'

const SECTION_LINKS = [
  { id: 'sec-overview', label: '1. Overview' },
  { id: 'sec-features', label: '2. Key Features' },
  { id: 'sec-findings', label: '3. Key Findings' },
  { id: 'sec-workflow', label: '4. Application Workflow' },
  { id: 'sec-limitations', label: '5. Limitations' },
  { id: 'sec-acknowledgements', label: '6. Acknowledgements' },
  { id: 'sec-disclaimer', label: '7. Disclaimer' },
]

const KEY_FEATURES = [
  {
    lead: 'Multi-Backbone U-Net',
    desc: 'Supports resnet34_unet, efficientnet_b2_unet, and custom_unet with checkpoint auto-detection.',
  },
  {
    lead: 'Composite Edge Loss',
    desc: 'Combines Soft Dice, BCE with Logits, and morphological Boundary Loss for sharp lesion borders.',
  },
  {
    lead: 'Single-Pass Uncertainty',
    desc: 'Computes pixel-wise binary entropy H(p) and Border Uncertainty Score without TTA overhead.',
  },
  {
    lead: 'GrabCut Baseline',
    desc: 'Runs automated graph-cut segmentation side-by-side for quantitative benchmarking.',
  },
  {
    lead: 'Lesion Morphology',
    desc: 'Extracts area, coverage, perimeter, circularity, asymmetry, and TP/FP/FN overlap metrics.',
  },
  {
    lead: 'Interactive Export',
    desc: 'Provides live threshold/opacity inspection with one-click JSON, CSV, and PNG downloads.',
  },
]

const KEY_FINDINGS = [
  {
    stat: '87.79%',
    sub: '80.23% IoU',
    tone: 'aqua',
    text: 'Mean test Dice coefficient achieved by resnet34_unet across 390 held-out images.',
  },
  {
    stat: '+26.32 pts',
    sub: 'vs. GrabCut',
    tone: 'aqua',
    text: 'Dice improvement over the classical GrabCut baseline on the same test split.',
  },
  {
    stat: '15.3x',
    sub: '126.0 vs 1933.8 ms',
    tone: 'coral',
    text: 'Faster mean inference latency with GPU-accelerated U-Net versus CPU GrabCut.',
  },
  {
    stat: '0.1370 vs. 0.3082',
    sub: 'Dice Std Dev',
    tone: 'coral',
    text: 'Lower variance across test cases, indicating stronger segmentation consistency.',
  },
  {
    stat: '87.65%',
    sub: 'Best Val Dice',
    tone: 'aqua',
    text: 'Peak validation Dice recorded during training with early stopping and LR scheduling.',
  },
]

const WORKFLOW_STEPS = [
  { step: '01', title: 'Input Image', sub: 'Sample or upload' },
  { step: '02', title: 'Preprocessing', sub: '256×256 normalize' },
  { step: '03', title: 'U-Net & GrabCut Inference', sub: 'Parallel masks' },
  { step: '04', title: 'Uncertainty & Morphology', sub: 'H(p) + shape' },
  { step: '05', title: 'Interactive Report', sub: 'JSON / CSV / PNG' },
]

const LIMITATIONS = [
  'Trained on ISIC 2018 dermoscopic images; performance on smartphone photos is not established.',
  'Missing patient IDs in metadata prevent a guarantee of complete patient-level independence.',
  'Morphological values are pixel-based estimates; physical units require spatial calibration.',
  'Segmentation masks delineate lesion boundaries only and do not classify benign vs. malignant.',
]

const ACKNOWLEDGEMENTS = [
  'International Skin Imaging Collaboration (ISIC 2018) for the Task 1 dermoscopic dataset.',
  'PyTorch & segmentation-models-pytorch teams for deep learning and pretrained encoder tooling.',
  'OpenCV community for GrabCut graph-cut segmentation and contour morphology utilities.',
]

function scrollToSection(id) {
  const el = document.getElementById(id)
  if (!el) return
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' })
}

export function AboutProjectView({ health }) {
  return (
    <div className="space-y-4">
      {/* Compact Section Jump-Bar */}
      <nav
        aria-label="About Project sections"
        className="rounded-xl bg-[#1E293B] border border-[#475569]/70 px-3 py-2.5 flex flex-wrap gap-1.5"
      >
        {SECTION_LINKS.map((sec) => (
          <button
            key={sec.id}
            type="button"
            onClick={() => scrollToSection(sec.id)}
            className="px-2.5 py-1 rounded-md text-[11px] font-medium font-mono-num bg-[#0A1426] text-[#94A3B8] border border-[#475569]/60 hover:border-[#22D3EE]/60 hover:text-[#22D3EE] transition-all duration-200 cursor-pointer"
          >
            {sec.label}
          </button>
        ))}
      </nav>

      {/* 1. Overview */}
      <div id="sec-overview" className="scroll-mt-20">
        <BentoCard
          title="1. Overview"
          badge={health?.arch || 'resnet34_unet'}
          badgeTone="aqua"
          className="border-l-2 border-l-[#22D3EE]/80"
        >
          <p className="text-xs sm:text-sm text-[#94A3B8] leading-relaxed">
            <strong className="text-[#CFD8E3]">SkinSegAI</strong> is an object-oriented
            dermoscopic lesion segmentation and comparative analysis workstation. It pairs
            U-Net deep learning backbones with a classical{' '}
            <strong className="text-[#FB923C]">GrabCut</strong> baseline, delivering
            single-pass predictive uncertainty heatmaps, lesion morphology metrics, and
            interactive visual comparison.
          </p>
        </BentoCard>
      </div>

      {/* 2. Key Features */}
      <div id="sec-features" className="scroll-mt-20">
        <BentoCard
          title="2. Key Features"
          badge={`${KEY_FEATURES.length} highlights`}
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {KEY_FEATURES.map((feat) => (
              <div
                key={feat.lead}
                className="rounded-lg bg-[#0A1426] border border-[#475569]/60 hover:border-[#22D3EE]/60 transition-all duration-200 p-3"
              >
                <div className="text-xs font-semibold text-[#22D3EE]">{feat.lead}</div>
                <p className="text-[11px] text-[#94A3B8] mt-1 leading-relaxed">{feat.desc}</p>
              </div>
            ))}
          </div>
        </BentoCard>
      </div>

      {/* 3. Key Findings */}
      <div id="sec-findings" className="scroll-mt-20">
        <BentoCard
          title="3. Key Findings"
          subtitle="Held-out ISIC 2018 test benchmark (n = 390)"
          badge="resnet34_unet vs GrabCut"
          badgeTone="aqua"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {KEY_FINDINGS.map((item) => {
              const isAqua = item.tone === 'aqua'
              return (
                <div
                  key={item.stat}
                  className={`rounded-lg bg-[#0A1426] border p-3 transition-all duration-200 flex flex-col justify-between ${
                    isAqua
                      ? 'border-[#22D3EE]/35 hover:border-[#22D3EE]/70'
                      : 'border-[#FB923C]/35 hover:border-[#FB923C]/70'
                  }`}
                >
                  <div>
                    <div
                      className={`text-lg font-bold font-mono-num ${
                        isAqua ? 'text-[#22D3EE]' : 'text-[#FB923C]'
                      }`}
                    >
                      {item.stat}
                    </div>
                    <span className="inline-block mt-0.5 text-[10px] font-mono-num uppercase tracking-wider text-[#94A3B8]">
                      {item.sub}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#94A3B8] mt-2 leading-relaxed">{item.text}</p>
                </div>
              )
            })}
          </div>
        </BentoCard>
      </div>

      {/* 4. Application Workflow */}
      <div id="sec-workflow" className="scroll-mt-20">
        <BentoCard
          title="4. Application Workflow"
          subtitle="End-to-end segmentation and reporting pipeline"
          badge="5 Stages"
          badgeTone="aqua"
        >
          <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-2">
            {WORKFLOW_STEPS.map((item, idx) => (
              <React.Fragment key={item.step}>
                <div className="flex-1 rounded-lg bg-[#0A1426] border border-[#22D3EE]/35 hover:border-[#22D3EE]/75 transition-all duration-200 px-3 py-2.5">
                  <div className="font-mono-num text-[10px] font-semibold text-[#22D3EE]">
                    STEP {item.step}
                  </div>
                  <div className="text-xs font-semibold text-[#CFD8E3] mt-0.5">{item.title}</div>
                  <div className="text-[11px] text-[#94A3B8] mt-0.5">{item.sub}</div>
                </div>
                {idx < WORKFLOW_STEPS.length - 1 && (
                  <ArrowRight className="h-4 w-4 text-[#22D3EE] shrink-0 self-center rotate-90 md:rotate-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </BentoCard>
      </div>

      {/* 5. Limitations & 6. Acknowledgements */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div id="sec-limitations" className="lg:col-span-6 scroll-mt-20">
          <BentoCard
            title="5. Limitations"
            badge="Scope"
            badgeTone="coral"
            className="h-full"
          >
            <ul className="space-y-2 text-xs text-[#94A3B8]">
              {LIMITATIONS.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#FB923C] mt-1.5 shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </BentoCard>
        </div>

        <div id="sec-acknowledgements" className="lg:col-span-6 scroll-mt-20">
          <BentoCard title="6. Acknowledgements" className="h-full">
            <ul className="space-y-2 text-xs text-[#94A3B8]">
              {ACKNOWLEDGEMENTS.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#22D3EE] mt-1.5 shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </BentoCard>
        </div>
      </div>

      {/* 7. Disclaimer */}
      <div id="sec-disclaimer" className="scroll-mt-20">
        <BentoCard
          title="7. Disclaimer"
          badge="Non-Diagnostic"
          badgeTone="coral"
          className="border-l-2 border-l-[#FB923C]/70"
        >
          <div className="flex items-start gap-2.5 text-xs text-[#94A3B8] leading-relaxed">
            <ShieldAlert className="h-4 w-4 text-[#FB923C] shrink-0 mt-0.5" />
            <p>
              SkinSegAI is an academic and educational research project. It is not clinically
              validated and must not be used as a substitute for professional medical diagnosis,
              treatment, or dermatological examination.
            </p>
          </div>
        </BentoCard>
      </div>
    </div>
  )
}
