import React from 'react'
import { Activity, Cpu, Layers } from 'lucide-react'

const TABS = [
  { id: 'workstation', label: 'Workstation' },
  { id: 'comparison', label: 'Performance Comparison' },
  { id: 'about', label: 'About Project' },
]

export function TopHeader({ activeTab, onTabChange, health, healthLoading }) {
  const isReady = health?.status === 'ok'
  const archLabel = health?.arch || 'Offline'
  const deviceLabel = health?.device ? health.device.toUpperCase() : '—'

  return (
    <header className="sticky top-0 z-30 border-b border-[#475569]/60 bg-[#0A1426]/92 backdrop-blur-md">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Brand mark */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-9 w-9 rounded-lg bg-[#22D3EE]/15 border border-[#22D3EE]/35 flex items-center justify-center shrink-0">
            <Activity className="h-4.5 w-4.5 text-[#22D3EE]" />
          </div>
          <div className="min-w-0">
            <span className="block text-base font-bold tracking-tight text-[#CFD8E3]">
              SkinSegAI
            </span>
            <p className="text-[11px] text-[#94A3B8] truncate hidden md:block">
              Dermoscopic Lesion Segmentation &amp; Comparative Analysis
            </p>
          </div>
        </div>

        {/* Top-level navigation tabs */}
        <nav
          aria-label="Primary navigation"
          className="flex items-center gap-1 p-1 rounded-xl bg-[#1E293B] border border-[#475569]/70"
        >
          {TABS.map((tab) => {
            const active = activeTab === tab.id
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onTabChange(tab.id)}
                className={`px-3 sm:px-4 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all duration-[180ms] cursor-pointer ${
                  active
                    ? 'bg-[#22D3EE]/15 text-[#22D3EE] border border-[#22D3EE]/40 shadow-sm'
                    : 'text-[#94A3B8] hover:text-[#CFD8E3] border border-transparent'
                }`}
              >
                {tab.label}
              </button>
            )
          })}
        </nav>

        {/* Active model / device status pill */}
        <div className="hidden lg:flex items-center gap-2">
          {healthLoading ? (
            <div className="h-8 w-52 rounded-full skeleton-shimmer border border-[#475569]/60" />
          ) : (
            <div className="inline-flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-[#1E293B] border border-[#475569]/70 text-xs">
              <span
                className={`h-2 w-2 rounded-full ${
                  isReady ? 'bg-[#22D3EE]' : 'bg-[#FB923C]'
                }`}
              />
              <span className="inline-flex items-center gap-1 text-[#22D3EE] font-mono-num">
                <Layers className="h-3.5 w-3.5 text-[#22D3EE]" />
                {archLabel}
              </span>
              <span className="text-[#475569]">|</span>
              <span className="inline-flex items-center gap-1 text-[#94A3B8] font-mono-num">
                <Cpu className="h-3.5 w-3.5 text-[#94A3B8]" />
                {deviceLabel}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
