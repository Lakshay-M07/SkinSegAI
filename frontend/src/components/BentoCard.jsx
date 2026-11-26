import React from 'react'

export function BentoCard({
  title,
  subtitle,
  badge,
  badgeTone = 'neutral',
  className = '',
  children,
}) {
  const badgeClasses = {
    neutral: 'bg-[#0A1426]/60 text-[#94A3B8] border-[#475569]/70',
    aqua: 'bg-[#22D3EE]/12 text-[#22D3EE] border-[#22D3EE]/40',
    teal: 'bg-[#22D3EE]/12 text-[#22D3EE] border-[#22D3EE]/40',
    coral: 'bg-[#FB923C]/12 text-[#FB923C] border-[#FB923C]/40',
  }[badgeTone] || 'bg-[#0A1426]/60 text-[#94A3B8] border-[#475569]/70'

  return (
    <section
      className={`rounded-xl bg-[#1E293B] border border-[#475569]/70 hover:border-[#475569] transition-colors duration-200 flex flex-col ${className}`}
    >
      {(title || badge) && (
        <header className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[#475569]/50">
          <div className="min-w-0">
            {title && (
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#94A3B8] truncate">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="text-xs text-[#94A3B8]/80 mt-0.5 truncate">{subtitle}</p>
            )}
          </div>
          {badge && (
            <span
              className={`shrink-0 inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium border font-mono-num ${badgeClasses}`}
            >
              {badge}
            </span>
          )}
        </header>
      )}
      <div className="p-4 flex-1 flex flex-col">{children}</div>
    </section>
  )
}

export function SkeletonCard({ lines = 3, heightClass = 'h-56', className = '' }) {
  return (
    <div
      className={`rounded-xl bg-[#1E293B] border border-[#475569]/70 p-4 flex flex-col gap-3 ${className}`}
    >
      <div className="h-3.5 w-32 rounded skeleton-shimmer" />
      <div className={`w-full rounded-lg skeleton-shimmer ${heightClass}`} />
      <div className="flex flex-col gap-2 mt-1">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-3 rounded skeleton-shimmer"
            style={{ width: `${88 - i * 16}%` }}
          />
        ))}
      </div>
    </div>
  )
}
