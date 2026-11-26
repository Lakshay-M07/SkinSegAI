import React, { useEffect, useRef, useState } from 'react'
import { api, openSession, forgetSession, asBase64, download } from './api/v2'
import { fetchHealth } from './api/client'
import { TopHeader } from './components/TopHeader'
import { WorkstationView } from './components/WorkstationView'
import { ComparisonView } from './components/ComparisonView'
import { AboutProjectView } from './components/AboutProjectView'
import { BentoCard } from './components/BentoCard'

const BTN = 'px-3 py-1.5 rounded-lg bg-[#22D3EE]/15 text-[#22D3EE] border border-[#22D3EE]/40 text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer'
const INPUT = 'bg-[#0A1426] border border-[#475569]/70 rounded-lg p-2 text-xs text-[#CFD8E3]'
const dataUrl = b64 => b64 ? `data:image/png;base64,${b64}` : ''
const blank = {models: ['custom_unet','resnet34_unet'], threshold: .5, overlay_alpha: .35, mc_passes: 1, include_uncertainty: true, include_morphology: true, include_disagreement: true, include_image_quality: true}

function renderAssistantText(content) {
  return String(content || '')
    .replace(/^\s*- /gm, '• ')
    .split(/(\*\*[^*\n]+\*\*)/g)
    .map((part, index) =>
      part.startsWith('**') && part.endsWith('**')
        ? <strong key={index}>{part.slice(2, -2)}</strong>
        : part
    )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('workstation')
  const [health, setHealth] = useState(null)
  const [catalogue, setCatalogue] = useState([])
  const [samples, setSamples] = useState([])
  const [provider, setProvider] = useState('')
  const [context, setContext] = useState(null)
  const [draft, setDraft] = useState(blank)
  const [pending, setPending] = useState(null)
  const [selectedSample, setSelectedSample] = useState('')
  const [sourceLabel, setSourceLabel] = useState('')
  const [previewUrl, setPreviewUrl] = useState('')
  const [gtPreviewUrl, setGtPreviewUrl] = useState('')
  const [results, setResults] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [overlayStyle, setOverlayStyle] = useState('overlay')
  const [pair, setPair] = useState(['resnet34_unet','custom_unet'])
  const [history, setHistory] = useState([])
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const sid = useRef(null), ctx = useRef(null), lock = useRef(false), mounted = useRef(true)
  const current = () => ctx.current?.revision
  const call = (path, options = {}) => api(path, {...options, session: sid.current})
  const post = (path, json) => call(path, {method: 'POST', json})
  const name = id => catalogue.find(m => m.model_id === id)?.display_name || id || 'Model'
  function apply(c, sync = true) {
    ctx.current = c; setContext(c)
    if (sync) setDraft(c.configuration)
  }
  function clearVisuals() { setResults(null); setAnalysis(null) }
  function clearImage() { setPreviewUrl(''); setGtPreviewUrl(''); setSelectedSample(''); setSourceLabel(''); clearVisuals() }
  async function loadImage(c) {
    setPreviewUrl(''); setGtPreviewUrl('')
    if (!c.image_uploaded) return
    const q = `/image?expected_revision=${c.revision}&view=`
    const original = await call(q + 'original', {binary: true})
    const reference = c.ground_truth_available ? await call(q + 'reference', {binary: true}) : null
    setPreviewUrl(dataUrl(await asBase64(original)))
    if (reference) setGtPreviewUrl(dataUrl(await asBase64(reference)))
  }
  async function loadResults(c) {
    clearVisuals()
    if (!c.analysis_completed) return
    const payload = await call('/results')
    if (payload.revision !== c.revision) throw new Error('Results changed; refresh the session.')
    const items = []
    for (const r of payload.results) {
      const views = {}
      for (const view of r.available_views) {
        views[view] = await asBase64(await call(`/results/${r.model_id}/${view}?expected_revision=${c.revision}`, {binary: true}))
      }
      items.push({...r, views, metrics: r.ground_truth_metrics})
    }
    const detail = await call(`/analysis?expected_revision=${c.revision}`)
    setResults({...payload, results: items}); setAnalysis(detail.analysis)
    setPair(p => {
      const ids = items.map(x => x.model_id)
      const first = ids.includes(p[0]) ? p[0] : ids[0]
      const second = ids.includes(p[1]) && p[1] !== first ? p[1] : ids.find(x => x !== first) || p.find(x => x !== first) || ''
      return [first, second]
    })
  }
  async function refresh() {
    const body = await call('/context')
    apply(body.context); setProvider(body.provider); setPending(null)
    await loadImage(body.context); await loadResults(body.context)
    const h = await call('/history'); setHistory(h.history || [])
  }
  async function run(fn, recover = true) {
    if (lock.current) return
    lock.current = true; setBusy(true); setError('')
    try { await fn() }
    catch (err) {
      setError(err.message || String(err))
      if (recover && sid.current) {
        try { await refresh() }
        catch (e) {
          clearVisuals(); setPending(null)
          if (e.status === 404) { forgetSession(); sid.current = null; ctx.current = null; setContext(null); clearImage(); setHistory([]) }
        }
      }
    } finally { lock.current = false; setBusy(false) }
  }
  async function initialize() {
    const [models, list, h] = await Promise.all([api('/models'), api('/samples'), fetchHealth().catch(() => null)])
    const session = await openSession()
    if (!mounted.current) return
    sid.current = session.session_id; setProvider(session.provider); setHealth(h)
    setCatalogue(models.models); setSamples(list.samples)
    apply(session.context)
    if (session.context.image_uploaded) {
      setSourceLabel('Current session image'); await loadImage(session.context); await loadResults(session.context)
    }
    if (session.stage === 'awaiting_confirmation') setError('A previous proposal may still exist. Prepare a new proposal before approving, or reset the session.')
    const hist = await call('/history'); setHistory(hist.history || [])
  }
  useEffect(() => {
    mounted.current = true
    run(initialize, false)
    return () => { mounted.current = false }
  }, [])
  async function acceptImage(body, label, sample = '') {
    apply(body.context); setPending(null); setHistory([]); clearVisuals()
    setSelectedSample(sample); setSourceLabel(label)
    await loadImage(body.context)
  }
  function chooseSample(id) {
    if (!id) { reset(); return }
    run(async () => acceptImage(await post('/sample', {sample_id: id, expected_revision: current()}), id, id))
  }
  async function upload(file, reference = null, label = file.name) {
    const form = new FormData(); form.append('image', file, file.name || 'image.png')
    if (reference) form.append('ground_truth', reference)
    form.append('expected_revision', String(current()))
    await acceptImage(await call('/image', {method: 'POST', form}), label)
  }
  function uploadReference(file) {
    run(async () => {
      if (!ctx.current?.image_uploaded) throw new Error('Select an image first.')
      const image = await call(`/image?expected_revision=${current()}&view=original`, {binary: true})
      await upload(new File([image], 'image.png', {type: 'image/png'}), file, sourceLabel)
    })
  }
  function edit(key, value) { setDraft(d => ({...d, [key]: value})) }
  function prepare() {
    run(async () => {
      const b = await post('/configuration', {...draft, expected_revision: current()})
      apply(b.context, false); setPending(b.proposal); setDraft(b.proposal.configuration)
    })
  }
  function confirm(approved) {
    run(async () => {
      if (!pending?.confirmation_token) throw new Error('Prepare a new proposal first.')
      const b = await post('/confirm', {confirmation_token: pending.confirmation_token, approved})
      setPending(null); apply(b.context)
      if (approved) { await loadResults(b.context); setHistory((await call('/history')).history || []) }
    })
  }
  function chat(e) {
    e.preventDefault()
    const text = message.trim(); if (!text) return
    run(async () => {
      const b = await post('/chat', {message: text, expected_revision: current()})
      setMessage(''); apply(b.context, false)
      setHistory(old => [...old, {role:'user', content:text}, {role:'assistant', content:b.message + (b.used_fallback ? '\n[Offline fallback used]' : '')}].slice(-40))
      if (b.proposal) { setPending(b.proposal); setDraft(b.proposal.configuration) }
    })
  }
  function reset() {
    run(async () => {
      const b = await post('/reset', {expected_revision: current()})
      apply(b.context); setPending(null); setHistory([]); clearImage()
    })
  }
  function release() { run(async () => { const b = await post('/release', {expected_revision:current()}); apply(b.context, false) }) }
  function exportReport() { run(async () => download(await call(`/report?expected_revision=${current()}`, {binary:true}), 'SkinSegAI_Report.zip')) }
  function reconnect() {
    run(async () => {
      if (sid.current) await call('/session', {method:'DELETE'}).catch(e => {if(e.status !== 404) throw e})
      forgetSession(); sid.current = null; ctx.current = null; setContext(null); setPending(null); setHistory([]); clearImage()
      await initialize()
    }, false)
  }
  const modelOptions = catalogue.length ? catalogue : []
  const viewKey = {overlay:'overlay', contour:'boundary', mask:'mask'}[overlayStyle]
  const renderResult = id => {
    const r = results?.results.find(x => x.model_id === id)
    return r ? {...r, overlay_base64:r.views[viewKey], mask_base64:r.views.mask} : null
  }
  const a = renderResult(pair[0]), b = renderResult(pair[1])
  const segmentResult = results ? {
    ...results, width:results.image.width, height:results.image.height,
    threshold:results.configuration.threshold, opacity:results.configuration.overlay_alpha, overlay_style:overlayStyle,
    unet:a, grabcut:b, uncertainty_heatmap_base64:a?.views.uncertainty, uncertainty_overlay_base64:a?.views.uncertainty_overlay, error_map_base64:a?.views.error_map,
  } : null
  const benchmark = {available:catalogue.some(m => m.benchmark), summary:catalogue.filter(m => m.benchmark).map(m => ({...m.benchmark, model_id:m.model_id, algorithm:m.display_name}))}
  const pairPicker = <div className="flex flex-wrap items-center gap-2 text-xs text-[#94A3B8]">
    {pair.map((id, i) => <label key={i}>Panel {i+1} <select className={INPUT} value={id} onChange={e => setPair(p => p.map((v,j) => j===i ? e.target.value : v))}>{modelOptions.filter(m => m.model_id !== pair[1-i]).map(m => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}</select></label>)}
  </div>
  const controlDisabled = busy || !context || !!pending
  return (
    <div className="min-h-screen flex flex-col bg-[#0A1426] text-[#CFD8E3]">
      <TopHeader activeTab={activeTab} onTabChange={setActiveTab} health={health} healthLoading={!health && busy} />
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-5">
        {error && <div role="alert" className="mb-3 text-xs text-[#FB923C]">{error}</div>}
        <div key={activeTab} className="tab-panel-enter space-y-4">
          {activeTab === 'workstation' && <>
            {pairPicker}
            {results && <p className="text-xs text-[#94A3B8]">Displayed results use approved threshold {results.configuration.threshold.toFixed(2)} and opacity {results.configuration.overlay_alpha.toFixed(2)}. New settings require approval.</p>}
            <fieldset disabled={controlDisabled} className="min-w-0">
              <WorkstationView samples={samples} samplesLoading={!context && busy} selectedSample={selectedSample}
                onSelectSample={chooseSample} previewUrl={previewUrl} gtPreviewUrl={gtPreviewUrl} sourceLabel={sourceLabel}
                onUploadImage={file => run(() => upload(file))} onUploadGroundTruth={uploadReference} healthLoading={!context && busy}
                threshold={draft.threshold} onThresholdChange={v => edit('threshold',v)} overlayStyle={overlayStyle} onOverlayStyleChange={setOverlayStyle}
                opacity={draft.overlay_alpha} onOpacityChange={v => edit('overlay_alpha',v)} includeUncertainty={draft.include_uncertainty}
                onIncludeUncertaintyChange={v => edit('include_uncertainty',v)} segmentResult={segmentResult} segmentLoading={busy} segmentError={error}
                resultRevision={context?.revision || 0} primaryName={name(pair[0])} secondaryName={name(pair[1])}
                extraControls={<div className="mt-3 space-y-2"><div className="flex flex-wrap gap-3 text-xs">{modelOptions.map(m => <label key={m.model_id}><input type="checkbox" checked={draft.models.includes(m.model_id)} onChange={e => edit('models',e.target.checked ? [...draft.models,m.model_id] : draft.models.filter(id => id!==m.model_id))}/> {m.display_name}</label>)}</div>
                <div className="flex flex-wrap gap-3 text-xs">{[['include_morphology','Morphology'],['include_disagreement','Agreement'],['include_image_quality','Image quality']].map(([k,label]) => <label key={k}><input type="checkbox" checked={draft[k]} onChange={e => edit(k,e.target.checked)}/> {label}</label>)}<label>Passes <input className={INPUT+' w-16'} type="number" min="1" max="20" value={draft.mc_passes} onChange={e => edit('mc_passes',Number(e.target.value))}/></label></div>
                <button className={BTN} disabled={!context?.image_uploaded || !draft.models.length || !Number.isInteger(draft.mc_passes) || draft.mc_passes<1 || draft.mc_passes>20} onClick={prepare}>Prepare segmentation</button> <a href="#assistant" className="text-xs text-[#22D3EE]">Open AI assistant</a></div>} />
            </fieldset>
            <div id="assistant"/>
            <BentoCard title="AI Assistant" subtitle="Local model guidance and verified segmentation analysis" badge={provider || 'Disconnected'} badgeTone={provider==='ollama'?'aqua':'neutral'}>
              {provider==='offline' && <p className="text-xs text-[#FB923C] mb-2">Ollama unavailable or offline session selected. Start Ollama, then reconnect for natural-language chat.</p>}
              <div aria-live="polite" className="max-h-72 overflow-auto space-y-2 text-xs mb-3">{history.map((h,i) => <div key={i} className="rounded-lg bg-[#0A1426] p-3 whitespace-pre-wrap break-words"><strong className="text-[#22D3EE]">{h.role==='user'?'You':'Assistant'}</strong>{'\n'}{h.role === 'assistant' ? renderAssistantText(h.content) : h.content}</div>)}</div>
              <form className="flex gap-2" onSubmit={chat}><input className={INPUT+' flex-1 min-w-0'} value={message} onChange={e => setMessage(e.target.value)} maxLength={4000} placeholder="Ask about models, settings or results…" disabled={busy || !context}/><button className={BTN} disabled={busy || !context || !message.trim()}>Send</button></form>
              {pending && <div className="mt-3 p-3 rounded-lg border border-[#22D3EE]/40 text-xs space-y-2"><strong>Pending approval — no inference has run for this proposal</strong><pre className="whitespace-pre-wrap break-words">{JSON.stringify(pending.configuration,null,2)}</pre>{pending.warnings?.map(w => <p key={w}>{w}</p>)}<div className="flex gap-2"><button className={BTN} disabled={busy} onClick={() => confirm(true)}>Approve &amp; run</button><button className={BTN} disabled={busy} onClick={() => confirm(false)}>Cancel</button></div></div>}
              <div className="mt-3 flex flex-wrap gap-2"><button className={BTN} disabled={busy || !context?.analysis_completed} onClick={exportReport}>Download ZIP report</button><button className={BTN} disabled={busy || !context} onClick={release}>Release cached models</button><button className={BTN} disabled={busy || !context} onClick={reset}>Reset session</button><button className={BTN} disabled={busy} onClick={reconnect}>Reconnect assistant (clears session)</button></div>
              {busy && <p className="mt-2 text-xs text-[#94A3B8]">Processing request…</p>}
            </BentoCard>
            {analysis && <BentoCard title="Advanced Analysis" subtitle="Verified Python analysis, not a medical diagnosis">{['uncertainty','model_agreement','threshold_sensitivity','image_quality'].map(k => <details key={k} className="text-xs py-2 border-b border-[#475569]/50"><summary className="cursor-pointer">{k.replaceAll('_',' ')}</summary><pre className="whitespace-pre-wrap break-words mt-2">{JSON.stringify(analysis[k],null,2)}</pre></details>)}</BentoCard>}
          </>}
          {activeTab === 'comparison' && <>{pairPicker}<ComparisonView benchmark={benchmark} benchmarkLoading={!catalogue.length && busy} segmentResult={segmentResult} sourceLabel={sourceLabel} primaryName={name(pair[0])} secondaryName={name(pair[1])} primaryId={pair[0]} secondaryId={pair[1]}/><button className={BTN} disabled={busy || !context?.analysis_completed} onClick={exportReport}>Download ZIP report</button></>}
          {activeTab === 'about' && <AboutProjectView health={health}/>}
        </div>
      </main>
      <footer className="border-t border-[#475569]/60 py-3 px-4 sm:px-6"><p className="max-w-[1440px] mx-auto text-center text-[11px] text-[#94A3B8]">Research use only — SkinSegAI is an academic segmentation workstation and is not a medical diagnostic tool.</p></footer>
    </div>
  )
}
