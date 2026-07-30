import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FiDownloadCloud, FiAlertTriangle, FiLoader, FiCheckCircle } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import './ExportClassifierModal.css';
import { getExportPreflight, getExportActiveJob, startExport, getBundleJob, downloadBundleJob } from '../../api';

const TIER_META = {
    'full': { label: 'Full bundle', sub: 'Model + calibration + evaluation results.' },
    'model+calibration': { label: 'Model + calibration', sub: 'Trained model and its calibrated thresholds (ready to run).' },
    'model': { label: 'Model only', sub: 'Just the trained model — the receiver calibrates it themselves.' },
};
const TIER_ORDER = ['full', 'model+calibration', 'model'];
const POLL_MS = 1500;

const ExportClassifierModal = ({ isOpen, classifierId, classifierName, onClose }) => {
    // view: loading | blocked | ready | running | done | error
    const [view, setView] = useState('loading');
    const [preflight, setPreflight] = useState(null);
    const [tier, setTier] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [phase, setPhase] = useState('');
    const [error, setError] = useState(null);
    const [doneFilename, setDoneFilename] = useState(null);
    const [downloading, setDownloading] = useState(false);
    const pollRef = useRef(null);
    const downloadedRef = useRef(false);

    const stopPolling = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };

    const triggerDownload = useCallback(async (jid, filename) => {
        setDownloading(true);
        try {
            await downloadBundleJob(jid, filename);
        } catch {
            setError('The bundle is ready but the download failed — try Download again.');
        } finally {
            setDownloading(false);
        }
    }, []);

    // Poll a running job until it finishes.
    const startPolling = useCallback((jid) => {
        stopPolling();
        pollRef.current = setInterval(async () => {
            try {
                const { data: job } = await getBundleJob(jid);
                if (job.status === 'running') {
                    setPhase(job.phase || 'Working…');
                } else if (job.status === 'done') {
                    stopPolling();
                    setDoneFilename(job.filename || (job.result && job.result.filename) || null);
                    setView('done');
                    if (!downloadedRef.current) {
                        downloadedRef.current = true;
                        triggerDownload(jid, job.filename || (job.result && job.result.filename));
                    }
                } else if (job.status === 'error') {
                    stopPolling();
                    setError(job.error || 'Export failed.');
                    setView('error');
                }
            } catch (e) {
                if (e?.response?.status === 404) {
                    stopPolling();
                    setError('Export job not found — it may have been cleaned up. Try again.');
                    setView('error');
                }
                // else transient — keep trying
            }
        }, POLL_MS);
    }, [triggerDownload]);

    const load = useCallback(async () => {
        setView('loading');
        setError(null);
        downloadedRef.current = false;
        try {
            // Resume an in-flight / ready export first.
            try {
                const { data } = await getExportActiveJob(classifierId);
                if (data.job) {
                    setJobId(data.job.job_id);
                    if (data.job.status === 'running') {
                        setPhase(data.job.phase || 'Working…');
                        setView('running');
                        startPolling(data.job.job_id);
                        return;
                    }
                    if (data.job.status === 'done') {
                        downloadedRef.current = true; // don't auto-download a resumed job
                        setDoneFilename(data.job.filename || null);
                        setView('done');
                        return;
                    }
                }
            } catch { /* fall through to preflight */ }

            const { data: pf } = await getExportPreflight(classifierId);
            setPreflight(pf);
            if (pf.drift || (pf.blockers && pf.blockers.length > 0)) {
                setView('blocked');
                return;
            }
            const avail = pf.tiers_available || [];
            setTier(TIER_ORDER.find((t) => avail.includes(t)) || null);
            setView('ready');
        } catch (e) {
            setError(e?.response?.data?.detail || 'Could not check export readiness.');
            setView('error');
        }
    }, [classifierId, startPolling]);

    // Go back to the tier picker (e.g. to export a different version after one
    // already finished), ignoring any prior done/running job.
    const startOver = useCallback(async () => {
        stopPolling();
        downloadedRef.current = false;
        setJobId(null);
        setError(null);
        setView('loading');
        try {
            const { data: pf } = await getExportPreflight(classifierId);
            setPreflight(pf);
            if (pf.drift || (pf.blockers && pf.blockers.length > 0)) { setView('blocked'); return; }
            const avail = pf.tiers_available || [];
            setTier(TIER_ORDER.find((t) => avail.includes(t)) || null);
            setView('ready');
        } catch (e) {
            setError(e?.response?.data?.detail || 'Could not load.');
            setView('error');
        }
    }, [classifierId]);

    useEffect(() => {
        if (isOpen) load();
        return () => stopPolling();
    }, [isOpen, load]);

    const handleStart = async () => {
        if (!tier) return;
        setError(null);
        setView('running');
        setPhase('Starting…');
        downloadedRef.current = false;
        try {
            const { data } = await startExport(classifierId, tier);
            setJobId(data.job_id);
            startPolling(data.job_id);
        } catch (e) {
            setError(e?.response?.data?.detail || 'Could not start the export.');
            setView('error');
        }
    };

    const body = () => {
        if (view === 'loading') {
            return <div style={styles.center}><FiLoader className="gavel-spin" size={22} /><span style={{ marginLeft: 10 }}>Checking…</span></div>;
        }

        if (view === 'blocked' && preflight) {
            return (
                <div>
                    <div style={styles.errorBox}>
                        <FiAlertTriangle style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>{preflight.reason || (preflight.blockers && preflight.blockers[0])}</span>
                    </div>
                    {preflight.blockers && preflight.blockers.length > 1 && (
                        <ul style={styles.list}>{preflight.blockers.slice(1).map((b, i) => <li key={i}>{b}</li>)}</ul>
                    )}
                    <div style={styles.actions}><button style={styles.ghostBtn} onClick={onClose}>Close</button></div>
                </div>
            );
        }

        if (view === 'ready' && preflight) {
            const avail = preflight.tiers_available || [];
            const unpublished = (preflight.unpublished && preflight.unpublished.length) || 0;
            return (
                <div>
                    {unpublished > 0 && (
                        <div style={styles.noteBox}>
                            <FiAlertTriangle style={{ flexShrink: 0, marginTop: 2 }} />
                            <span>
                                {unpublished} rule{unpublished === 1 ? ' is' : 's are'} not in the public library —
                                bundles can only carry library rules. Contribute drafts via a gavel-rules pull
                                request first.
                            </span>
                        </div>
                    )}
                    <p style={styles.lead}>Choose what to include in the bundle:</p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {TIER_ORDER.filter((t) => avail.includes(t)).map((t) => (
                            <label key={t} style={{ ...styles.tierRow, ...(tier === t ? styles.tierRowActive : {}) }}>
                                <input type="radio" name="tier" checked={tier === t} onChange={() => setTier(t)} style={{ marginTop: 3 }} />
                                <span>
                                    <span style={styles.tierLabel}>{TIER_META[t].label}</span>
                                    <span style={styles.tierSub}>{TIER_META[t].sub}</span>
                                </span>
                            </label>
                        ))}
                    </div>
                    <div style={styles.actions}>
                        <button style={styles.ghostBtn} onClick={onClose}>Cancel</button>
                        <button style={styles.primaryBtn} onClick={handleStart} disabled={!tier}>
                            <FiDownloadCloud size={14} /> Export
                        </button>
                    </div>
                </div>
            );
        }

        if (view === 'running') {
            return (
                <div>
                    <div style={styles.center}>
                        <FiLoader className="gavel-spin" size={20} />
                        <span style={{ marginLeft: 10 }}>{phase || 'Working…'}</span>
                    </div>
                    <p style={styles.note}>
                        This runs on the server — you can close this window and it'll keep going.
                        Reopen Export to check on it or download when it's ready.
                    </p>
                    <div style={styles.actions}><button style={styles.ghostBtn} onClick={onClose}>Close</button></div>
                </div>
            );
        }

        if (view === 'done') {
            return (
                <div>
                    <div style={{ ...styles.center, color: '#6ee7b7' }}>
                        <FiCheckCircle size={20} />
                        <span style={{ marginLeft: 10 }}>Bundle ready{doneFilename ? `: ${doneFilename}` : ''}.</span>
                    </div>
                    {error && <div style={styles.errorBox}><FiAlertTriangle /> {error}</div>}
                    <div style={styles.actions}>
                        <button style={styles.ghostBtn} onClick={onClose}>Close</button>
                        <button style={styles.ghostBtn} onClick={startOver}>Export again</button>
                        <button style={styles.primaryBtn} onClick={() => triggerDownload(jobId, doneFilename)} disabled={downloading}>
                            {downloading ? <><FiLoader className="gavel-spin" size={14} /> Downloading…</> : <><FiDownloadCloud size={14} /> Download</>}
                        </button>
                    </div>
                </div>
            );
        }

        // error
        return (
            <div>
                <div style={styles.errorBox}><FiAlertTriangle /> {error || 'Something went wrong.'}</div>
                <div style={styles.actions}>
                    <button style={styles.ghostBtn} onClick={onClose}>Close</button>
                    <button style={styles.primaryBtn} onClick={load}>Try again</button>
                </div>
            </div>
        );
    };

    return (
        <GlassModal isOpen={isOpen} onClose={onClose} title={`Export “${classifierName || 'rule set'}”`}>
            {body()}
        </GlassModal>
    );
};

const styles = {
    center: { display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px 0', color: '#cbd5e1' },
    lead: { color: '#cbd5e1', marginBottom: 12 },
    note: { color: '#94a3b8', fontSize: '0.85rem', marginTop: 12 },
    list: { color: '#e2e8f0', margin: '6px 0 4px 0', paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 },
    errorBox: { display: 'flex', gap: 8, alignItems: 'flex-start', background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.4)', color: '#fca5a5', padding: '10px 12px', borderRadius: 8, fontSize: '0.9rem', marginTop: 12 },
    noteBox: { display: 'flex', gap: 8, alignItems: 'flex-start', background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.4)', color: '#fcd34d', padding: '10px 12px', borderRadius: 8, fontSize: '0.88rem', marginBottom: 14 },
    tierRow: { display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer', padding: '12px 14px', borderRadius: 10, border: '1px solid rgba(148,163,184,0.25)', background: 'rgba(148,163,184,0.06)' },
    tierRowActive: { borderColor: 'rgba(16,185,129,0.6)', background: 'rgba(16,185,129,0.12)' },
    tierLabel: { display: 'block', color: '#f1f5f9', fontWeight: 600 },
    tierSub: { display: 'block', color: '#94a3b8', fontSize: '0.82rem', marginTop: 2 },
    actions: { display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 },
    ghostBtn: { padding: '9px 16px', borderRadius: 8, cursor: 'pointer', background: 'transparent', color: '#cbd5e1', border: '1px solid rgba(148,163,184,0.35)' },
    primaryBtn: { display: 'inline-flex', alignItems: 'center', gap: 8, padding: '9px 18px', borderRadius: 8, cursor: 'pointer', background: 'rgba(16,185,129,0.9)', color: '#06281d', border: 'none', fontWeight: 600 },
};

export default ExportClassifierModal;
