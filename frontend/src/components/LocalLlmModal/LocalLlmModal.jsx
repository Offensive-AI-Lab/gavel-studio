// LocalLlmModal — the one place the user manages AI generation access: set
// or swap a local Hugging Face model, jump to the OpenAI key modal to change
// that, and — when both are configured — pick which one actually runs
// generation. Mirrors OpenAiKeyModal.jsx's imperative open/saved-broadcast
// pattern; see that file for the reasoning.
//
// Mounted once from App.jsx. It has no props: it opens when something calls
// promptForLocalLlm() — either the "Set local LLM" button next to "Set API
// key" on a blocked AI step, or the persistent "AI provider" button in
// WizardModal's header (so it's reachable even when nothing is blocked, to
// swap models or providers mid-flow).
//
// Saving the model applies it to the running backend immediately — the next
// AI request loads it onto this machine's GPU (utils/device.py picks CUDA >
// MPS > CPU automatically, same as probe training). Local always ran ahead
// of OpenAI until this provider picker existed; now with both configured,
// utils/local_llm.py defaults the preference to OpenAI so setting a model
// here never silently steals traffic from an already-working key.
import { useCallback, useEffect, useRef, useState } from 'react';
import { FiCpu, FiKey, FiLoader } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import ReactiveButton from '../ReactiveButton/ReactiveButton';
import { notifyLocalLlmSaved, subscribeLocalLlmPrompt } from './localLlmPrompt';
import { promptForOpenAiKey } from '../OpenAiKeyModal/openAiKeyPrompt';
import {
    getAiProviderStatus, saveAiProvider, saveLocalLlm, clearLocalLlm,
    getLocalLlmLoadStatus, warmUpLocalLlm,
} from '../../api';

const WARMUP_POLL_MS = 1500;

const LocalLlmModal = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [model, setModel] = useState('');
    const [openaiConfigured, setOpenaiConfigured] = useState(false);
    const [localConfigured, setLocalConfigured] = useState(false);
    const [activeProvider, setActiveProvider] = useState('openai');
    const [saving, setSaving] = useState(false);
    const [switching, setSwitching] = useState(false);
    const [error, setError] = useState('');
    const [warmingUp, setWarmingUp] = useState(false);
    const [justLoaded, setJustLoaded] = useState(false); // brief "Ready ✓" pause before auto-close
    const [warmupElapsed, setWarmupElapsed] = useState(0);
    const [progress, setProgress] = useState(null); // {desc, n, total, percent} | null
    const [deviceLabel, setDeviceLabel] = useState('');
    const openRef = useRef(false);
    const onSavedRef = useRef(null);
    const pollRef = useRef(null);

    const stopPolling = useCallback(() => {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    }, []);
    useEffect(() => stopPolling, [stopPolling]); // clear any timer on unmount

    // Starts the background load and polls until it's ready (or fails), so
    // the modal can't hand control back to a chat step that would otherwise
    // silently block for minutes on a cold model load with no explanation.
    const waitForWarmup = useCallback(() => new Promise((resolve) => {
        setWarmingUp(true);
        setWarmupElapsed(0);
        setProgress(null);
        const started = Date.now();
        warmUpLocalLlm().catch(() => {});
        pollRef.current = setInterval(async () => {
            setWarmupElapsed(Math.round((Date.now() - started) / 1000));
            try {
                const res = await getLocalLlmLoadStatus();
                const d = res?.data || {};
                if (d.progress) setProgress(d.progress);
                if (d.state === 'ready') {
                    stopPolling(); setWarmingUp(false); setProgress(null);
                    setDeviceLabel(d.device || '');
                    resolve(true);
                } else if (d.state === 'error') {
                    stopPolling(); setWarmingUp(false); setProgress(null);
                    setError(d.detail || 'The model failed to load.');
                    resolve(false);
                }
                // 'loading' / 'not_loaded' / 'not_configured' → keep polling.
            } catch { /* transient — keep polling */ }
        }, WARMUP_POLL_MS);
    }), [stopPolling]);

    const refreshStatus = useCallback(async () => {
        try {
            const res = await getAiProviderStatus();
            const d = res?.data || {};
            setOpenaiConfigured(!!d.openai_configured);
            setLocalConfigured(!!d.local_configured);
            setActiveProvider(d.active_provider || 'openai');
            setModel(d.local_model || '');
        } catch { /* best-effort prefill */ }
        // Separate call: whether a model path is SET (above) vs whether it's
        // actually loaded onto the GPU in this backend process (below) — a
        // fresh restart has the former true and the latter not yet.
        try {
            const res = await getLocalLlmLoadStatus();
            const d = res?.data || {};
            setDeviceLabel(d.state === 'ready' ? (d.device || '') : '');
        } catch { /* best-effort */ }
    }, []);

    const open = useCallback(({ onSaved = null } = {}) => {
        onSavedRef.current = onSaved;
        if (!openRef.current) {
            setError('');
            setSaving(false);
            refreshStatus();
        }
        openRef.current = true;
        setIsOpen(true);
    }, [refreshStatus]);

    useEffect(() => subscribeLocalLlmPrompt(open), [open]);
    // The OpenAI key modal can be opened from inside this one ("Change key");
    // once it saves, our own status (openaiConfigured, and whether a provider
    // choice is now available) needs to catch up.
    useEffect(() => {
        if (isOpen) refreshStatus();
    }, [isOpen, refreshStatus]);

    const close = useCallback(() => {
        openRef.current = false;
        setIsOpen(false);
    }, []);

    const handleSaveModel = async () => {
        const value = model.trim();
        if (!value) {
            setError('Enter a Hugging Face model path to continue.');
            return;
        }
        setSaving(true);
        setError('');
        try {
            await saveLocalLlm(value);
            await refreshStatus();
            const ready = await waitForWarmup();
            if (!ready) { setSaving(false); return; }
            // Pause on the "Loaded on <device>" confirmation before closing —
            // without this the modal closes the instant it's ready and the
            // device line never has a chance to actually be seen.
            setJustLoaded(true);
            await new Promise((r) => setTimeout(r, 1400));
            setJustLoaded(false);
            const onSaved = onSavedRef.current;
            onSavedRef.current = null;
            close();
            notifyLocalLlmSaved();
            onSaved?.();
        } catch (e) {
            setSaving(false);
            setWarmingUp(false);
            const detail = e?.response?.data?.detail;
            setError(typeof detail === 'string' && detail
                ? detail
                : 'That model path was not accepted. Check it and try again.');
        }
    };

    const handleClearModel = async () => {
        setSaving(true);
        setError('');
        try {
            await clearLocalLlm();
            setModel('');
            await refreshStatus();
            notifyLocalLlmSaved();
        } catch {
            setError('Could not clear the local model. Try again.');
        } finally {
            setSaving(false);
        }
    };

    const handleSwitchProvider = async (provider) => {
        if (provider === activeProvider || switching) return;
        setSwitching(true);
        setError('');
        try {
            await saveAiProvider(provider);
            setActiveProvider(provider);
            notifyLocalLlmSaved();
        } catch (e) {
            const detail = e?.response?.data?.detail;
            setError(typeof detail === 'string' && detail ? detail : 'Could not switch provider.');
        } finally {
            setSwitching(false);
        }
    };

    const bothConfigured = openaiConfigured && localConfigured;
    const busy = saving || warmingUp;

    return (
        <GlassModal isOpen={isOpen} onClose={busy ? () => {} : close} title="AI generation settings">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>

                {warmingUp && (
                    <div style={warmupBannerStyle}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <FiLoader size={15} style={{ animation: 'gavel-spin 1s linear infinite', flexShrink: 0 }} />
                            <span>
                                {progress?.desc || 'Loading model onto the GPU'}
                                {progress ? ` — ${progress.n}/${progress.total} (${progress.percent}%)` : `… ${warmupElapsed}s`}
                            </span>
                        </div>
                        <div style={progressTrackStyle}>
                            <div style={{ ...progressFillStyle, width: `${progress?.percent ?? 0}%` }} />
                        </div>
                        <span style={{ opacity: 0.75, fontSize: '0.78rem' }}>
                            Larger models (7B+) can take a few minutes on first load.
                        </span>
                    </div>
                )}

                {bothConfigured && (
                    <div>
                        <label style={labelStyle}>Use for generation</label>
                        <div style={{ display: 'flex', gap: 8 }}>
                            {[
                                { id: 'openai', label: 'OpenAI' },
                                { id: 'local', label: 'Local model' },
                            ].map(({ id, label }) => (
                                <button
                                    key={id}
                                    type="button"
                                    onClick={() => handleSwitchProvider(id)}
                                    disabled={switching || busy}
                                    style={activeProvider === id ? providerBtnActiveStyle : providerBtnStyle}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                <div>
                    <div style={rowHeaderStyle}>
                        <label style={labelStyle}>OpenAI key</label>
                        <span style={statusPillStyle(openaiConfigured)}>
                            {openaiConfigured ? 'Set' : 'Not set'}
                        </span>
                    </div>
                    <button
                        type="button"
                        onClick={() => promptForOpenAiKey({ onSaved: refreshStatus })}
                        disabled={busy}
                        style={secondaryBtnStyle}
                    >
                        <FiKey size={13} /> {openaiConfigured ? 'Change key' : 'Set API key'}
                    </button>
                </div>

                <div>
                    <div style={rowHeaderStyle}>
                        <label style={labelStyle} htmlFor="local-llm-input">Local model (Hugging Face path)</label>
                        <span style={statusPillStyle(localConfigured)}>
                            {localConfigured ? 'Set' : 'Not set'}
                        </span>
                    </div>
                    <input
                        id="local-llm-input"
                        className="glass-input"
                        type="text"
                        placeholder="e.g. HuggingFaceTB/SmolLM2-135M-Instruct"
                        value={model}
                        onChange={(e) => { setModel(e.target.value); if (error) setError(''); }}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !busy) handleSaveModel(); }}
                        maxLength={256}
                        autoComplete="off"
                        disabled={busy}
                    />
                    <p style={{ margin: '8px 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                        Runs on this machine's own GPU. Loads (and is validated) as soon
                        as you save it below.
                    </p>
                    {deviceLabel && !warmingUp && (
                        <p style={{
                            margin: '6px 0 0', fontSize: '0.8rem', fontWeight: justLoaded ? 700 : 400,
                            color: '#86efac',
                        }}>
                            {justLoaded ? '✓ Ready — l' : 'L'}oaded on {deviceLabel}
                        </p>
                    )}
                    <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                        <ReactiveButton
                            label={warmingUp ? 'Loading…' : (justLoaded ? 'Ready ✓' : (saving ? 'Saving…' : (localConfigured ? 'Save new model' : 'Use this model')))}
                            onClick={handleSaveModel}
                            Icon={FiCpu}
                            disabled={busy}
                            style={{ flex: 1, justifyContent: 'center', ...(busy ? { opacity: 0.6, cursor: 'not-allowed' } : {}) }}
                        />
                        {localConfigured && (
                            <button type="button" onClick={handleClearModel} disabled={busy} style={secondaryBtnStyle}>
                                Clear
                            </button>
                        )}
                    </div>
                </div>

                {error && <div style={errorStyle} role="alert">{error}</div>}

                <button onClick={close} style={cancelBtnStyle} disabled={busy}>Done</button>
            </div>
            <style>{'@keyframes gavel-spin { to { transform: rotate(360deg); } }'}</style>
        </GlassModal>
    );
};

const labelStyle = { display: 'block', fontWeight: '600', fontSize: '0.9rem', color: '#cbd5e1' };
const rowHeaderStyle = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' };
const cancelBtnStyle = { padding: '12px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.18)', background: 'rgba(15, 23, 42, 0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: '600', fontSize: '1rem' };
const secondaryBtnStyle = { display: 'inline-flex', alignItems: 'center', gap: 6, padding: '10px 14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.22)', background: 'rgba(15, 23, 42, 0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: '600', fontSize: '0.85rem' };
const providerBtnStyle = { flex: 1, padding: '10px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.22)', background: 'rgba(15, 23, 42, 0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: '600', fontSize: '0.85rem' };
const providerBtnActiveStyle = { ...providerBtnStyle, border: '1px solid rgba(129, 140, 248, 0.55)', background: 'rgba(99, 102, 241, 0.22)', color: '#e0e7ff' };
const errorStyle = { background: 'rgba(239, 68, 68, 0.10)', border: '1px solid rgba(248, 113, 113, 0.35)', borderRadius: '10px', padding: '10px 14px', color: '#fca5a5', fontSize: '0.85rem', lineHeight: 1.5 };
const warmupBannerStyle = {
    display: 'flex', flexDirection: 'column', gap: 8,
    background: 'rgba(99, 102, 241, 0.12)', border: '1px solid rgba(129, 140, 248, 0.30)',
    borderRadius: '10px', padding: '12px 14px', color: '#c7d2fe', fontSize: '0.85rem', lineHeight: 1.5,
};
const progressTrackStyle = {
    height: 6, borderRadius: 999, background: 'rgba(148, 163, 184, 0.18)', overflow: 'hidden',
};
const progressFillStyle = {
    height: '100%', borderRadius: 999, background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
    transition: 'width 0.3s ease',
};
const statusPillStyle = (on) => ({
    fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999,
    background: on ? 'rgba(34, 197, 94, 0.16)' : 'rgba(148, 163, 184, 0.16)',
    color: on ? '#86efac' : '#94a3b8',
});

export default LocalLlmModal;
