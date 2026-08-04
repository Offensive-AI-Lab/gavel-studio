import { useState, useRef, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FiHome, FiShield, FiFileText, FiSend, FiSettings, FiRadio, FiTrash2, FiMessageSquare, FiDatabase, FiServer, FiAlertTriangle, FiRefreshCw, FiSliders } from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import {
    analyzeRealtime, analyzeStored, listSampleGroups, getSampleGroup, getClassifierDetails,
    startRealtimeSession, getRealtimeSessionStatus, realtimeSessionKeepalive, endRealtimeSession,
    endRealtimeSessionUnload, sessionAnalyzeStored, sessionAnalyzeLive,
} from '../api';
import { useTutorialContent } from '../contexts/TutorialContext';
import { groupProgressLabel } from '../utils/ruleLogic';
import '../css/RealtimeViewer.css';

// Elapsed-time formatter (M:SS) for the first-time model-load notice.
const fmtSecs = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

// Bright, dark-theme-friendly categorical palette so CE dots, chart lines,
// legend swatches, token tints and threshold names all read clearly — and stay
// tellable apart — against the dark monitor background. Ordered so neighbouring
// entries differ strongly in hue.
const CE_COLORS = [
    '#a78bfa', '#34d399', '#f472b6', '#fbbf24', '#60a5fa', '#fb923c',
    '#22d3ee', '#a3e635', '#f87171', '#c084fc', '#2dd4bf', '#fde047',
    '#818cf8', '#4ade80', '#e879f9', '#38bdf8', '#fdba74', '#5eead4',
    '#fca5a5', '#bef264', '#93c5fd', '#d8b4fe', '#f9a8d4', '#fcd34d',
];

// Color for the CE at position `idx`. The first colors come from the curated
// palette above (hand-separated). Past it we space hue by the GOLDEN ANGLE
// (~137.5°) so consecutive colors always land far apart and the sequence never
// clusters, however many CEs there are — and we cycle three saturation/lightness
// bands so two colors that fall near the same hue still differ in brightness.
// Depends ONLY on idx, so a CE keeps ONE color everywhere (legend, token tints,
// chart lines, threshold list) no matter how the helper is called.
function ceColor(idx) {
    if (idx < 0) return '#94a3b8';
    if (idx < CE_COLORS.length) return CE_COLORS[idx];
    const hue = (idx * 137.508) % 360;
    const band = idx % 3;
    const sat = [85, 72, 95][band];
    const light = [66, 56, 72][band];
    return `hsl(${hue}, ${sat}%, ${light}%)`;
}

// Translucent version of a ceColor — works for BOTH the hex palette and the
// generated hsl() colors (you can't just append a hex-alpha suffix to hsl()).
function withAlpha(color, alpha) {
    if (!color) return undefined;
    if (color.startsWith('hsl(')) {
        return color.replace('hsl(', 'hsla(').replace(')', `, ${alpha})`);
    }
    const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255).toString(16).padStart(2, '0');
    return `${color}${a}`;
}

// The stored sample carries the FULL conversation in `messages`; the
// `user_preview` field is truncated to 160 chars for the picker list only.
// For the analysis pane show the complete user turn(s) so no words are hidden.
function storedUserText(sample) {
    if (!sample) return '';
    const msgs = sample.messages;
    if (Array.isArray(msgs) && msgs.length) {
        const userTurns = msgs
            .filter((m) => (m.role || '').toLowerCase() === 'user')
            .map((m) => m.content || '')
            .filter(Boolean);
        if (userTurns.length) return userTurns.join('\n\n');
    }
    return sample.user_preview || '';
}

export default function RealtimeViewer() {
    const { classifierId } = useParams();
    const navigate = useNavigate();

    const [guardrail, setGuardrail] = useState(null);
    const [mode, setMode] = useState('live'); // 'live' | 'stored'
    const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
    const [showSettings, setShowSettings] = useState(false);
    const [maxTokens, setMaxTokens] = useState(128);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [analyses, setAnalyses] = useState([]);
    // Highlighted CE set — clicking sidebar CEs toggles membership; an empty
    // set means "no focus" and every CE is shown.
    const [selectedCEs, setSelectedCEs] = useState(() => new Set());

    // --- Stored (test-samples) mode state ---
    const [sampleGroups, setSampleGroups] = useState(null); // null = not loaded
    const [activeGroupKey, setActiveGroupKey] = useState(null);
    const [groupSamples, setGroupSamples] = useState([]);
    const [activeSampleIdx, setActiveSampleIdx] = useState(null);
    const [storedAnalysis, setStoredAnalysis] = useState(null);
    const [storedLoading, setStoredLoading] = useState(false);
    // Warm remote session lifecycle: starting | queued | loading | ready | dead | error |
    // uncalibrated | local.
    // 'local' = no remote GPU worker is configured, so we fall back to loading the
    // model on THIS machine (only viable on a capable PC) via the legacy endpoints.
    const [sessionState, setSessionState] = useState('starting');
    const [sessionError, setSessionError] = useState(null);
    const [sessionSecs, setSessionSecs] = useState(0);
    const [sessionEpoch, setSessionEpoch] = useState(0);   // bump to (re)start the session

    const chatEndRef = useRef(null);
    const inputRef = useRef(null);
    const sessionActiveRef = useRef(false);   // a remote session is live (→ end it on exit)
    const pendingStartRef = useRef(false);    // a start POST is in flight (its session must still be ended if we bail)
    const autoRetryRef = useRef(0);           // mid-session crash auto-restarts used (capped)

    const usingSession = sessionState !== 'local';
    const sessionReady = sessionState === 'ready' || sessionState === 'local';
    const sessionBusy = sessionState === 'starting' || sessionState === 'queued' || sessionState === 'loading';
    const sessionDead = sessionState === 'dead' || sessionState === 'error';
    // Off-box wording for the banner/pill.
    const provNoun = 'Remote GPU';    // pill label
    const provWhere = 'remote GPU';   // "...on the X"

    useEffect(() => {
        getClassifierDetails(classifierId).then(res => setGuardrail(res.data)).catch(() => {});
    }, [classifierId]);

    // Start a warm remote session (the model loads ONCE on the remote GPU, so this
    // works on any client PC). Falls back to loading the model on THIS machine only
    // if no worker is configured (a capable-PC-only path).
    const beginSession = useCallback(() => {
        let cancelled = false;
        sessionActiveRef.current = false;
        pendingStartRef.current = true;   // a start POST is now in flight
        setSessionError(null);
        setSessionState('starting');
        startRealtimeSession(classifierId)
            .then(res => {
                pendingStartRef.current = false;
                // The backend signals 'local' (NOT an error) when the warm session
                // can't be used — worker not configured/reachable, the warm job
                // couldn't start, or a locally-uploaded model. We just run the model
                // on this machine; if it can't load, the analyze call reports that.
                if (res.data?.fallback === 'local') {
                    if (!cancelled) setSessionState('local');
                    return;
                }
                // A real off-box session now exists.
                sessionActiveRef.current = true;
                if (cancelled) {
                    // We already left realtime while the start was in flight — end
                    // the just-created session so its GPU job isn't orphaned.
                    sessionActiveRef.current = false;
                    endRealtimeSession(classifierId).catch(() => {});
                    return;
                }
                setSessionState('queued');
            })
            .catch(err => {
                pendingStartRef.current = false;
                if (cancelled) return;
                // 409 = the rule set isn't calibrated (or calibration is still
                // running). Its own state, not a generic failure: monitoring
                // uncalibrated would silently score every CE at 0.5, so the
                // backend refuses and we point the user at Calibration.
                setSessionState(err.response?.status === 409 ? 'uncalibrated' : 'error');
                setSessionError(err.response?.data?.detail || 'Failed to start the realtime session.');
            });
        return () => { cancelled = true; };
    }, [classifierId]);

    // Open the session on entering realtime / on restart (sessionEpoch); CLOSE it
    // on leaving (route change / unmount). The start is DEFERRED a tick so a
    // TRANSIENT mount — React StrictMode's dev double-invoke (mount→unmount→mount),
    // or a fast in/out navigation — cancels BEFORE we submit a remote job. Without
    // this, the throwaway mount starts a session and its cleanup immediately ends
    // it (and, since sessions are keyed by classifier, can even kill the real one).
    useEffect(() => {
        let cancelled = false;
        let cancelStart = null;
        const timer = setTimeout(() => {
            if (!cancelled) cancelStart = beginSession();
        }, 150);
        return () => {
            cancelled = true;
            clearTimeout(timer);
            if (cancelStart) cancelStart();
            if (sessionActiveRef.current || pendingStartRef.current) {
                sessionActiveRef.current = false;
                pendingStartRef.current = false;
                endRealtimeSession(classifierId).catch(() => {});
            }
        };
    }, [beginSession, classifierId, sessionEpoch]);

    // Tab close / refresh / browser quit — best-effort teardown that survives unload.
    // (Backstop if it doesn't fire: the backend stale-session sweep + the job's own
    // idle timeout reclaim the GPU.)
    useEffect(() => {
        const onLeave = () => {
            if (sessionActiveRef.current || pendingStartRef.current) endRealtimeSessionUnload(classifierId);
        };
        window.addEventListener('pagehide', onLeave);
        return () => window.removeEventListener('pagehide', onLeave);
    }, [classifierId]);

    // While the session is coming up, poll until it's ready (or dies / queues).
    useEffect(() => {
        if (sessionState !== 'queued' && sessionState !== 'loading') return;
        let alive = true;
        const id = setInterval(() => {
            getRealtimeSessionStatus(classifierId).then(res => {
                if (!alive) return;
                const st = res.data?.status;   // backend returns {status: ...}
                if (st === 'ready') setSessionState('ready');
                else if (st === 'queued') setSessionState('queued');
                else if (st === 'loading') setSessionState('loading');
                else if (st === 'dead') { sessionActiveRef.current = false; setSessionState('dead'); setSessionError(res.data?.error || 'The session ended unexpectedly.'); }
                else if (st === 'stopped' || st === 'none') { sessionActiveRef.current = false; setSessionState('dead'); setSessionError('The session stopped before it was ready.'); }
            }).catch(() => {});
        }, 3000);
        return () => { alive = false; clearInterval(id); };
    }, [sessionState, classifierId]);

    // While ready: keepalive (so the job isn't reclaimed) + watch for a mid-session crash.
    useEffect(() => {
        if (sessionState !== 'ready') return;
        let alive = true;
        const tick = () => {
            realtimeSessionKeepalive(classifierId).catch(() => {});
            getRealtimeSessionStatus(classifierId).then(res => {
                if (!alive) return;
                const st = res.data?.status;   // backend returns {status: ...}
                if (st === 'dead' || st === 'stopped' || st === 'none') {
                    sessionActiveRef.current = false;
                    setSessionState('dead');
                    setSessionError(res.data?.error || 'The session ended unexpectedly.');
                }
            }).catch(() => {});
        };
        const id = setInterval(tick, 20000);
        return () => { alive = false; clearInterval(id); };
    }, [sessionState, classifierId]);

    // Lazy-load the conversation groups the first time Test Samples mode opens.
    useEffect(() => {
        if (mode !== 'stored' || sampleGroups !== null) return;
        listSampleGroups(classifierId)
            .then(res => setSampleGroups(res.data?.groups || []))
            .catch(() => setSampleGroups([]));
    }, [mode, sampleGroups, classifierId]);

    const pageHelp = {
        title: 'Realtime Monitor',
        summary: 'Watch the trained rule set score CEs token by token. Two modes: chat live with the LLM, or browse the stored dialogues each CE was trained/calibrated on and see exactly which words trip which CEs.',
        sections: [
            {
                heading: 'Two modes',
                bullets: [
                    'Live Chat — type a message; the LLM replies and the rule set scores its reply.',
                    'Test Samples — pick a CE the rule set was trained on, pick one of its stored dialogues, and see the per-token activations + rule firing.',
                ],
            },
            {
                heading: 'How to read it',
                bullets: [
                    'Each token is tinted by the strongest CE active on it; hover for the exact per-CE probabilities.',
                    'The line chart shows every CE\'s probability across the response; dashed lines are the calibrated thresholds.',
                    'A CE "fires" (windowed + patience) when it persistently crosses its threshold; rules fire when their condition is satisfied.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    useEffect(() => {
        if (mode !== 'live') return;
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        if (!loading) inputRef.current?.focus();
    }, [messages, analyses, loading, mode]);

    // Elapsed-seconds counter while the remote session is starting up.
    useEffect(() => {
        if (!sessionBusy) { setSessionSecs(0); return; }
        const t0 = Date.now();
        const id = setInterval(() => setSessionSecs(Math.floor((Date.now() - t0) / 1000)), 1000);
        return () => clearInterval(id);
    }, [sessionBusy]);

    const handleSend = async () => {
        const text = input.trim();
        if (!text || loading || !sessionReady) return;
        setMessages(prev => [...prev, { role: 'user', content: text }]);
        setInput('');
        setLoading(true);
        try {
            const history = messages.length > 0 ? messages : undefined;
            const args = { system_prompt: systemPrompt, user_message: text, history, max_new_tokens: maxTokens };
            const res = usingSession
                ? await sessionAnalyzeLive(classifierId, args)
                : await analyzeRealtime(classifierId, args);
            const data = res.data;
            setMessages(prev => [...prev, { role: 'assistant', content: data.generated_text }]);
            setAnalyses(prev => [...prev, data]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.response?.data?.detail || 'Analysis failed'}` }]);
        } finally { setLoading(false); }
    };

    const handleClear = () => { setMessages([]); setAnalyses([]); setSelectedCEs(new Set()); };

    // Mid-session resilience: if the warm session dies, automatically re-establish
    // it down the failover ladder (the backend's session/start walks
    // worker → local). Capped so a persistently-failing setup eventually
    // surfaces the manual "Restart session" button instead of looping forever.
    // 'error' (a hard start failure, e.g. not-trained) is NOT auto-retried.
    const MAX_AUTO_RESTARTS = 3;
    useEffect(() => {
        if (sessionState === 'ready' || sessionState === 'local') {
            autoRetryRef.current = 0;                            // healthy → reset the budget
            return;
        }
        if (sessionState !== 'dead') return;
        if (autoRetryRef.current >= MAX_AUTO_RESTARTS) return;   // exhausted → manual button
        autoRetryRef.current += 1;
        const n = autoRetryRef.current;
        setSessionError(`Session lost — reconnecting on the next available GPU (attempt ${n}/${MAX_AUTO_RESTARTS})…`);
        const t = setTimeout(() => setSessionEpoch(e => e + 1), 1500);
        return () => clearTimeout(t);
    }, [sessionState]);

    const restartSession = useCallback(async () => {
        // End first (AWAIT), then re-run the mount effect via the epoch bump so the
        // fresh start is wired to the unmount cleanup and never reuses a session
        // that's mid-teardown.
        autoRetryRef.current = 0;             // a manual restart earns a fresh auto-retry budget
        sessionActiveRef.current = false;
        pendingStartRef.current = false;
        await endRealtimeSession(classifierId).catch(() => {});
        setSessionEpoch(e => e + 1);
    }, [classifierId]);

    const pickGroup = useCallback(async (group) => {
        setActiveGroupKey(group.key);
        setGroupSamples([]);
        setActiveSampleIdx(null);
        setStoredAnalysis(null);
        try {
            const res = await getSampleGroup(classifierId, group.key);
            setGroupSamples(res.data?.samples || []);
        } catch { setGroupSamples([]); }
    }, [classifierId]);

    const pickSample = useCallback(async (sample) => {
        if (!sessionReady) return;
        setActiveSampleIdx(sample.index);
        setStoredAnalysis(null);
        setStoredLoading(true);
        try {
            const res = usingSession
                ? await sessionAnalyzeStored(classifierId, sample.messages)
                : await analyzeStored(classifierId, sample.messages);
            setStoredAnalysis({ ...res.data, _sample: sample });
        } catch (err) {
            setStoredAnalysis({ _error: err.response?.data?.detail || 'Analysis failed' });
        } finally { setStoredLoading(false); }
    }, [classifierId, sessionReady, usingSession]);

    // Sidebar derives from whichever mode is active.
    const activeAnalyses = mode === 'live' ? analyses : (storedAnalysis && !storedAnalysis._error ? [storedAnalysis] : []);
    const allCEs = [];
    const ceSet = new Set();
    activeAnalyses.forEach(a => {
        if (a.labels) Object.keys(a.labels).forEach(ce => { if (!ceSet.has(ce)) { ceSet.add(ce); allCEs.push(ce); } });
    });
    const latestAnalysis = activeAnalyses.length > 0 ? activeAnalyses[activeAnalyses.length - 1] : null;
    const triggeredCEs = new Set(latestAnalysis?.triggered_ces || []);
    const latestRuleTriggers = latestAnalysis?.rule_triggers || [];

    return (
        <Layout raw>
            <div className="rtv">
                {/* Header */}
                <div className="rtv-header">
                    <Breadcrumb items={[
                        { label: 'Hub', icon: FiHome, to: '/workspace' },
                        { label: 'Rule Sets', icon: FiShield, to: '/guardrails' },
                        { label: guardrail?.name || 'Rule Set', icon: FiFileText, to: `/classifiers/${classifierId}/rules` },
                        { label: 'Monitor', icon: FiRadio },
                    ]} style={{ marginBottom: 0 }} />
                    <div style={{ flex: 1 }}>
                        <h2 className="rtv-title"><FiRadio size={18} /> Realtime CE Monitor</h2>
                        {guardrail && <div className="rtv-subtitle">Rule Set: {guardrail.name || `#${classifierId}`}</div>}
                    </div>
                    {/* Mode toggle */}
                    <div className="rtv-mode-toggle">
                        <button className={mode === 'live' ? 'active' : ''} onClick={() => setMode('live')}>
                            <FiMessageSquare size={13} /> Live Chat
                        </button>
                        <button className={mode === 'stored' ? 'active' : ''} onClick={() => setMode('stored')}>
                            <FiDatabase size={13} /> Test Samples
                        </button>
                    </div>
                    <div className="rtv-header-actions">
                        {usingSession && (
                            <span className={`rtv-session-pill ${sessionState}`}
                                title={`Realtime runs on the ${provWhere} — no model is loaded on this PC.`}>
                                <FiServer size={13} />
                                {sessionState === 'ready' ? `${provNoun} session`
                                    : sessionBusy ? 'Starting…'
                                    : sessionDead ? 'Session ended' : provNoun}
                            </span>
                        )}
                        {sessionDead && (
                            <button className="rtv-restart-btn" onClick={restartSession} title="Start a fresh session">
                                <FiRefreshCw size={13} /> Restart
                            </button>
                        )}
                        {mode === 'live' && (
                            <>
                                <button className="rtv-icon-btn" onClick={() => setShowSettings(!showSettings)} title="Settings">
                                    <FiSettings size={15} />
                                </button>
                                <button className="rtv-icon-btn danger" onClick={handleClear} title="Clear conversation">
                                    <FiTrash2 size={15} />
                                </button>
                            </>
                        )}
                    </div>
                </div>

                {usingSession && sessionBusy && (
                    <div className="rtv-firstload" role="status" aria-live="polite">
                        <FiServer className="rtv-firstload-spin" size={20} />
                        <div className="rtv-firstload-text">
                            <strong>
                                {sessionState === 'queued'
                                    ? 'Uploading the model to the remote GPU…'
                                    : sessionState === 'loading' ? `Loading the model on the ${provWhere}…`
                                    : 'Starting the remote GPU session…'}
                            </strong>
                            <span>
                                The target model loads once on the {provWhere} (not on this PC) — this can
                                take a minute or two. Once it&apos;s ready, every conversation is classified
                                in seconds, and it works on any machine.
                                {sessionSecs > 0 && <> &nbsp;·&nbsp; {fmtSecs(sessionSecs)} elapsed</>}
                            </span>
                            <div className="rtv-firstload-bar"><div className="rtv-firstload-bar-fill" /></div>
                        </div>
                    </div>
                )}

                {sessionState === 'uncalibrated' && (
                    <div className="rtv-firstload is-error" role="alert">
                        <FiAlertTriangle size={20} style={{ color: '#fbbf24', flexShrink: 0 }} />
                        <div className="rtv-firstload-text">
                            {/* Two ways to land here — no calibration yet, or a policy edited
                                since training — so the heading stays general and the backend's
                                message says which. Both CTAs are offered rather than guessing. */}
                            <strong>Monitoring is blocked.</strong>
                            <span>{sessionError} Scoring on thresholds that don&apos;t match the
                            current model would look functional and mean nothing.</span>
                        </div>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <button className="rtv-restart-btn" onClick={() => navigate(`/classifiers/${classifierId}/evaluate`)}>
                                <FiSliders size={13} /> Go to Calibration
                            </button>
                            <button className="rtv-restart-btn" onClick={() => navigate(`/classifiers/${classifierId}/rules`)}>
                                <FiShield size={13} /> Go to Rule Set
                            </button>
                            <button className="rtv-restart-btn" onClick={restartSession}><FiRefreshCw size={13} /> Retry</button>
                        </div>
                    </div>
                )}

                {sessionDead && (
                    <div className="rtv-firstload is-error" role="alert">
                        <FiAlertTriangle size={20} style={{ color: '#f87171', flexShrink: 0 }} />
                        <div className="rtv-firstload-text">
                            <strong>The realtime session ended.</strong>
                            <span>{sessionError || 'The warm job stopped (a crash, timeout, or it was reclaimed).'} Start a fresh one to keep going.</span>
                        </div>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <button className="rtv-restart-btn" onClick={restartSession}><FiRefreshCw size={13} /> Restart session</button>
                            <button className="rtv-restart-btn" onClick={() => navigate('/guardrails')}><FiShield size={13} /> Back to Rule Sets</button>
                            <button className="rtv-restart-btn" onClick={() => navigate('/workspace')}><FiHome size={13} /> Go to Hub</button>
                        </div>
                    </div>
                )}

                {/* Settings (live only) */}
                {mode === 'live' && showSettings && (
                    <div className="rtv-settings">
                        <div>
                            <label>System Prompt</label>
                            <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} rows={2} />
                        </div>
                        <div>
                            <label>Max Tokens</label>
                            <input type="number" value={maxTokens} onChange={e => setMaxTokens(Math.max(1, parseInt(e.target.value) || 1))} min={1} max={512} />
                        </div>
                    </div>
                )}

                {/* Main area */}
                <div className="rtv-main">
                    <div className="rtv-chat-col">
                        {mode === 'live' ? (
                            <>
                                <div className="rtv-messages">
                                    {messages.length === 0 && (
                                        <div className="rtv-empty">
                                            <FiRadio size={32} />
                                            <span>Send a message to see per-token CE activations in real time.</span>
                                        </div>
                                    )}
                                    {messages.map((msg, i) => {
                                        const isUser = msg.role === 'user';
                                        const analysis = !isUser ? analyses[Math.floor(i / 2)] : null;
                                        let extraStyle = {};
                                        let triggered = false;
                                        if (!isUser && analysis?.triggered_ces?.length > 0) {
                                            const ceNames = analysis.labels ? Object.keys(analysis.labels) : [];
                                            const c = ceColor(ceNames.indexOf(analysis.triggered_ces[0]), ceNames.length);
                                            extraStyle = { borderLeftColor: c, background: `${c}08` };
                                            triggered = true;
                                        }
                                        return (
                                            <div key={i} className={`rtv-msg ${isUser ? 'user' : 'assistant'} ${triggered ? 'triggered' : ''}`} style={extraStyle}>
                                                <div className="rtv-msg-role">{msg.role}</div>
                                                {!isUser && analysis ? (
                                                    <>
                                                        <TokenText analysis={analysis} selectedCEs={selectedCEs} />
                                                        <ActivationChart analysis={analysis} selectedCEs={selectedCEs} />
                                                        {analysis.rule_triggers?.length > 0 && <RuleTriggersStrip rules={analysis.rule_triggers} />}
                                                    </>
                                                ) : (
                                                    <div className="rtv-msg-text">{msg.content}</div>
                                                )}
                                            </div>
                                        );
                                    })}
                                    {loading && (
                                        <div className="rtv-msg assistant">
                                            <div className="rtv-msg-role">assistant</div>
                                            <div className="rtv-loading-dots"><span /><span /><span /></div>
                                        </div>
                                    )}
                                    <div ref={chatEndRef} />
                                </div>
                                <div className="rtv-input-bar">
                                    <input
                                        ref={inputRef} className="rtv-input" value={input}
                                        onChange={e => setInput(e.target.value)}
                                        onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
                                        placeholder={sessionReady ? 'Type a message...' : 'Waiting for the realtime session…'}
                                        disabled={loading || !sessionReady} autoFocus
                                    />
                                    <button className="rtv-send-btn" onClick={handleSend} disabled={loading || !input.trim() || !sessionReady}>
                                        <FiSend size={16} />
                                    </button>
                                </div>
                            </>
                        ) : (
                            <StoredMode
                                groups={sampleGroups}
                                activeGroupKey={activeGroupKey}
                                onPickGroup={pickGroup}
                                samples={groupSamples}
                                activeSampleIdx={activeSampleIdx}
                                onPickSample={pickSample}
                                analysis={storedAnalysis}
                                loading={storedLoading}
                                selectedCEs={selectedCEs}
                            />
                        )}
                    </div>

                    {/* CE Sidebar — shared by both modes */}
                    <div className="rtv-sidebar">
                        <div className="rtv-sidebar-section">
                            <h4 className="rtv-sidebar-title">
                                Cognitive Elements
                                <span style={{ fontSize: 9, fontWeight: 600, color: '#94a3b8', marginLeft: 6, letterSpacing: '0.03em' }}>
                                    (value = threshold)
                                </span>
                            </h4>
                            {allCEs.length === 0 ? (
                                <p className="rtv-no-ces">No analysis yet.</p>
                            ) : allCEs.map((ce, idx) => (
                                <div
                                    key={ce}
                                    className={`rtv-ce-item ${selectedCEs.has(ce) ? 'active' : ''}`}
                                    onClick={() => setSelectedCEs(prev => {
                                        const next = new Set(prev);
                                        if (next.has(ce)) next.delete(ce); else next.add(ce);
                                        return next;
                                    })}
                                    style={{ color: ceColor(idx, allCEs.length), background: selectedCEs.has(ce) ? withAlpha(ceColor(idx, allCEs.length), 0.10) : undefined }}
                                >
                                    <span className="rtv-ce-dot" style={{ background: ceColor(idx, allCEs.length) }} />
                                    <span className="rtv-ce-name" style={{ color: '#e2e8f0' }}>{ce}</span>
                                    {triggeredCEs.has(ce) && <span className="rtv-triggered-badge">TRIGGERED</span>}
                                    <span title="Calibrated threshold" style={{ marginLeft: 'auto', paddingLeft: 8, fontSize: 11, color: '#94a3b8', fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
                                        {(latestAnalysis?.thresholds_used?.[ce]?.threshold ?? 0.5).toFixed(2)}
                                    </span>
                                </div>
                            ))}
                        </div>

                        {latestRuleTriggers.length > 0 && (
                            <div className="rtv-sidebar-section">
                                <h4 className="rtv-sidebar-title">Rule Triggers</h4>
                                {latestRuleTriggers.map(rt => (
                                    <div key={rt.rule_name} style={{
                                        padding: '8px 10px', marginBottom: 6, borderRadius: 8,
                                        background: rt.fired ? 'rgba(34, 197, 94, 0.1)' : 'rgba(148, 163, 184, 0.05)',
                                        border: `1px solid ${rt.fired ? 'rgba(34, 197, 94, 0.35)' : 'rgba(148, 163, 184, 0.18)'}`,
                                    }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <span style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 12 }}>{rt.rule_name}</span>
                                            <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', color: rt.fired ? '#4ade80' : '#94a3b8' }}>
                                                {rt.fired ? 'FIRED' : 'NOT FIRED'}
                                            </span>
                                        </div>
                                        {rt.condition && (
                                            <div style={{ fontSize: 10, color: '#64748b', marginTop: 4, fontFamily: 'monospace' }} title="Firing condition over the rule's CE groups">
                                                {rt.condition}
                                            </div>
                                        )}
                                        {/* Per-group progress: hits vs. what the condition
                                          * demands (e.g. "2/3 of solicited_action"). Every
                                          * group is part of the condition; a legacy rule
                                          * with an unreferenced one is skipped here. */}
                                        {Object.entries(rt.groups || {}).filter(([, info]) => info?.quantifier != null).length > 0 && (
                                            <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4, display: 'flex', flexDirection: 'column', gap: 2 }}>
                                                {Object.entries(rt.groups || {})
                                                    .filter(([, info]) => info?.quantifier != null)
                                                    .map(([gname, info]) => (
                                                        <span key={gname} style={{ color: info.satisfied ? '#4ade80' : '#94a3b8' }}>
                                                            {info.satisfied ? '✓' : '○'} {groupProgressLabel(gname, info)}
                                                        </span>
                                                    ))}
                                            </div>
                                        )}
                                        {!rt.fired && (rt.groups_unmet?.length || 0) > 0 && (
                                            <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
                                                {rt.groups_unmet.length} group{rt.groups_unmet.length === 1 ? '' : 's'} unmet
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </Layout>
    );
}


/* ---- Stored (test-samples) mode ----
 * Browse stored conversations the way the reference Test Sample Navigation
 * does: pick a dataset group (a rule's positive/negative/calibration set, or a
 * CE's calibration dialogues), then a conversation, then it's classified. */
function StoredMode({ groups, activeGroupKey, onPickGroup, samples, activeSampleIdx, onPickSample, analysis, loading, selectedCEs }) {
    return (
        <div className="rtv-stored">
            <div className="rtv-stored-pickers">
                <div className="rtv-stored-col">
                    <div className="rtv-stored-label">Dataset</div>
                    <div className="rtv-stored-list">
                        {groups === null ? <div className="rtv-stored-hint">Loading…</div>
                            : groups.length === 0 ? <div className="rtv-stored-hint">No stored conversations for this rule set yet.</div>
                            : groups.map(g => (
                                <button key={g.key}
                                    className={`rtv-stored-item ${activeGroupKey === g.key ? 'active' : ''}`}
                                    onClick={() => onPickGroup(g)} disabled={g.count === 0}>
                                    <span className="rtv-stored-sample-text">{g.label}</span>
                                    <span className="rtv-stored-count">{g.count}</span>
                                </button>
                            ))}
                    </div>
                </div>
                <div className="rtv-stored-col">
                    <div className="rtv-stored-label">Conversation</div>
                    <div className="rtv-stored-list">
                        {!activeGroupKey ? <div className="rtv-stored-hint">Pick a dataset first.</div>
                            : samples.length === 0 ? <div className="rtv-stored-hint">No conversations.</div>
                            : samples.map(s => (
                                <button key={s.index}
                                    className={`rtv-stored-item ${activeSampleIdx === s.index ? 'active' : ''}`}
                                    onClick={() => onPickSample(s)}>
                                    <span className="rtv-stored-sample-text">
                                        <strong>Conversation {s.index + 1}</strong>
                                        {(s.first_preview || s.user_preview) && (
                                            <span style={{ opacity: 0.6 }}> · {(s.first_preview || s.user_preview).slice(0, 80)}</span>
                                        )}
                                    </span>
                                </button>
                            ))}
                    </div>
                </div>
            </div>

            <div className="rtv-stored-analysis">
                {loading ? (
                    <div className="rtv-msg assistant"><div className="rtv-loading-dots"><span /><span /><span /></div></div>
                ) : analysis?._error ? (
                    <div className="rtv-msg-text" style={{ color: '#f87171' }}>Error: {analysis._error}</div>
                ) : analysis?.turns?.length ? (
                    // Real ping-pong: render every turn in order; each assistant
                    // reply gets its own per-token CE graph + rule strip.
                    <>
                        {analysis.turns.map((turn, ti) => {
                            const hasAnalysis = turn.role === 'assistant'
                                && ((turn.tokens && turn.tokens.length) || (turn.windows && turn.windows.length));
                            if (hasAnalysis) {
                                const ta = { ...turn, labels: analysis.labels };
                                return (
                                    <div key={ti} className="rtv-msg assistant">
                                        <div className="rtv-msg-role">assistant</div>
                                        <TokenText analysis={ta} selectedCEs={selectedCEs} />
                                        <ActivationChart analysis={ta} selectedCEs={selectedCEs} />
                                        {turn.rule_triggers?.length > 0 && <RuleTriggersStrip rules={turn.rule_triggers} />}
                                    </div>
                                );
                            }
                            return (
                                <div key={ti} className={`rtv-msg ${turn.role === 'assistant' ? 'assistant' : 'user'}`}>
                                    <div className="rtv-msg-role">{turn.role}</div>
                                    <div className="rtv-msg-text" style={{ whiteSpace: 'pre-wrap' }}>{turn.content}</div>
                                </div>
                            );
                        })}
                    </>
                ) : analysis ? (
                    // Fallback for the legacy single-analysis response shape.
                    <>
                        {storedUserText(analysis._sample) && (
                            <div className="rtv-msg user"><div className="rtv-msg-role">user</div>
                                <div className="rtv-msg-text" style={{ whiteSpace: 'pre-wrap' }}>{storedUserText(analysis._sample)}</div></div>
                        )}
                        <div className="rtv-msg assistant">
                            <div className="rtv-msg-role">assistant</div>
                            <TokenText analysis={analysis} selectedCEs={selectedCEs} />
                            <ActivationChart analysis={analysis} selectedCEs={selectedCEs} />
                            {analysis.rule_triggers?.length > 0 && <RuleTriggersStrip rules={analysis.rule_triggers} />}
                        </div>
                    </>
                ) : (
                    <div className="rtv-empty"><FiDatabase size={28} /><span>Pick a CE and one of its dialogues to analyze.</span></div>
                )}
            </div>
        </div>
    );
}


/* ---- Per-token colored text (reference-parity render_colored_tokens) ----
 * Each token is tinted by the strongest CE above its threshold on that token.
 * Selecting CEs in the sidebar narrows the highlight to those CEs. Falls back
 * to the per-window display if the backend didn't return per-token data. */
function TokenText({ analysis, selectedCEs }) {
    const tokens = analysis?.tokens || [];
    if (!tokens.length) return <WindowDisplay analysis={analysis} selectedCEs={selectedCEs} />;
    const ceNames = analysis.labels ? Object.keys(analysis.labels) : [];
    // Flowing text with per-token CE highlights: each token's leading space (when
    // it starts a new word) is rendered OUTSIDE the colour, so words read normally
    // ("provocative") while sub-word continuations join their highlight tightly.
    return (
        <div className="rtv-msg-text" style={{ lineHeight: 2.1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {tokens.map((t, i) => {
                const raw = t.token ?? '';
                const lead = raw.startsWith(' ') ? ' ' : '';
                const word = lead ? raw.slice(1) : raw;
                const triggered = t.triggered_ces || [];
                const pool = selectedCEs?.size ? triggered.filter(c => selectedCEs.has(c)) : triggered;
                let ce = null, maxP = 0;
                pool.forEach(c => { const p = t.probabilities?.[c] || 0; if (p > maxP) { maxP = p; ce = c; } });
                const color = ce ? ceColor(ceNames.indexOf(ce), ceNames.length) : null;
                const tip = [`token ${t.token_index}`, ...ceNames.map(c => `${c}: ${(t.probabilities?.[c] || 0).toFixed(3)}`)].join('\n');
                return (
                    <span key={i}>
                        {lead}
                        {word !== '' && (
                            <span title={tip} style={{
                                display: 'inline-block',
                                background: color ? withAlpha(color, 0.19) : undefined,
                                boxShadow: color ? `inset 0 -2px 0 ${color}` : undefined,
                                borderRadius: 3,
                                // breathing room so each token reads as its own box
                                padding: '1px 4px',
                                marginRight: 4,
                            }}>{word}</span>
                        )}
                    </span>
                );
            })}
        </div>
    );
}


// Probability formatter: real decimals for meaningful values, SI prefixes
// (µ = millionths, n = billionths) for the tiny ones, matching the reference.
function fmtProb(p) {
    if (!p || p <= 0) return '0';
    if (p >= 0.001) return p.toFixed(3);
    if (p >= 5e-7) return (p * 1e6).toFixed(1) + 'µ';
    return (p * 1e9).toFixed(0) + 'n';
}

/* ---- Activation line chart (reference-parity create_probability_plot) ----
 * Hand-rolled SVG (no chart dependency): one polyline per CE plotting its
 * probability across token index. Hovering shows every CE's value at that token
 * (like Plotly's "x unified"); the legend toggles CEs; thresholds show per CE. */
function ActivationChart({ analysis, selectedCEs }) {
    const tokens = analysis?.tokens || [];
    const ceNames = analysis?.labels ? Object.keys(analysis.labels) : [];
    const svgRef = useRef(null);
    const outerRef = useRef(null);
    const [hoverIdx, setHoverIdx] = useState(null);
    const [hoverPx, setHoverPx] = useState(0);   // cursor x within the visible chart
    const [hidden, setHidden] = useState(() => new Set());
    if (tokens.length < 2 || ceNames.length === 0) return null;

    const H = 320, padL = 36, padR = 14, padT = 16, padB = 26;
    const N = tokens.length;
    // Give every token real horizontal room so it's clear which token drives a
    // CE. The chart scrolls sideways when there are more tokens than fit.
    const perToken = 16;
    const W = Math.max(720, padL + padR + N * perToken);
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const x = i => padL + (N === 1 ? 0 : (i / (N - 1)) * plotW);
    const y = p => padT + (1 - Math.max(0, Math.min(1, p))) * plotH;
    const grid = [0, 0.25, 0.5, 0.75, 1];
    // Plot the lines the user is focused on: respect the sidebar selection AND
    // the per-legend toggles. Thresholds show ONLY for a clicked CE.
    const shownCEs = ceNames.filter(ce => (!selectedCEs?.size || selectedCEs.has(ce)) && !hidden.has(ce));

    const onMove = (e) => {
        const svg = svgRef.current; if (!svg) return;
        const rect = svg.getBoundingClientRect();
        const px = (e.clientX - rect.left) * (W / rect.width);
        const i = Math.round(((px - padL) / plotW) * (N - 1));
        setHoverIdx(Math.max(0, Math.min(N - 1, i)));
        const outer = outerRef.current;
        if (outer) setHoverPx(e.clientX - outer.getBoundingClientRect().left);
    };

    // Unified hover: every visible CE's probability at the hovered token,
    // strongest first (mirrors Plotly's "x unified" tooltip in the reference).
    const hoverList = hoverIdx == null ? [] :
        (selectedCEs?.size ? [...selectedCEs] : ceNames.filter(ce => !hidden.has(ce)))
            .map(ce => ({ ce, p: tokens[hoverIdx].probabilities?.[ce] || 0 }))
            .sort((a, b) => b.p - a.p);
    const hoverShown = hoverList.slice(0, 12);
    const hoverRest = hoverList.length - hoverShown.length;
    const toggle = (ce) => setHidden(prev => { const n = new Set(prev); n.has(ce) ? n.delete(ce) : n.add(ce); return n; });

    return (
        <div ref={outerRef} style={{ marginTop: 10, width: '100%', maxWidth: '100%', boxSizing: 'border-box', position: 'relative', background: 'rgba(2,6,23,0.55)', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 10, padding: '10px 12px 8px' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#cbd5e1', marginBottom: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                <span>CE activation over tokens {selectedCEs?.size ? `· ${[...selectedCEs].join(' · ')}` : ''}</span>
                <span style={{ fontSize: 10, color: '#64748b', fontWeight: 500 }}>{N} tokens · hover for values{selectedCEs?.size ? ' · dashed = threshold' : ' · click legend to toggle'}</span>
            </div>
            <div style={{ overflowX: 'auto', overflowY: 'hidden', width: '100%' }}>
            <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onMouseMove={onMove} onMouseLeave={() => setHoverIdx(null)} style={{ display: 'block', width: '100%', minWidth: W, height: H, cursor: 'crosshair' }}>
                <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth={1} />
                {/* faint separator per token, so each token reads as its own slot */}
                {tokens.map((t, i) => (
                    <line key={`tk-${i}`} x1={x(i)} x2={x(i)} y1={padT} y2={padT + plotH} stroke="rgba(148,163,184,0.05)" strokeWidth={1} />
                ))}
                {grid.map(g => (
                    <g key={g}>
                        <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke="rgba(148,163,184,0.12)" strokeWidth={1} />
                        <text x={padL - 6} y={y(g) + 3} fontSize={9} fill="#94a3b8" textAnchor="end">{g}</text>
                    </g>
                ))}
                {selectedCEs?.size > 0 && shownCEs.map(ce => {
                    const thr = analysis.thresholds_used?.[ce]?.threshold ?? 0.5;
                    const c = ceColor(ceNames.indexOf(ce), ceNames.length);
                    return <line key={`thr-${ce}`} x1={padL} x2={W - padR} y1={y(thr)} y2={y(thr)} stroke={c} strokeOpacity={0.6} strokeDasharray="5 4" strokeWidth={1.2} />;
                })}
                {shownCEs.map(ce => {
                    const c = ceColor(ceNames.indexOf(ce), ceNames.length);
                    const pts = tokens.map((t, i) => `${x(i).toFixed(1)},${y(t.probabilities?.[ce] || 0).toFixed(1)}`).join(' ');
                    return <polyline key={ce} points={pts} fill="none" stroke={c} strokeWidth={selectedCEs?.size ? 2.4 : 1.8} opacity={0.9} strokeLinejoin="round" strokeLinecap="round" />;
                })}
                {selectedCEs?.size > 0 && shownCEs.map(ce => tokens.map((t, i) => (
                    <circle key={`dot-${ce}-${i}`} cx={x(i)} cy={y(t.probabilities?.[ce] || 0)} r={2.4}
                        fill={ceColor(ceNames.indexOf(ce), ceNames.length)} />
                )))}
                {/* hover crosshair + a marker on each visible line at that token */}
                {hoverIdx != null && (
                    <g>
                        <line x1={x(hoverIdx)} x2={x(hoverIdx)} y1={padT} y2={padT + plotH} stroke="rgba(226,232,240,0.45)" strokeWidth={1} />
                        {shownCEs.map(ce => (
                            <circle key={`hv-${ce}`} cx={x(hoverIdx)} cy={y(tokens[hoverIdx].probabilities?.[ce] || 0)} r={3}
                                fill={ceColor(ceNames.indexOf(ce), ceNames.length)} stroke="#0b1020" strokeWidth={1} />
                        ))}
                    </g>
                )}
                <text x={padL + plotW / 2} y={H - 3} fontSize={9} fill="#64748b" textAnchor="middle">token index &#8594;</text>
            </svg>
            </div>
            {/* unified hover tooltip: every CE's probability at the hovered token */}
            {hoverIdx != null && hoverShown.length > 0 && (() => {
                // Follow the cursor on the side away from it, clamped to the visible
                // width — so the tooltip hugs the hovered token instead of blanketing
                // a whole corner of the plot (and the lines there).
                const cw = outerRef.current?.clientWidth || 0;
                const tipW = 300;
                const side = hoverPx > cw / 2 ? 'left' : 'right';
                let left = side === 'right' ? hoverPx + 18 : hoverPx - tipW - 18;
                left = Math.max(8, Math.min(left, Math.max(8, cw - tipW - 8)));
                return (
                <div style={{ position: 'absolute', top: 34, left, width: tipW, maxWidth: 'calc(100% - 16px)', background: 'rgba(2,6,23,0.96)', border: '1px solid rgba(148,163,184,0.28)', borderRadius: 10, padding: '11px 14px', fontSize: 14, color: '#e2e8f0', pointerEvents: 'none', boxShadow: '0 10px 28px -8px rgba(0,0,0,0.65)', zIndex: 5 }}>
                    <div style={{ marginBottom: 7, display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ color: '#94a3b8', fontSize: 11, fontWeight: 600, letterSpacing: '0.03em', textTransform: 'uppercase' }}>token {tokens[hoverIdx].token_index}</span>
                        <span style={{ color: '#f1f5f9', fontWeight: 700, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
                            {(() => { const t = (tokens[hoverIdx].token ?? '').replace(/\s+/g, ' ').trim(); return t === '' ? '(space)' : `"${t}"`; })()}
                        </span>
                    </div>
                    {hoverShown.map(({ ce, p }) => (
                        <div key={ce} style={{ display: 'flex', alignItems: 'center', gap: 9, lineHeight: 1.75 }}>
                            <span style={{ width: 11, height: 11, borderRadius: 3, background: ceColor(ceNames.indexOf(ce), ceNames.length), flexShrink: 0 }} />
                            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ce}</span>
                            <span style={{ color: p >= 0.001 ? '#f1f5f9' : '#64748b', fontWeight: p >= 0.001 ? 700 : 400, fontVariantNumeric: 'tabular-nums', marginLeft: 6 }}>{fmtProb(p)}</span>
                        </div>
                    ))}
                    {hoverRest > 0 && <div style={{ color: '#64748b', marginTop: 4, fontSize: 12 }}>+{hoverRest} more ≈0</div>}
                </div>
                );
            })()}
            {!selectedCEs?.size && ceNames.length > 1 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 8 }}>
                    {ceNames.map(ce => {
                        const off = hidden.has(ce);
                        return (
                            <span key={ce} onClick={() => toggle(ce)} title={off ? 'Show this CE' : 'Hide this CE'} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 10, color: off ? '#64748b' : '#cbd5e1', cursor: 'pointer', opacity: off ? 0.55 : 1, textDecoration: off ? 'line-through' : 'none', userSelect: 'none' }}>
                                <span style={{ width: 12, height: 3, borderRadius: 2, background: ceColor(ceNames.indexOf(ce), ceNames.length), flexShrink: 0 }} />
                                {ce}
                            </span>
                        );
                    })}
                </div>
            )}
        </div>
    );
}


/* ---- Window Display (fallback when no per-token data) ---- */
function WindowDisplay({ analysis, selectedCEs }) {
    if (!analysis?.windows) return <div className="rtv-msg-text">{analysis?.generated_text}</div>;
    const ceNames = analysis.labels ? Object.keys(analysis.labels) : [];
    return (
        <div className="rtv-msg-text" style={{ lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
            {analysis.windows.map((win, i) => {
                const probs = win.probabilities || {};
                const triggeredInWindow = win.window_triggered_ces || [];
                const candidates = selectedCEs?.size
                    ? triggeredInWindow.filter(ce => selectedCEs.has(ce))
                    : triggeredInWindow;
                let highlightCE = null, maxP = 0;
                candidates.forEach(ce => { const p = probs[ce] || 0; if (p > maxP) { maxP = p; highlightCE = ce; } });
                const color = highlightCE ? ceColor(ceNames.indexOf(highlightCE), ceNames.length) : undefined;
                const tooltip = [`window ${win.window_index} · ${win.token_count} tokens`, ...ceNames.map(ce => `${ce}: ${(probs[ce] || 0).toFixed(3)}`)].join('\n');
                return (
                    <span key={i} title={tooltip} style={{
                        padding: '2px 1px', borderRadius: 4,
                        background: color ? withAlpha(color, 0.15) : 'transparent',
                        borderBottom: color ? `2px solid ${color}` : undefined,
                    }}>{win.text}</span>
                );
            })}
        </div>
    );
}


/* ---- Rule Triggers Strip (per-message badges) ---- */
function RuleTriggersStrip({ rules }) {
    if (!rules || rules.length === 0) return null;
    return (
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6, paddingTop: 8, borderTop: '1px solid rgba(148, 163, 184, 0.12)' }}>
            {rules.map(rt => (
                <span key={rt.rule_name}
                    title={rt.fired
                        ? `Condition satisfied${rt.condition ? ` · ${rt.condition}` : ''}`
                        : `Not fired · ${(rt.groups_unmet || [])
                            .map((g) => groupProgressLabel(g, (rt.groups || {})[g]))
                            .join(' · ') || 'no CE triggers'}`}
                    style={{
                        padding: '2px 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.03em', borderRadius: 999,
                        background: rt.fired ? 'rgba(34, 197, 94, 0.18)' : 'rgba(148, 163, 184, 0.12)',
                        color: rt.fired ? '#86efac' : '#94a3b8',
                        border: `1px solid ${rt.fired ? 'rgba(34, 197, 94, 0.4)' : 'rgba(148, 163, 184, 0.2)'}`,
                    }}>
                    {rt.fired ? '✓' : '○'} {rt.rule_name}
                </span>
            ))}
        </div>
    );
}
