// CE pipeline — single conversational step (mirrors the rule's scenario chat).
//
// The user describes the concept in a chat; the AI asks clarifying questions
// (chat bubbles) until it can propose a Cognitive Element. The proposal renders
// inline for review, and "Approve & Build" hands off to the wizard's onFinish
// (which generates the CE's training + calibration data in the background).
//
// generateCe(description, preferType, history) is stateless — `description` is
// the first user message; each clarification Q&A goes into `history`.
import { useState, useEffect, useRef } from 'react';
import { FiSend, FiRefreshCw, FiCheckCircle, FiAlertTriangle } from 'react-icons/fi';
import { generateCe } from '../../api';
import {
    getStepState, startStep, completeStep,
    card, primaryBtn, secondaryBtn, muted, fieldStyle, errorBanner,
} from '../RuleWizardSteps/wizardShared';

const GREETING = "Describe the Cognitive Element you want to capture — one CONTEXT (a domain/setting) or one ACTION (a behaviour). Be specific about what's in and out of scope.";

function ceCats(ce) {
    const cats = [...(ce.assigned_categories || [])];
    if (ce.new_category?.name) cats.push(ce.new_category.name);
    return cats;
}

function CEPreview({ ce }) {
    return (
        <div style={card}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 15, color: '#e2e8f0' }}>{ce.name}</span>
                {ce.type && <span style={{ padding: '2px 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: 'rgba(99,102,241,0.18)', color: '#a5b4fc' }}>{ce.type}</span>}
            </div>
            <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.5 }}>{ce.definition}</div>
            {ceCats(ce).length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
                    {ceCats(ce).map((c, i) => (
                        <span key={i} style={{ padding: '2px 8px', borderRadius: 999, fontSize: 11, background: 'rgba(148,163,184,0.16)', color: '#cbd5e1' }}>{c}</span>
                    ))}
                </div>
            )}
            {(ce.in_scope_examples || []).length > 0 && (
                <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#818cf8', fontWeight: 700, marginBottom: 4 }}>In-scope examples</div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {ce.in_scope_examples.slice(0, 8).map((ex, i) => (
                            <li key={i} style={{ fontSize: 12.5, color: '#cbd5e1', marginBottom: 3 }}>{typeof ex === 'string' ? ex : JSON.stringify(ex)}</li>
                        ))}
                    </ul>
                </div>
            )}
            {(ce.out_of_scope_notes || []).length > 0 && (
                <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#f59e0b', fontWeight: 700, marginBottom: 4 }}>Out of scope</div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {ce.out_of_scope_notes.slice(0, 6).map((ex, i) => (
                            <li key={i} style={{ fontSize: 12.5, color: '#cbd5e1', marginBottom: 3 }}>{typeof ex === 'string' ? ex : JSON.stringify(ex)}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default function Step1CEChat({ run, onPatchStep, onAdvance }) {
    const state = getStepState(run, '1');
    const data = state.data || {};

    const [messages, setMessages] = useState(data.messages || [{ role: 'assistant', content: GREETING }]);
    const [input, setInput] = useState('');
    const [concept, setConcept] = useState(data.concept || '');
    const [history, setHistory] = useState(data.history || []);
    const [pendingQuestion, setPendingQuestion] = useState(null);
    const [ceData, setCeData] = useState(data.ce_data || null);
    const [sending, setSending] = useState(false);
    const [approving, setApproving] = useState(false);
    const [error, setError] = useState(null);
    const scrollRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    // Auto-focus the input on load and after each reply lands so the user can
    // keep typing without clicking back into the field every turn. The input
    // only exists until a CE is proposed (ceData), so skip once that happens.
    useEffect(() => {
        if (!sending && !ceData) inputRef.current?.focus();
    }, [sending, ceData]);

    // Call the (stateless) CE generator with the running concept + clarify
    // history. Renders the model's reply as a chat bubble.
    const generate = async (theConcept, hist, msgs) => {
        setSending(true); setError(null);
        try {
            const res = await generateCe(theConcept, null, hist);
            const d = res.data || {};
            if (d.needs_clarification) {
                const q = d.clarification_question || 'Could you clarify?';
                setPendingQuestion(q);
                const next = [...msgs, { role: 'assistant', content: q }];
                setMessages(next);
                await startStep(onPatchStep, '1', { messages: next, concept: theConcept, history: hist });
            } else if (d.refuse) {
                const r = d.refuse_reason || 'This concept is already covered by an existing Cognitive Element.';
                setPendingQuestion(null);
                setMessages([...msgs, { role: 'assistant', content: r }]);
            } else if (d.ce_data) {
                setCeData(d.ce_data);
                setPendingQuestion(null);
                const next = [...msgs, { role: 'assistant', content: `Proposed: “${d.ce_data.name}”. Review it below and Approve & Build.` }];
                setMessages(next);
                await completeStep(onPatchStep, '1', {
                    ce_data: d.ce_data, concept: theConcept, history: hist, messages: next,
                });
            } else {
                throw new Error(d.error || 'No cognitive element was produced.');
            }
        } catch (e) {
            setError(e?.response?.data?.detail || e.message);
        } finally {
            setSending(false);
        }
    };

    const send = async () => {
        const text = input.trim();
        if (!text || sending || ceData) return;
        const msgs = [...messages, { role: 'user', content: text }];
        setMessages(msgs);
        setInput('');
        if (!concept) {
            // First message = the concept itself.
            setConcept(text);
            await generate(text, [], msgs);
        } else {
            // Answering a clarification.
            const hist = [...history, { question: pendingQuestion, answer: text }];
            setHistory(hist);
            await generate(concept, hist, msgs);
        }
    };

    const restart = async () => {
        const greet = [{ role: 'assistant', content: GREETING }];
        setMessages(greet); setInput(''); setConcept(''); setHistory([]);
        setPendingQuestion(null); setCeData(null); setError(null);
        await onPatchStep('1', { status: 'in_progress', data: { messages: greet } });
    };

    const handleApprove = async () => {
        setApproving(true);
        try { await onAdvance?.(); }
        catch (e) { setApproving(false); setError(e?.message || 'Could not start the build.'); }
    };

    return (
        <div>
            <div style={card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <strong>Cognitive Element Chat</strong>
                    <button onClick={restart} style={{ ...secondaryBtn, padding: '4px 10px', fontSize: 12 }}>
                        <FiRefreshCw size={12} /> Restart
                    </button>
                </div>
                <div
                    ref={scrollRef}
                    style={{ height: 320, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, padding: '4px 2px' }}
                >
                    {messages.map((m, i) => {
                        const isUser = m.role === 'user';
                        return (
                            <div key={i} style={{
                                alignSelf: isUser ? 'flex-end' : 'flex-start',
                                maxWidth: '85%', padding: '10px 14px', borderRadius: 16,
                                borderBottomRightRadius: isUser ? 4 : 16,
                                borderBottomLeftRadius: isUser ? 16 : 4,
                                background: isUser ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : 'rgba(148, 163, 184, 0.14)',
                                color: isUser ? '#fff' : '#e2e8f0', fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap',
                            }}>
                                {m.content}
                            </div>
                        );
                    })}
                    {sending && (
                        <div style={{ alignSelf: 'flex-start', ...muted, fontSize: 13, padding: '4px 8px' }}>Thinking…</div>
                    )}
                </div>
                {!ceData && (
                    <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                        <input
                            ref={inputRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter' && !sending) send(); }}
                            placeholder={concept ? 'Answer the question…' : 'Describe the concept to capture…'}
                            style={{ ...fieldStyle, flex: 1 }}
                            disabled={sending}
                        />
                        <button onClick={send} disabled={sending || !input.trim()} style={primaryBtn}>
                            <FiSend /> Send
                        </button>
                    </div>
                )}
            </div>

            {error && <div style={card}><div style={errorBanner}><FiAlertTriangle /> {error}</div></div>}

            {ceData && (
                <>
                    <CEPreview ce={ceData} />
                    <div style={card}>
                        <div style={{ ...muted, fontSize: 13, marginBottom: 10 }}>
                            Approve to generate this CE's training + calibration data automatically in the background — watch the task tray (top-right).
                        </div>
                        <button onClick={handleApprove} disabled={approving} style={primaryBtn}>
                            {approving ? <><FiRefreshCw /> Starting…</> : <><FiCheckCircle /> Approve &amp; Build</>}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
