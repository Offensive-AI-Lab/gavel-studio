// Step 1 — Scenario Ideation.
//
// Multi-turn chat with the LLM that ends when it emits the
// SCENARIO_FINAL: marker. The session_id is held in step.data so a
// page refresh doesn't lose the in-progress chat (worth noting: the
// session itself is in-memory backend-side, so a backend restart
// DOES drop the conversation — Phase 6 would move this to DB).
//
// On `is_final` the wizard saves description + name to step.data and
// the user can advance. Step 2A reads description from this step's data.
import { useState, useEffect, useRef } from 'react';
import { FiSend, FiCheckCircle, FiRefreshCw } from 'react-icons/fi';
import { startScenarioChat, sendScenarioChatMessage } from '../../api';
import InlineHelp from '../../components/InlineHelp/InlineHelp';
import { automatedPipeline } from '../../components/InlineHelp/instructorHelp';
import {
    getStepState, startStep, completeStep,
    card, primaryBtn, secondaryBtn, fieldStyle, successBanner, muted,
} from './wizardShared';


export default function Step1Scenario({ run, onPatchStep, onAdvance }) {
    const state = getStepState(run, '1');
    const data = state.data || {};

    const [sessionId, setSessionId] = useState(data.session_id || null);
    const [messages, setMessages] = useState(data.messages || []);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [finalized, setFinalized] = useState(state.status === 'completed');
    const [scenario, setScenario] = useState(data.description || '');
    const [name, setName] = useState(data.name || '');
    const scrollRef = useRef(null);
    const inputRef = useRef(null);

    // Bootstrap the chat if we don't already have a session.
    useEffect(() => {
        if (sessionId || finalized) return;
        let cancelled = false;
        (async () => {
            try {
                const res = await startScenarioChat();
                if (cancelled) return;
                const sid = res.data.session_id;
                const initial = res.data.message;
                setSessionId(sid);
                const initialMessages = [{ role: 'assistant', content: initial }];
                setMessages(initialMessages);
                startStep(onPatchStep, '1', { session_id: sid, messages: initialMessages });
            } catch (err) {
                console.error('Start chat failed:', err);
            }
        })();
        return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    // Auto-focus the input once a reply lands (and on first load) so the user
    // can keep typing without clicking back into the field every turn. Skipped
    // while sending and once the scenario is finalized (the wizard advances).
    useEffect(() => {
        if (!sending && sessionId && !finalized) {
            inputRef.current?.focus();
        }
    }, [sending, sessionId, finalized]);

    const send = async () => {
        const text = input.trim();
        if (!text || !sessionId || sending) return;
        setSending(true);
        const newMessages = [...messages, { role: 'user', content: text }];
        setMessages(newMessages);
        setInput('');
        try {
            const res = await sendScenarioChatMessage(sessionId, text);
            const reply = res.data.message;
            const isFinal = res.data.is_final;
            const description = res.data.scenario_description;
            const updated = [...newMessages, { role: 'assistant', content: reply }];
            setMessages(updated);

            let resolvedName = name;
            if (isFinal && description) {
                setFinalized(true);
                setScenario(description);
                // Prefer the concise name the AI proposed; fall back to a
                // content-word slug (NOT the first words of the sentence).
                if (!name) {
                    const auto = (res.data.scenario_name || '').trim()
                        || description
                            .toLowerCase()
                            .replace(/[^a-z0-9\s]/g, '')
                            .split(/\s+/)
                            .filter(Boolean)
                            .slice(0, 4)
                            .join('_');
                    resolvedName = auto;
                    setName(auto);
                }
            }
            await onPatchStep('1', {
                status: isFinal ? 'completed' : 'in_progress',
                data: {
                    session_id: sessionId,
                    messages: updated,
                    description: description || data.description,
                    name: resolvedName || data.name,
                },
            });

            // Scenario finalized → jump straight to step 2A (the AI-named
            // scenario + description are already saved above; no manual
            // "review + Save edits" gate). The edit panel still renders if the
            // user navigates back to this step.
            if (isFinal && description) {
                await onAdvance?.();
            }
        } catch (err) {
            console.error('Chat send failed:', err);
        } finally {
            setSending(false);
        }
    };

    const saveScenarioOverride = async () => {
        // The user edited the name/description manually after the LLM
        // finalized — persist their changes so step 2A reads the
        // canonical values.
        await completeStep(onPatchStep, '1', {
            session_id: sessionId,
            messages,
            description: scenario,
            name,
        });
    };

    const reset = async () => {
        setSessionId(null); setMessages([]); setInput('');
        setFinalized(false); setScenario(''); setName('');
        const res = await startScenarioChat();
        const sid = res.data.session_id;
        const initial = res.data.message;
        const initialMessages = [{ role: 'assistant', content: initial }];
        setSessionId(sid);
        setMessages(initialMessages);
        await onPatchStep('1', {
            status: 'in_progress',
            data: { session_id: sid, messages: initialMessages, description: '', name: '' },
        });
    };

    return (
        <div>
            <InlineHelp content={automatedPipeline} />
            <div style={card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <strong>Scenario Chat</strong>
                    <button onClick={reset} style={{ ...secondaryBtn, padding: '4px 10px', fontSize: 12 }}>
                        <FiRefreshCw size={12} /> Restart
                    </button>
                </div>
                <div
                    ref={scrollRef}
                    style={{
                        height: 340, overflowY: 'auto',
                        display: 'flex', flexDirection: 'column', gap: 10,
                        padding: '4px 2px',
                    }}
                >
                    {messages.map((m, i) => {
                        const isUser = m.role === 'user';
                        return (
                            <div
                                key={i}
                                style={{
                                    alignSelf: isUser ? 'flex-end' : 'flex-start',
                                    maxWidth: '85%',
                                    padding: '10px 14px',
                                    borderRadius: 16,
                                    borderBottomRightRadius: isUser ? 4 : 16,
                                    borderBottomLeftRadius: isUser ? 16 : 4,
                                    background: isUser
                                        ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
                                        : 'rgba(148, 163, 184, 0.14)',
                                    color: isUser ? '#fff' : '#e2e8f0',
                                    fontSize: 14, lineHeight: 1.5,
                                    whiteSpace: 'pre-wrap',
                                    boxShadow: isUser ? '0 4px 14px -6px rgba(99,102,241,0.6)' : 'none',
                                }}
                            >
                                {m.content}
                            </div>
                        );
                    })}
                </div>
                {!finalized && (
                    <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                        <input
                            ref={inputRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter' && !sending) send(); }}
                            placeholder="Describe the misuse you want to catch…"
                            style={{ ...fieldStyle, flex: 1 }}
                            disabled={sending || !sessionId}
                        />
                        <button onClick={send} disabled={sending || !sessionId || !input.trim()} style={primaryBtn}>
                            <FiSend /> Send
                        </button>
                    </div>
                )}
            </div>

            {finalized && (
                <div style={card}>
                    <div style={successBanner}>
                        <FiCheckCircle /> Scenario finalized. Tweak the name + description below if needed, then move on to step 2A.
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, marginTop: 12 }}>
                        <div>
                            <div style={{ ...muted, fontSize: 12, marginBottom: 4 }}>Scenario name (snake_case)</div>
                            <input
                                value={name}
                                onChange={e => setName(e.target.value)}
                                style={fieldStyle}
                                placeholder="e.g. medical_advice_without_disclaimer"
                            />
                        </div>
                        <div>
                            <div style={{ ...muted, fontSize: 12, marginBottom: 4 }}>Description</div>
                            <textarea
                                value={scenario}
                                onChange={e => setScenario(e.target.value)}
                                rows={5}
                                style={{ ...fieldStyle, resize: 'vertical' }}
                            />
                        </div>
                        <div>
                            <button onClick={saveScenarioOverride} style={primaryBtn}>Save edits</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
