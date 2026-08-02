import { useState, useEffect } from 'react';
import { FiArrowRight, FiLoader, FiAlertTriangle } from 'react-icons/fi';
import ReactiveButton from '../ReactiveButton/ReactiveButton';
import { deriveScenario, generateRuleDefaults, getRuleDefaultsStatus } from '../../api';
import { useTaskTray } from '../../contexts/TaskTrayContext';
import { runInTray, sleep } from '../../hooks/runInTray';

// Final step for a freshly-created rule: derive a misuse scenario from the
// rule's CEs/roles, let the user review/edit it, then kick off generation of
// the rule's Test Set + calibration set.
//
// The generation (100 positive + 100 negative + 50 calibration dialogues)
// takes a while, so it runs in the BACKGROUND via the task tray. Clicking
// Generate fires the job and immediately calls `onDone` (so the page can
// navigate away without making the user wait), but the rule itself stays HIDDEN
// — `finalize` (which flips is_ready=TRUE) is invoked by the job ONLY after the
// sets finish, so the rule never appears in Drafts/Browse half-built.

export default function RuleDefaultsStep({ ruleId, onDone, finalize }) {
    const tray = useTaskTray();
    const [scenario, setScenario] = useState('');
    const [deriving, setDeriving] = useState(true);
    const [deriveError, setDeriveError] = useState('');
    const [error, setError] = useState('');

    // Prefill the scenario by deriving one from the rule. Failure is soft —
    // the user can still type their own scenario into the empty box.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await deriveScenario(ruleId);
                if (!cancelled) setScenario(res.data?.scenario || '');
            } catch {
                if (!cancelled) setDeriveError('Could not auto-write a scenario — describe the misuse this rule should catch below.');
            } finally {
                if (!cancelled) setDeriving(false);
            }
        })();
        return () => { cancelled = true; };
    }, [ruleId]);

    const handleGenerate = () => {
        if (!scenario.trim()) { setError('Write a scenario first.'); return; }
        setError('');
        const instructions = scenario.trim();

        // Kick the generation into the background tray task. The user can leave;
        // the chip (top-right) tracks the three buckets and reports when ready.
        runInTray(tray, {
            kind: 'rule',
            title: 'Generating test & calibration set',
            runningSubtitle: 'Positive · negative · calibration dialogues…',
            successSubtitle: 'Rule ready — find it in Drafts.',
            job: async (update) => {
                await generateRuleDefaults(ruleId, instructions);
                // Poll until all three buckets are ready.
                for (;;) {
                    await sleep(2500);
                    const res = await getRuleDefaultsStatus(ruleId);
                    const st = res.data?.state;
                    if (st === 'ready') break;
                    if (st === 'error') throw new Error('Generation failed for one or more sets.');
                    const ready = (res.data?.datasets || []).filter(d => d.status === 'ready').length;
                    update({ subtitle: `${ready}/3 sets ready…` });
                }
                // Sets are ready → NOW reveal the rule (embed + is_ready=TRUE).
                // It stayed hidden everywhere until this point, so it never shows
                // half-built. Only after this does it appear in Drafts/Browse.
                if (finalize) {
                    update({ subtitle: 'Finalizing…' });
                    await finalize();
                }
            },
        });

        // Don't block — let the wizard/page move on immediately.
        onDone();
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={infoBox}>
                <p style={{ margin: 0, fontWeight: 700, color: '#f1f5f9' }}>Test Set &amp; calibration</p>
                <p style={{ margin: '6px 0 0', fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.55 }}>
                    Review the scenario below — the model generates the rule's test set
                    (100 positive + 100 negative) and calibration set (50) from it. This
                    runs <strong>in the background</strong>: you can leave this page once you
                    start it, and the task tray (top-right) shows when it's ready. The rule
                    can't be published until these finish.
                </p>
                <p style={{ margin: '8px 0 0', fontSize: '0.82rem', color: '#94a3b8', lineHeight: 1.5 }}>
                    <strong>Why no training set?</strong> The rule set is trained on each
                    Cognitive Element's own training data, not on rules. A rule is just a
                    boolean combination of those CEs, so it only needs a <strong>test set</strong>
                    (to measure how well it fires) and a <strong>calibration set</strong> (to tune
                    each CE's threshold) — there's nothing rule-specific to train.
                </p>
            </div>

            <label style={labelStyle}>Misuse scenario</label>
            {deriving ? (
                <div style={mutedRow}>
                    <FiLoader size={14} style={{ animation: 'spin 1s linear infinite' }} /> Analyzing the rule…
                </div>
            ) : (
                <textarea
                    value={scenario}
                    onChange={(e) => setScenario(e.target.value)}
                    rows={5}
                    placeholder="Describe the misuse this rule should catch…"
                    style={{
                        padding: '12px 14px', borderRadius: 12,
                        border: '2px solid rgba(148, 163, 184, 0.22)',
                        background: 'rgba(2, 6, 23, 0.55)', color: '#f1f5f9',
                        fontSize: '0.9rem', fontFamily: 'inherit', lineHeight: 1.5,
                        outline: 'none', resize: 'vertical',
                    }}
                />
            )}
            {deriveError && <div style={warnStyle}><FiAlertTriangle size={13} /> {deriveError}</div>}
            {error && <div style={errStyle}>{error}</div>}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: '0.78rem', color: '#64748b' }}>
                    Generation runs in the background — you don't have to wait here.
                </span>
                <ReactiveButton
                    label="Generate test & calibration set"
                    Icon={FiArrowRight}
                    onClick={handleGenerate}
                    disabled={deriving || !scenario.trim()}
                />
            </div>

            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}

const infoBox = {
    background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.14), rgba(99, 102, 241, 0.14))',
    border: '1px solid rgba(148, 163, 184, 0.18)', borderRadius: 14, padding: 16,
};
const labelStyle = { fontSize: '0.82rem', fontWeight: 600, color: '#94a3b8' };
const mutedRow = { display: 'flex', alignItems: 'center', gap: 8, color: '#94a3b8', fontSize: '0.88rem', padding: '10px 0' };
const warnStyle = {
    display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: '#fcd34d',
    background: 'rgba(245, 158, 11, 0.14)', border: '1px solid rgba(251, 191, 36, 0.30)',
    borderRadius: 8, padding: '8px 12px',
};
const errStyle = {
    fontSize: '0.82rem', color: '#fca5a5',
    background: 'rgba(239, 68, 68, 0.14)', border: '1px solid rgba(248, 113, 113, 0.30)',
    borderRadius: 8, padding: '8px 12px',
};
