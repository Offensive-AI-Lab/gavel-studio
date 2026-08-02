// Compare guardrails side by side, in two modes:
//   * same_policy  — guardrails sharing this one's policy (same rules/CEs),
//                    typically across DIFFERENT base models.
//   * same_model   — every trained guardrail on this one's base model,
//                    regardless of policy.
// Each guardrail shows its latest post-training evaluation: weighted averages
// and per-use-case metrics.
import { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FiHome, FiShield, FiFileText, FiCpu, FiLayers, FiRefreshCw } from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import { getPolicyComparison } from '../api';
import { useTutorialContent } from '../contexts/TutorialContext';

const page = { padding: '28px 32px', maxWidth: 1200, margin: '0 auto' };
const muted = { color: '#94a3b8' };
const card = { background: 'rgba(15,23,42,0.55)', border: '1px solid rgba(148,163,184,0.16)', borderRadius: 12, padding: 18, marginBottom: 18 };
const sectionTitle = { fontSize: 15, fontWeight: 700, color: '#e2e8f0', margin: '0 0 12px' };
const th = { textAlign: 'left', padding: '10px 12px', fontSize: 12, color: '#94a3b8', borderBottom: '1px solid rgba(148,163,184,0.18)', fontWeight: 600, whiteSpace: 'nowrap' };
const td = { padding: '10px 12px', fontSize: 13, color: '#e2e8f0', borderBottom: '1px solid rgba(148,163,184,0.08)' };
const winnerCell = { color: '#6ee7b7', fontWeight: 700 };
const ctaBtn = { display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 10, border: '1px solid rgba(129,140,248,0.4)', background: 'rgba(99,102,241,0.16)', color: '#c7d2fe', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' };

const PER_USECASE_METRICS = [
    { key: 'F1', lowerIsBetter: false },
    { key: 'TPR', lowerIsBetter: false },
    { key: 'FPR', lowerIsBetter: true },
    { key: 'Accuracy', lowerIsBetter: false },
];

const MODES = [
    { key: 'same_policy', label: 'Same rule set · across models' },
    { key: 'same_model', label: 'Same model · all rule sets' },
];

// Index of the "best" value in a row (or -1 if no clear winner / <2 numbers).
function bestIndex(values, lowerIsBetter) {
    const nums = values.map(v => (typeof v === 'number' && isFinite(v) ? v : null));
    const present = nums.filter(v => v !== null);
    if (present.length < 2) return -1;
    if (present.every(v => v === present[0])) return -1;
    const target = lowerIsBetter ? Math.min(...present) : Math.max(...present);
    return nums.findIndex(v => v === target);
}

function modeTag(style, text, color, bg, border) {
    return <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.03em', padding: '1px 7px', borderRadius: 999, color, background: bg, border: `1px solid ${border}`, ...style }}>{text}</span>;
}

export default function ComparePolicy() {
    const { classifierId } = useParams();
    const navigate = useNavigate();
    const [mode, setMode] = useState('same_policy');
    const [data, setData] = useState(null);
    const [ucMetric, setUcMetric] = useState('F1');
    const [reloadTick, setReloadTick] = useState(0);   // bumped by "Try again"

    const loadComparison = useCallback(() => setReloadTick(t => t + 1), []);

    useEffect(() => {
        let cancelled = false;
        getPolicyComparison(classifierId, mode)
            .then(res => { if (!cancelled) setData(res.data); })
            .catch(() => { if (!cancelled) setData({ error: true, source_classifier_id: Number(classifierId), mode, classifiers: [] }); });
        return () => { cancelled = true; };
    }, [classifierId, mode, reloadTick]);

    // Derived loading (avoids setState-in-effect): true until we have data for
    // THIS guardrail AND mode. Error path stores a matching sentinel.
    const loading = !data || String(data.source_classifier_id) !== String(classifierId) || data.mode !== mode;
    const classifiers = useMemo(() => data?.classifiers || [], [data]);
    const isSameModel = mode === 'same_model';

    const pageHelp = {
        title: 'Compare Rule Sets',
        summary: 'Compare trained rule sets side by side by their latest evaluation. Two modes: the SAME rule set across different base models, or all rule sets on the SAME model.',
        sections: [
            {
                heading: 'The two modes',
                bullets: [
                    'Same rule set · across models — rule sets built from the same rules/CEs (matched by a model-independent fingerprint), to see how one rule set does on different LLMs.',
                    'Same model · all rule sets — every trained rule set on this base model, even if they use different rules.',
                    'The greener value in each row is the better one (for FPR, lower is better).',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    const usecases = useMemo(() => {
        const seen = new Set();
        const order = [];
        classifiers.forEach(c => (c.usecase_rows || []).forEach(r => {
            if (r.Usecase && !seen.has(r.Usecase)) { seen.add(r.Usecase); order.push(r.Usecase); }
        }));
        return order;
    }, [classifiers]);

    const wavgKeys = useMemo(() => {
        const seen = new Set();
        const order = [];
        classifiers.forEach(c => Object.keys(c.weighted_averages || {}).forEach(k => {
            if (!seen.has(k)) { seen.add(k); order.push(k); }
        }));
        return order;
    }, [classifiers]);

    const ucLower = PER_USECASE_METRICS.find(m => m.key === ucMetric)?.lowerIsBetter || false;

    // Column header for one rule set — model + name, plus a rule-set tag in
    // same-model mode (where rule sets can differ).
    const colHeader = (c, short = false) => (
        <div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
                {!short && <FiCpu size={12} />}{c.model_name}
            </div>
            <div style={{ fontWeight: 500, color: '#64748b', marginTop: 2 }}>
                {c.name}{c.is_source ? ' · this one' : ''}
            </div>
            {isSameModel && (
                <div style={{ marginTop: 4 }}>
                    {c.same_policy_as_source
                        ? modeTag({}, 'same rule set', '#6ee7b7', 'rgba(16,185,129,0.15)', 'rgba(16,185,129,0.35)')
                        : modeTag({}, `${(c.rule_names || []).length} rule${(c.rule_names || []).length === 1 ? '' : 's'} · diff rule set`, '#fcd34d', 'rgba(245,158,11,0.15)', 'rgba(251,191,36,0.35)')}
                </div>
            )}
        </div>
    );

    const emptyMsg = classifiers.length === 0
        ? (isSameModel
            ? 'No trained rule sets on this model yet.'
            : 'This rule set isn\'t trained yet (or predates rule-set fingerprints), so there\'s no rule set to match on.')
        : null;

    return (
        <Layout>
            <div style={page}>
                <Breadcrumb items={[
                    { label: 'Hub', icon: FiHome, to: '/workspace' },
                    { label: 'Rule Sets', icon: FiShield, to: '/guardrails' },
                    { label: 'Rule Set', icon: FiFileText, to: `/classifiers/${classifierId}/rules` },
                    { label: 'Compare' },
                ]} />
                <h1 style={{ margin: '0 0 6px', fontSize: 22, color: '#f1f5f9' }}>Compare</h1>

                {/* Mode toggle */}
                <div style={{ display: 'inline-flex', gap: 4, padding: 4, borderRadius: 12, background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(148,163,184,0.16)', margin: '10px 0 18px' }}>
                    {MODES.map(m => {
                        const on = mode === m.key;
                        return (
                            <button key={m.key} onClick={() => setMode(m.key)}
                                style={{
                                    padding: '7px 14px', borderRadius: 9, border: 'none', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600,
                                    background: on ? 'linear-gradient(135deg,#818cf8,#3b82f6)' : 'transparent',
                                    color: on ? '#fff' : '#94a3b8',
                                }}>
                                {m.label}
                            </button>
                        );
                    })}
                </div>

                {loading && <div style={muted}>Loading…</div>}

                {!loading && data?.error && (
                    <div style={card}>
                        <div style={{ color: '#fca5a5', fontSize: 14, marginBottom: 14 }}>Couldn&apos;t load the comparison. Please try again.</div>
                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                            <button onClick={loadComparison} style={ctaBtn}><FiRefreshCw size={14} /> Try again</button>
                            <button onClick={() => navigate('/guardrails')} style={ctaBtn}><FiShield size={14} /> Back to Rule Sets</button>
                            <button onClick={() => navigate('/workspace')} style={ctaBtn}><FiHome size={14} /> Go to Hub</button>
                        </div>
                    </div>
                )}

                {!loading && !data?.error && (
                    <>
                        {/* Context header */}
                        <div style={card}>
                            {isSameModel ? (
                                <>
                                    <div style={sectionTitle}>Model: {data.source_model_name}</div>
                                    <div style={{ ...muted, fontSize: 13 }}>
                                        Every trained rule set on this base model. They may use different rules — those rows that don&apos;t apply to a rule set show &ldquo;—&rdquo;.
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div style={sectionTitle}>Rule set</div>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                        {(data.rule_names || []).length === 0
                                            ? <span style={muted}>—</span>
                                            : data.rule_names.map(n => (
                                                <span key={n} style={{ fontSize: 12, padding: '3px 10px', borderRadius: 999, background: 'rgba(99,102,241,0.16)', color: '#c7d2fe', border: '1px solid rgba(129,140,248,0.3)' }}>{n}</span>
                                            ))}
                                    </div>
                                </>
                            )}
                            <div style={{ ...muted, fontSize: 12, marginTop: 10 }}>
                                {classifiers.length} rule set{classifiers.length === 1 ? '' : 's'} {isSameModel ? 'on this model' : 'share this rule set'}.
                            </div>
                        </div>

                        {emptyMsg && (
                            <div style={card}><div style={{ ...muted, fontSize: 14 }}>{emptyMsg}</div></div>
                        )}

                        {classifiers.length === 1 && (
                            <div style={card}>
                                <div style={{ color: '#fcd34d', fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Nothing to compare yet</div>
                                <div style={{ ...muted, fontSize: 13 }}>
                                    {isSameModel
                                        ? 'Only one trained rule set on this model. Train another rule set on the same model to compare them here.'
                                        : 'Only this rule set uses these rules. Use "Apply to another model" to copy it onto a different base model, then train + evaluate it — it shows up here automatically.'}
                                </div>
                            </div>
                        )}

                        {classifiers.length >= 2 && (
                            <>
                                {/* Weighted averages */}
                                <div style={card}>
                                    <div style={sectionTitle}>Weighted averages</div>
                                    <div style={{ overflowX: 'auto' }}>
                                        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 520 }}>
                                            <thead>
                                                <tr>
                                                    <th style={th}>Metric</th>
                                                    {classifiers.map(c => <th key={c.classifier_id} style={th}>{colHeader(c)}</th>)}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {wavgKeys.length === 0 ? (
                                                    <tr><td style={td} colSpan={classifiers.length + 1}><span style={muted}>No evaluation metrics yet for these rule sets.</span></td></tr>
                                                ) : wavgKeys.map(key => {
                                                    const lower = key.toLowerCase().includes('fpr');
                                                    const vals = classifiers.map(c => {
                                                        const v = c.weighted_averages?.[key];
                                                        return typeof v === 'number' ? v : null;
                                                    });
                                                    const best = bestIndex(vals, lower);
                                                    return (
                                                        <tr key={key}>
                                                            <td style={td}>{key}{lower ? ' (lower better)' : ''}</td>
                                                            {vals.map((v, i) => (
                                                                <td key={i} style={{ ...td, ...(i === best ? winnerCell : {}) }}>
                                                                    {v == null ? '—' : `${(v * 100).toFixed(1)}%`}
                                                                </td>
                                                            ))}
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* Per use-case */}
                                {usecases.length > 0 && (
                                    <div style={card}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                                            <div style={sectionTitle}>Per use-case</div>
                                            {/* Segmented control instead of a native <select> — the OS-rendered
                                                option list can't be themed and clashes with the dark UI. */}
                                            <div style={{ display: 'inline-flex', gap: 4, padding: 4, borderRadius: 10, background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(148,163,184,0.16)' }}>
                                                {PER_USECASE_METRICS.map(m => {
                                                    const on = ucMetric === m.key;
                                                    return (
                                                        <button key={m.key} type="button" onClick={() => setUcMetric(m.key)}
                                                            style={{
                                                                padding: '5px 13px', borderRadius: 7, border: 'none', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                                                                background: on ? 'linear-gradient(135deg,#818cf8,#3b82f6)' : 'transparent',
                                                                color: on ? '#fff' : '#94a3b8',
                                                                boxShadow: on ? '0 2px 8px -2px rgba(99,102,241,0.55)' : 'none',
                                                                transition: 'background 0.15s, color 0.15s',
                                                            }}>
                                                            {m.key}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                        <div style={{ overflowX: 'auto' }}>
                                            <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 520 }}>
                                                <thead>
                                                    <tr>
                                                        <th style={th}>Use case ({ucMetric}{ucLower ? ', lower better' : ''})</th>
                                                        {classifiers.map(c => <th key={c.classifier_id} style={th}>{colHeader(c, true)}</th>)}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {usecases.map(uc => {
                                                        const vals = classifiers.map(c => {
                                                            const row = (c.usecase_rows || []).find(r => r.Usecase === uc);
                                                            const v = row?.[ucMetric];
                                                            return typeof v === 'number' ? v : null;
                                                        });
                                                        const best = bestIndex(vals, ucLower);
                                                        return (
                                                            <tr key={uc}>
                                                                <td style={td}>{uc}</td>
                                                                {vals.map((v, i) => (
                                                                    <td key={i} style={{ ...td, ...(i === best ? winnerCell : {}) }}>
                                                                        {v == null ? '—' : v.toFixed(3)}
                                                                    </td>
                                                                ))}
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}

                                {classifiers.some(c => !c.has_eval) && (
                                    <div style={{ ...muted, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <FiLayers size={12} /> Rule sets shown with &ldquo;—&rdquo; haven&apos;t been evaluated since their last training.
                                    </div>
                                )}
                            </>
                        )}
                    </>
                )}
            </div>
        </Layout>
    );
}
