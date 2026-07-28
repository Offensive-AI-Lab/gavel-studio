// Rule page — everything about ONE rule, guardrail-independent. The user can
// read the predicate, expand every Cognitive Element to see its definition +
// examples, and view the rule's single Test Set (+ calibration) with sample
// dialogues. Each rule has exactly ONE test/calibration set — the
// auto-generated default one; there are no user-defined custom sets.
// Calibration and evaluation are NOT here (they're per-guardrail).
import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    FiHome, FiUsers, FiChevronDown, FiChevronUp, FiCpu, FiFileText, FiClock, FiBookmark, FiEdit2,
} from 'react-icons/fi';
import {
    getRuleDetail, previewRuleTestSets,
    getRuleBookmarks, addRuleBookmark, removeRuleBookmark,
    getCEBookmarks, addCEBookmark, removeCEBookmark,
} from '../api';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import BuildRuleFromCEsModal from './BuildRuleFromCEsModal';
import { roleLabel } from '../utils/roleLabels';
import { recordRecent } from '../utils/recents';
import { useTutorialContent } from '../contexts/TutorialContext';

function readUser() {
    try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
}

const TYPE_LABEL = { positive: 'Positive', negative: 'Negative', positive_calibration: 'Calibration' };

const page = { padding: '28px 32px', maxWidth: 1000, margin: '0 auto' };
const card = { background: 'rgba(15,23,42,0.55)', border: '1px solid rgba(148,163,184,0.16)', borderRadius: 12, padding: 16, marginBottom: 16 };
const sectionTitle = { display: 'flex', alignItems: 'center', gap: 8, fontSize: 15, fontWeight: 700, color: '#e2e8f0', marginBottom: 12 };
const muted = { color: '#94a3b8' };
const primaryBtn = { display: 'inline-flex', alignItems: 'center', gap: 6, background: 'linear-gradient(135deg,#818cf8,#3b82f6)', color: '#fff', border: 'none', borderRadius: 8, padding: '9px 14px', fontWeight: 700, cursor: 'pointer' };
const ghostBtn = { display: 'inline-flex', alignItems: 'center', gap: 6, background: 'rgba(148,163,184,0.12)', color: '#cbd5e1', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 8, padding: '7px 12px', fontWeight: 600, cursor: 'pointer' };
const chipS = (bg, c) => ({ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '2px 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: bg, color: c });

function Convo({ convo }) {
    return (
        <div style={{ padding: 8, borderRadius: 6, background: 'rgba(2,6,23,0.5)', border: '1px solid rgba(148,163,184,0.12)' }}>
            {(convo || []).slice(0, 4).map((m, i) => (
                <div key={i} style={{ fontSize: 11.5, color: '#cbd5e1', marginBottom: 3 }}>
                    <b style={{ color: m.role === 'assistant' ? '#60a5fa' : '#a78bfa' }}>{m.role}:</b>{' '}
                    {String(m.content || '').slice(0, 200)}{String(m.content || '').length > 200 ? '…' : ''}
                </div>
            ))}
        </div>
    );
}

function BucketChips({ buckets }) {
    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {buckets.map((b) => {
                const ready = b.status === 'ready';
                return (
                    <span key={b.dataset_type} style={chipS(ready ? 'rgba(16,185,129,0.14)' : 'rgba(148,163,184,0.14)', ready ? '#34d399' : '#94a3b8')}>
                        {!ready && <FiClock size={11} />}
                        {TYPE_LABEL[b.dataset_type] || b.dataset_type}: {ready ? b.count : (b.status === 'generating' ? '…' : b.status)}
                    </span>
                );
            })}
        </div>
    );
}

// Renders the rule's test set: scenario, bucket chips, samples.
function TestSetView({ set }) {
    const [open, setOpen] = useState(false);
    const samples = [];
    (set.buckets || []).forEach(b => (b.samples || []).slice(0, 1).forEach(c => c && samples.push({ type: b.dataset_type, convo: c })));
    return (
        <div style={{ padding: 12, borderRadius: 8, background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(148,163,184,0.16)' }}>
            <div style={{ minWidth: 0 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: set.accent }}>{set.name}</span>
                {set.scenario && <div style={{ ...muted, fontSize: 12, marginTop: 4 }}>{set.scenario.length > 280 ? set.scenario.slice(0, 280) + '…' : set.scenario}</div>}
                <BucketChips buckets={set.buckets} />
            </div>
            {samples.length > 0 && (
                <>
                    <button onClick={() => setOpen(o => !o)} style={{ marginTop: 8, background: 'none', border: 'none', color: '#94a3b8', fontSize: 12, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, padding: 0 }}>
                        {open ? <FiChevronUp /> : <FiChevronDown />} Sample dialogues
                    </button>
                    {open && (
                        <div style={{ marginTop: 6, display: 'grid', gap: 8 }}>
                            {samples.map((s, i) => (
                                <div key={i}>
                                    <div style={{ fontSize: 10, textTransform: 'uppercase', color: '#64748b', marginBottom: 3 }}>{TYPE_LABEL[s.type] || s.type}</div>
                                    <Convo convo={s.convo} />
                                </div>
                            ))}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function CeRow({ ce, bookmarked, onToggleBookmark }) {
    const [open, setOpen] = useState(false);
    const roleColor = ce.role === 'fallback' ? '#fbbf24' : (ce.role === 'sufficient' ? '#34d399' : '#60a5fa');
    // Display-only labels (necessary→Necessary, fallback→Any of, sufficient→
    // Supporting). Fallback groups are 0-indexed in the DB and shown 1-indexed.
    const roleText = roleLabel(ce.role);
    const groupSuffix = ce.role === 'fallback' ? ` · G${(ce.fallback_group || 0) + 1}` : '';
    return (
        <div style={{ borderBottom: '1px solid rgba(148,163,184,0.1)' }}>
            <button onClick={() => setOpen(o => !o)} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, background: 'none', border: 'none', cursor: 'pointer', padding: '10px 2px', color: '#e2e8f0' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{ce.name}</span>
                    <span style={chipS('rgba(148,163,184,0.14)', roleColor)}>{roleText}{groupSuffix}</span>
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                    {onToggleBookmark && ce.ce_id != null && (
                        <span
                            role="button"
                            tabIndex={0}
                            onClick={(e) => { e.stopPropagation(); onToggleBookmark(); }}
                            onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); onToggleBookmark(); } }}
                            title={bookmarked ? 'Remove CE from your Library' : 'Save CE to your Library'}
                            style={{ display: 'inline-flex', alignItems: 'center', color: bookmarked ? '#fcd34d' : '#94a3b8' }}
                        >
                            <FiBookmark size={15} fill={bookmarked ? '#fcd34d' : 'none'} />
                        </span>
                    )}
                    {open ? <FiChevronUp /> : <FiChevronDown />}
                </span>
            </button>
            {open && (
                <div style={{ padding: '0 2px 12px' }}>
                    {ce.definition && <div style={{ fontSize: 12.5, color: '#cbd5e1', lineHeight: 1.5 }}>{ce.definition}</div>}
                    {(ce.examples || []).length > 0 && (
                        <div style={{ marginTop: 8 }}>
                            <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#818cf8', fontWeight: 700, marginBottom: 4 }}>Examples</div>
                            <ul style={{ margin: 0, paddingLeft: 18 }}>
                                {ce.examples.slice(0, 6).map((ex, i) => (
                                    <li key={i} style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 3 }}>
                                        {typeof ex === 'string' ? ex : String(ex?.input || JSON.stringify(ex)).slice(0, 200)}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {!ce.definition && (ce.examples || []).length === 0 && <div style={{ ...muted, fontSize: 12 }}>No definition or examples on file.</div>}
                </div>
            )}
        </div>
    );
}

export default function RulePage() {
    const { ruleId } = useParams();
    const navigate = useNavigate();
    const rid = parseInt(ruleId, 10);
    const [detail, setDetail] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState(false);   // true when the rule fetch fails
    const [bookmarked, setBookmarked] = useState(false);
    const [bmBusy, setBmBusy] = useState(false);
    const [ceBmIds, setCeBmIds] = useState(() => new Set());   // this user's CE bookmarks
    const user = readUser();

    // Load the user's CE bookmarks so each CE row shows its saved state.
    useEffect(() => {
        let cancelled = false;
        if (!user?.user_id) return;
        Promise.resolve(getCEBookmarks?.(user.user_id))
            .then((res) => {
                if (cancelled) return;
                setCeBmIds(new Set((res?.data?.bookmarks || []).map((b) => b.ce_id).filter((x) => x != null)));
            })
            .catch(() => {});
        return () => { cancelled = true; };
    }, [user?.user_id]);

    const toggleCeBookmark = async (ceId) => {
        if (ceId == null || !user?.user_id) return;
        const wasSaved = ceBmIds.has(ceId);
        setCeBmIds((prev) => { const n = new Set(prev); wasSaved ? n.delete(ceId) : n.add(ceId); return n; });
        try {
            if (wasSaved) await removeCEBookmark?.(user.user_id, ceId);
            else await addCEBookmark?.(user.user_id, ceId);
        } catch {
            setCeBmIds((prev) => { const n = new Set(prev); wasSaved ? n.add(ceId) : n.delete(ceId); return n; });
        }
    };

    const loadPreview = useCallback(async () => {
        try { const res = await previewRuleTestSets(rid); setPreview(res.data); } catch { setPreview(null); }
    }, [rid]);

    // Bookmark state — only meaningful for published rules (have a public_id).
    useEffect(() => {
        let cancelled = false;
        if (!user?.user_id) return;
        (async () => {
            try {
                const res = await getRuleBookmarks(user.user_id);
                if (cancelled) return;
                const ids = new Set((res.data?.bookmarks || []).map(b => b.rule_id));
                setBookmarked(ids.has(rid));
            } catch { /* ignore */ }
        })();
        return () => { cancelled = true; };
    }, [rid, user?.user_id]);

    const toggleBookmark = async () => {
        if (!user?.user_id || bmBusy) return;
        setBmBusy(true);
        try {
            if (bookmarked) { await removeRuleBookmark(user.user_id, rid); setBookmarked(false); }
            else { await addRuleBookmark(user.user_id, rid); setBookmarked(true); }
        } catch { /* notify wrapper surfaces errors */ }
        finally { setBmBusy(false); }
    };

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await getRuleDetail(rid);
                if (!cancelled) {
                    setDetail(res.data);
                    setLoadError(false);
                    recordRecent('rule', { id: rid, name: res.data?.name || `Rule #${rid}`, path: `/rules/${rid}` });
                }
            } catch {
                if (!cancelled) { setDetail(null); setLoadError(true); }
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        loadPreview();
        return () => { cancelled = true; };
    }, [rid, loadPreview]);

    // Each rule has a single test + calibration set.
    const def = preview?.default || {};
    const hasDefault = (def.buckets || []).length > 0;
    const testSet = hasDefault
        ? { name: 'Test Set', accent: '#818cf8', scenario: def.scenario_instructions, buckets: def.buckets }
        : null;

    const ruleName = detail?.name || `Rule #${ruleId}`;
    const ces = detail?.ces || [];

    // "Edit" = fork this rule into a NEW draft (prefilled build-from-CEs wizard,
    // forced to a new name). Available to a signed-in user once the rule has CEs.
    const [editOpen, setEditOpen] = useState(false);
    const editableCes = ces.filter((c) => c && c.ce_id != null);
    const canEdit = !!user?.user_id && editableCes.length > 0;
    const editBase = {
        name: detail?.name,
        ces: editableCes.map((c) => ({ ce_id: c.ce_id, name: c.name, role: c.role, fallback_group: c.fallback_group, category: c.category })),
        categories: detail?.categories || [],
    };

    const pageHelp = {
        title: 'Rule Page',
        summary: "Everything about one rule, independent of any rule set — its plain-language explanation, boolean logic, the cognitive elements it combines, and its auto-generated test & calibration set.",
        sections: [
            {
                heading: 'On this page',
                bullets: [
                    detail?.description
                        ? '"What this rule detects" explains, in plain words, what makes the rule fire.'
                        : 'This rule has no written explanation yet.',
                    'Boolean Logic shows the exact predicate over its cognitive elements (CEs).',
                    `Cognitive Elements (${ces.length}) — click any to expand its definition and examples.`,
                    'Test & Calibration Set holds the auto-generated positive / negative / calibration dialogues.',
                ],
            },
            {
                heading: 'Tips',
                bullets: [
                    detail?.public_id
                        ? 'This is a published rule — you can rate it and Save (bookmark) it for reuse on your rule sets.'
                        : 'This is a local draft — publish it from Browse or Drafts to share it.',
                    'Roles: Necessary (AND) · Any of (OR within a group, AND across groups) · Supporting (raises confidence, not part of the boolean logic).',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    return (
        <div style={page}>
            <Breadcrumb items={[
                { label: 'Hub', icon: FiHome, to: '/workspace' },
                { label: 'Community', icon: FiUsers, to: '/community' },
                { label: ruleName },
            ]} />
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', margin: '0 0 18px' }}>
                <h1 style={{ margin: 0, fontSize: 22, color: '#f1f5f9' }}>{ruleName}</h1>
                {!loading && !loadError && canEdit && (
                    <button onClick={() => setEditOpen(true)} style={ghostBtn} title="Edit — start a new draft from this rule's elements and logic">
                        <FiEdit2 /> Edit
                    </button>
                )}
            </div>

            {loading && <div style={muted}>Loading…</div>}

            {!loading && loadError && (
                <div style={{ ...card, textAlign: 'center', padding: '32px 16px' }}>
                    <div style={{ color: '#f1f5f9', fontSize: 16, fontWeight: 700, marginBottom: 6 }}>This rule couldn’t be loaded</div>
                    <div style={{ ...muted, fontSize: 13, marginBottom: 18 }}>
                        It may have been removed, or it isn’t available to you. Try one of these instead.
                    </div>
                    <div style={{ display: 'inline-flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
                        <button onClick={() => navigate('/community')} style={primaryBtn}>
                            <FiUsers /> Browse Community
                        </button>
                        <button onClick={() => navigate('/workspace')} style={ghostBtn}>
                            <FiHome /> Go to Hub
                        </button>
                    </div>
                </div>
            )}

            {!loading && !loadError && (
                <>
                    {detail?.public_id && (
                        <div style={{ ...card, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 16, flexWrap: 'wrap' }}>
                            {user?.user_id && (
                                <button
                                    onClick={toggleBookmark}
                                    disabled={bmBusy}
                                    style={bookmarked
                                        ? { ...ghostBtn, color: '#fbbf24', borderColor: 'rgba(251,191,36,0.4)' }
                                        : primaryBtn}
                                >
                                    <FiBookmark /> {bookmarked ? 'Remove' : 'Save'}
                                </button>
                            )}
                        </div>
                    )}

                    {detail?.description && (
                        <div style={card}>
                            <div style={sectionTitle}><FiFileText /> What this rule detects</div>
                            <div style={{ fontSize: 13.5, color: '#f1f5f9', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                                {detail.description}
                            </div>
                        </div>
                    )}

                    {detail?.predicate && (
                        <div style={card}>
                            <div style={sectionTitle}><FiCpu /> Boolean Logic</div>
                            <code style={{ display: 'block', background: 'rgba(2,6,23,0.6)', padding: 10, borderRadius: 8, color: '#cbd5e1', fontSize: 12.5, wordBreak: 'break-word' }}>{detail.predicate}</code>
                        </div>
                    )}

                    <div style={card}>
                        <div style={sectionTitle}><FiCpu /> Cognitive Elements ({ces.length})</div>
                        {ces.length === 0 ? <div style={muted}>No cognitive elements linked to this rule.</div> : ces.map(ce => (
                            <CeRow
                                key={ce.ce_id}
                                ce={ce}
                                bookmarked={ceBmIds.has(ce.ce_id)}
                                onToggleBookmark={user?.user_id ? () => toggleCeBookmark(ce.ce_id) : undefined}
                            />
                        ))}
                    </div>

                    <div style={card}>
                        <div style={sectionTitle}><FiFileText /> Test &amp; Calibration Set</div>
                        {testSet ? (
                            <TestSetView set={testSet} />
                        ) : (
                            <div style={{ ...muted, fontSize: 12.5 }}>
                                No test set yet. The test set is generated when a rule is created or published;
                                seeded library rules pull theirs from HF when first opened.
                            </div>
                        )}
                    </div>
                </>
            )}

            {/* Edit → fork this rule into a new draft via the build-from-CEs wizard. */}
            {canEdit && (
                <BuildRuleFromCEsModal open={editOpen} onClose={() => setEditOpen(false)} baseRule={editBase} />
            )}
        </div>
    );
}
