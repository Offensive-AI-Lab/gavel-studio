// Rule set page — everything about ONE public rule set: its member rules (each
// expandable to show predicate + Cognitive Elements with roles), plus Bookmark
// and "Fork into my workspace" actions. Model-agnostic, Community-scoped.
import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
    FiHome, FiUsers, FiLayers, FiChevronDown, FiChevronUp, FiCpu, FiFileText,
    FiPlus, FiMinus,
} from 'react-icons/fi';
import {
    getRuleSetDetail, getRuleSetBookmarks, addRuleSetBookmark, removeRuleSetBookmark,
} from '../api';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import RuleLogicPreview from '../components/RuleLogicPreview/RuleLogicPreview';
import { extractLogic } from '../utils/ruleLogic';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import { useTutorialContent } from '../contexts/TutorialContext';

function readUser() {
    try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
}

const page = { padding: '28px 32px', maxWidth: 1000, margin: '0 auto' };
const card = { background: 'rgba(15,23,42,0.55)', border: '1px solid rgba(148,163,184,0.16)', borderRadius: 12, padding: 16, marginBottom: 16 };
const sectionTitle = { display: 'flex', alignItems: 'center', gap: 8, fontSize: 15, fontWeight: 700, color: '#e2e8f0', marginBottom: 12 };
const muted = { color: '#94a3b8' };
const primaryBtn = { display: 'inline-flex', alignItems: 'center', gap: 6, background: 'linear-gradient(135deg,#818cf8,#3b82f6)', color: '#fff', border: 'none', borderRadius: 8, padding: '9px 14px', fontWeight: 700, cursor: 'pointer' };
const ghostBtn = { display: 'inline-flex', alignItems: 'center', gap: 6, background: 'rgba(148,163,184,0.12)', color: '#cbd5e1', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 8, padding: '7px 12px', fontWeight: 600, cursor: 'pointer' };
const chipS = (bg, c) => ({ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '2px 8px', borderRadius: 999, fontSize: 11, fontWeight: 600, background: bg, color: c });

function MemberRule({ rule }) {
    const [open, setOpen] = useState(false);
    const ces = Array.isArray(rule.active_ces) ? rule.active_ces : [];
    const logic = extractLogic(rule);
    return (
        <div style={{ border: '1px solid rgba(148,163,184,0.14)', borderRadius: 10, background: 'rgba(2,6,23,0.45)', marginBottom: 8 }}>
            <button
                onClick={() => setOpen(!open)}
                style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '12px 14px', textAlign: 'left' }}
            >
                <FiFileText size={15} style={{ color: '#a5b4fc', flexShrink: 0 }} />
                <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.92rem' }}>{rule.name}</span>
                <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 12 }}>{ces.length} CE{ces.length === 1 ? '' : 's'}</span>
                {open ? <FiChevronUp size={16} color="#94a3b8" /> : <FiChevronDown size={16} color="#94a3b8" />}
            </button>
            {open && (
                <div style={{ padding: '0 14px 14px' }}>
                    {logic.groups ? (
                        <RuleLogicPreview title={null} groups={logic.groups} condition={logic.condition} style={{ marginBottom: 10 }} />
                    ) : rule.predicate ? (
                        // Legacy row without groups — derived predicate only.
                        <div style={{ ...muted, fontSize: 12.5, marginBottom: 10, fontFamily: 'monospace' }}>{rule.predicate}</div>
                    ) : null}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {ces.map((ce) => (
                            <div key={ce.ce_id ?? ce.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <FiCpu size={13} style={{ color: '#67e8f9', flexShrink: 0 }} />
                                <span style={{ color: '#e2e8f0', fontSize: 13 }}>{ce.name}</span>
                            </div>
                        ))}
                        {ces.length === 0 && <span style={muted}>No cognitive elements.</span>}
                    </div>
                    {rule.rule_id != null && (
                        <Link to={`/rules/${rule.rule_id}`} style={{ ...ghostBtn, marginTop: 12, textDecoration: 'none' }}>
                            <FiFileText size={13} /> Open rule page
                        </Link>
                    )}
                </div>
            )}
        </div>
    );
}

export default function RuleSetPage() {
    const { ruleSetPublicId } = useParams();
    const navigate = useNavigate();
    const user = readUser();
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(true);
    const [notFound, setNotFound] = useState(false);
    const [isBookmarked, setIsBookmarked] = useState(false);

    const load = useCallback(async () => {
        setLoading(true);
        try {
            const res = await getRuleSetDetail(ruleSetPublicId);
            setDetail(res.data);
            if (user?.user_id) {
                try {
                    const bk = await getRuleSetBookmarks(user.user_id);
                    const ids = new Set((bk.data?.bookmarks || []).map((b) => b.rule_set_id));
                    setIsBookmarked(ids.has(res.data.rule_set_id));
                } catch { /* non-fatal */ }
            }
        } catch (e) {
            if (e?.response?.status === 404) setNotFound(true);
        } finally {
            setLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [ruleSetPublicId]);

    useEffect(() => { load(); }, [load]);

    useTutorialContent({
        title: 'Community · Rule Set',
        summary: 'A shared, model-agnostic collection of published rules. Expand each rule to see its cognitive elements, then fork the set into your workspace to customize and train it.',
    });

    const toggleBookmark = async () => {
        if (!user?.user_id) return showAlertDialog({ title: 'Sign in', message: 'Sign in to bookmark rule sets.', variant: 'info' });
        try {
            if (isBookmarked) {
                await removeRuleSetBookmark(user.user_id, detail.rule_set_id);
                setIsBookmarked(false);
            } else {
                await addRuleSetBookmark(user.user_id, detail.rule_set_id);
                setIsBookmarked(true);
            }
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not update bookmark.', variant: 'error' });
        }
    };

    if (loading) return <div style={page}><p style={muted}>Loading…</p></div>;
    if (notFound || !detail) {
        return (
            <div style={page}>
                <Breadcrumb items={[
                    { label: 'Hub', icon: FiHome, to: '/workspace' },
                    { label: 'Community', icon: FiUsers, to: '/community/rule-sets' },
                    { label: 'Rule set' },
                ]} />
                <div style={card}><p style={muted}>This rule set could not be found.</p></div>
                <button style={ghostBtn} onClick={() => navigate('/community/rule-sets')}>Back to Rule Sets</button>
            </div>
        );
    }

    const members = Array.isArray(detail.member_rules) ? detail.member_rules : [];
    const categories = Array.isArray(detail.categories) ? detail.categories.filter(Boolean) : [];
    const author = detail.created_by_username;

    return (
        <div style={page}>
            <Breadcrumb items={[
                { label: 'Hub', icon: FiHome, to: '/workspace' },
                { label: 'Community', icon: FiUsers, to: '/community/rule-sets' },
                { label: 'Rule Sets', icon: FiLayers, to: '/community/rule-sets' },
                { label: detail.name || 'Rule set' },
            ]} />

            <div style={card}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 44, height: 44, borderRadius: 12, background: 'linear-gradient(135deg, rgba(129,140,248,0.25), rgba(59,130,246,0.18))', color: '#c7d2fe', flexShrink: 0 }}>
                        <FiLayers size={22} />
                    </span>
                    <div style={{ minWidth: 0, flex: 1 }}>
                        <h1 style={{ margin: 0, fontSize: 22, color: '#f8fafc' }}>{detail.name}</h1>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginTop: 4, ...muted, fontSize: 13 }}>
                            <span>{members.length} rule{members.length === 1 ? '' : 's'}</span>
                            {author && <Link to={`/profile/${author}`} style={{ color: '#a5b4fc', textDecoration: 'none', fontWeight: 600 }}>by @{author}</Link>}
                        </div>
                        {detail.description && <p style={{ ...muted, marginTop: 10, marginBottom: 0 }}>{detail.description}</p>}
                        {categories.length > 0 && (
                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                                {categories.map((c) => <span key={c} style={chipS('rgba(99,102,241,0.14)', '#c7d2fe')}>{c}</span>)}
                            </div>
                        )}
                    </div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 16 }}>
                    <button style={primaryBtn} onClick={toggleBookmark}>
                        {isBookmarked ? <FiMinus size={14} /> : <FiPlus size={14} />}
                        {isBookmarked ? 'Remove bookmark' : 'Bookmark'}
                    </button>
                </div>
            </div>

            <div style={card}>
                <div style={sectionTitle}><FiFileText size={16} /> Rules in this set</div>
                {members.length === 0 ? (
                    <p style={muted}>This rule set has no rules.</p>
                ) : (
                    members.map((r) => <MemberRule key={r.rule_id ?? r.public_id} rule={r} />)
                )}
            </div>
        </div>
    );
}
