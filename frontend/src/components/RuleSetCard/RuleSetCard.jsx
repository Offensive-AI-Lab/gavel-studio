// RuleSetCard — one public rule set in the Community "Rule Sets" tab.
//
// A rule set is a model-agnostic, shareable collection of already-published
// rules. This card mirrors RuleCard's chrome (Public pill, "by @author", a
// Save/Remove bookmark button gated on a public_id) but its body lists the
// MEMBER RULES rather than CEs, and its primary action is "Fork into my
// workspace" (clone the set into a new private, model-less rule set).

import { Link, useNavigate } from 'react-router-dom';
import { FiLayers, FiChevronDown, FiChevronUp, FiPlus, FiMinus, FiFileText } from 'react-icons/fi';

const RuleSetCard = ({
    ruleSet,
    isExpanded,
    onToggle,
    onBookmark,
    bookmarkLabel = 'Save',
    isBookmarked = false,
}) => {
    const navigate = useNavigate();
    const members = Array.isArray(ruleSet.member_rules) ? ruleSet.member_rules : [];
    const categories = Array.isArray(ruleSet.categories) ? ruleSet.categories.filter(Boolean) : [];
    const author = ruleSet.created_by_username;
    const hasPublicId = !!ruleSet.public_id;
    // Match RuleCard: a Draft/Public pill renders only when the row actually
    // carries the boolean; a bookmark button only when it's public + bookmarkable.
    const showStatus = typeof ruleSet.is_local_draft === 'boolean';
    const isDraft = ruleSet.is_local_draft === true;
    const canBookmark = !!onBookmark && !isDraft && hasPublicId;

    return (
        <div className="rule-card" style={cardStyle}>
            <div style={headerStyle}>
                <button onClick={onToggle} style={titleBtnStyle} aria-expanded={isExpanded}>
                    <span style={iconWrapStyle}><FiLayers size={18} /></span>
                    <span style={{ minWidth: 0 }}>
                        <span style={nameStyle}>{ruleSet.name || 'Rule set'}</span>
                        <span style={metaRowStyle}>
                            {members.length} rule{members.length === 1 ? '' : 's'}
                            {showStatus && (
                                <span style={isDraft ? draftPill : publicPill}>
                                    {isDraft ? 'Draft' : 'Public'}
                                </span>
                            )}
                            {author && (
                                <Link
                                    to={`/profile/${author}`}
                                    onClick={(e) => e.stopPropagation()}
                                    style={authorLinkStyle}
                                >
                                    by @{author}
                                </Link>
                            )}
                        </span>
                    </span>
                    <span style={{ marginLeft: 'auto', color: '#94a3b8' }}>
                        {isExpanded ? <FiChevronUp size={18} /> : <FiChevronDown size={18} />}
                    </span>
                </button>
            </div>

            {categories.length > 0 && (
                <div style={chipRowStyle}>
                    {categories.map((c) => (
                        <span key={c} style={categoryChipStyle}>{c}</span>
                    ))}
                </div>
            )}

            <div style={actionRowStyle}>
                {canBookmark && (
                    <button className="bookmark-btn" onClick={() => onBookmark(ruleSet)}>
                        {isBookmarked ? <FiMinus size={14} /> : <FiPlus size={14} />}
                        {isBookmarked ? 'Remove' : bookmarkLabel}
                    </button>
                )}
                {hasPublicId && (
                    <button style={ghostBtnStyle} onClick={() => navigate(`/rule-sets/${ruleSet.public_id}`)}>
                        <FiFileText size={14} /> Rule set page
                    </button>
                )}
            </div>

            {isExpanded && (
                <div style={bodyStyle}>
                    {ruleSet.description && (
                        <p style={{ color: '#cbd5e1', margin: '0 0 12px', fontSize: '0.9rem' }}>{ruleSet.description}</p>
                    )}
                    <h4 style={sectionTitleStyle}>Rules in this set</h4>
                    {members.length === 0 ? (
                        <p style={{ color: '#94a3b8', fontSize: '0.88rem' }}>No member rules.</p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {members.map((m) => (
                                <Link
                                    key={m.rule_id ?? m.public_id}
                                    to={`/rules/${m.rule_id}`}
                                    style={memberRowStyle}
                                >
                                    <FiFileText size={13} style={{ flexShrink: 0, color: '#a5b4fc' }} />
                                    <span style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.88rem' }}>{m.name}</span>
                                </Link>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const cardStyle = {
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.72) 0%, rgba(15, 23, 42, 0.60) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '14px',
    padding: '16px 18px',
    marginBottom: '14px',
    boxShadow: '0 6px 18px -8px rgba(2, 6, 23, 0.45)',
};
const headerStyle = { display: 'flex', alignItems: 'center', gap: '10px' };
const titleBtnStyle = {
    display: 'flex', alignItems: 'center', gap: '12px', width: '100%',
    background: 'none', border: 'none', cursor: 'pointer', padding: 0, textAlign: 'left',
};
const iconWrapStyle = {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: '38px', height: '38px', borderRadius: '10px', flexShrink: 0,
    background: 'linear-gradient(135deg, rgba(129,140,248,0.25), rgba(59,130,246,0.18))',
    color: '#c7d2fe',
};
const nameStyle = { display: 'block', color: '#f1f5f9', fontWeight: 700, fontSize: '1.02rem', letterSpacing: '-0.01em' };
const metaRowStyle = { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', color: '#94a3b8', fontSize: '0.82rem', marginTop: '2px' };
const publicPill = { padding: '1px 8px', borderRadius: '999px', background: 'rgba(16,185,129,0.18)', color: '#6ee7b7', border: '1px solid rgba(16,185,129,0.4)', fontWeight: 700, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.04em' };
const draftPill = { padding: '1px 8px', borderRadius: '999px', background: 'rgba(245,158,11,0.18)', color: '#fcd34d', border: '1px solid rgba(245,158,11,0.4)', fontWeight: 700, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.04em' };
const authorLinkStyle = { color: '#a5b4fc', textDecoration: 'none', fontWeight: 600 };
const chipRowStyle = { display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '10px' };
const categoryChipStyle = { padding: '2px 10px', borderRadius: '999px', background: 'rgba(99,102,241,0.14)', border: '1px solid rgba(129,140,248,0.32)', color: '#c7d2fe', fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.03em' };
const actionRowStyle = { display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '12px' };
const ghostBtnStyle = { display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '6px 14px', borderRadius: '999px', border: '1px solid rgba(148,163,184,0.25)', background: 'rgba(15,23,42,0.55)', color: '#cbd5e1', cursor: 'pointer', fontWeight: 600, fontSize: '0.84rem' };
const bodyStyle = { marginTop: '14px', paddingTop: '14px', borderTop: '1px solid rgba(148,163,184,0.14)' };
const sectionTitleStyle = { margin: '0 0 8px', color: '#e2e8f0', fontSize: '0.86rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' };
const memberRowStyle = { display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', borderRadius: '10px', border: '1px solid rgba(148,163,184,0.14)', background: 'rgba(2,6,23,0.5)', textDecoration: 'none' };

export default RuleSetCard;
