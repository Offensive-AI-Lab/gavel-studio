import React from 'react';
import { Link } from 'react-router-dom';
import { FiChevronDown, FiChevronUp, FiPlus, FiMinus, FiDownload, FiTrash2, FiFileText } from 'react-icons/fi';
import ExportToRegistryModal from '../ExportToRegistry/ExportToRegistryModal';
import { prettyName } from '../../utils/ruleLogic';
import './CognitiveElementCard.css';

const CognitiveElementCard = ({
    ce,
    isOpen,
    onToggle,
    samples,
    onBookmark,
    isBookmarked,
    bookmarkLabel,
    onDelete,
}) => {
    // Contributions go out as gavel-rules pull requests, so the card's outbound
    // action is Export (registry-format files), not an in-studio publish.
    const [exportOpen, setExportOpen] = React.useState(false);
    // The definition can be long and the header only shows a one-line preview
    // (clamped next to the action buttons), so the full text lives in the
    // expanded body — mirroring RuleCard's "What this rule detects" block, with
    // a Show more/less toggle for long definitions.
    const [defExpanded, setDefExpanded] = React.useState(false);
    const definition = (ce?.definition || '').trim();
    const defIsLong = definition.length > 180;
    // Show the delete affordance only when the caller wires it up (currently
    // the Drafts page). Keeps Browse / Bookmarks untouched.
    const canDelete = typeof onDelete === 'function';
    // v2 CEs carry a closed-enum role (directive_to_user / llm_task /
    // llm_behavior / topic) — the primary badge — plus an optional human
    // title (display name) and free-form tags.
    const role = (ce?.role || '').trim();
    const displayName = (ce?.title || '').trim() || ce?.name;
    const tags = Array.isArray(ce?.tags) ? ce.tags.filter(Boolean) : [];
    return (
        <div className={`ce-card ${isOpen ? 'expanded' : ''}`}>
            <div className="ce-header" onClick={() => onToggle(ce.ce_id)}>
                <div className="ce-info">
                    <div className="ce-icon">CE</div>
                    <div className="ce-title">
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }} title={ce.name}>
                            {displayName}
                            {role && (
                                // Primary badge: the CE's v2 role (closed enum,
                                // underscores pretty-printed for display).
                                <span className="ce-role-pill" style={{
                                    fontSize: '0.65rem',
                                    fontWeight: 800,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    padding: '3px 10px',
                                    borderRadius: '999px',
                                    color: '#c7d2fe',
                                    background: 'rgba(99, 102, 241, 0.18)',
                                    border: '1px solid rgba(129, 140, 248, 0.40)',
                                }}>
                                    {prettyName(role)}
                                </span>
                            )}
                            {typeof ce.is_local_draft === 'boolean' && (
                                // Match RuleCard's badge style — see RuleCard.jsx
                                // for design rationale.
                                <span style={{
                                    fontSize: '0.65rem',
                                    fontWeight: 800,
                                    textTransform: 'uppercase',
                                    padding: '3px 10px',
                                    borderRadius: '999px',
                                    color: '#ffffff',
                                    background: ce.is_local_draft
                                        ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                                        : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                                    boxShadow: ce.is_local_draft
                                        ? '0 2px 6px -1px rgba(245, 158, 11, 0.40)'
                                        : '0 2px 6px -1px rgba(16, 185, 129, 0.40)',
                                    letterSpacing: '0.06em',
                                    border: 'none',
                                }}>
                                    {ce.is_local_draft ? 'Draft' : 'Public'}
                                </span>
                            )}
                        </h3>
                        {/* Definition preview — clamped to 2 wrapped lines so a
                          * long definition never forces the card (and the page)
                          * to scroll sideways. The full text is in the expanded
                          * body under "What this CE means". */}
                        <p>{ce.definition || 'No definition provided'}</p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', marginTop: '6px' }}>
                            {ce.categories && ce.categories.length > 0 && (
                                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                    {ce.categories.map((c) => (
                                        <span key={c} className="pill pill-soft" style={{ fontSize: '0.75rem', padding: '4px 10px' }}>{c}</span>
                                    ))}
                                </div>
                            )}
                            {tags.length > 0 && (
                                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                    {tags.map((t) => (
                                        <span key={t} style={{
                                            fontSize: '0.72rem', padding: '2px 8px', borderRadius: '999px',
                                            color: '#94a3b8', background: 'rgba(148, 163, 184, 0.10)',
                                            border: '1px solid rgba(148, 163, 184, 0.24)',
                                        }}>#{t}</span>
                                    ))}
                                </div>
                            )}
                            {/* Author link. Only renders for CEs that flow
                              * through the publish pipeline (created_by_username
                              * is populated). Drafts and legacy rows without an
                              * author simply don't show the link. Clicking
                              * navigates to /profile/<username> — Profile.jsx
                              * routes that. stopPropagation prevents the click
                              * from bubbling up to the card-toggle handler. */}
                            {ce.created_by_username && (
                                <Link
                                    to={`/profile/${ce.created_by_username}`}
                                    onClick={(e) => e.stopPropagation()}
                                    style={authorLinkStyle}
                                >
                                    by @{ce.created_by_username}
                                </Link>
                            )}
                        </div>
                    </div>
                </div>
                <div className="ce-actions" onClick={(e) => e.stopPropagation()}>
                    {/* Drafts can't be bookmarked here — this bookmark path is
                        keyed by public_id, which drafts don't have yet.
                        Drafts live in "My Drafts". */}
                    {onBookmark && !ce.is_local_draft && ce.public_id && (
                        <button
                            className="bookmark-btn"
                            onClick={() => onBookmark(ce)}
                            aria-label="Bookmark CE"
                        >
                            {isBookmarked ? <FiMinus /> : <FiPlus />}
                            {bookmarkLabel || (isBookmarked ? 'Remove' : 'Save')}
                        </button>
                    )}
                    {ce?.ce_id && (
                        <button
                            className="bookmark-btn"
                            onClick={(e) => { e.stopPropagation(); setExportOpen(true); }}
                            aria-label="Export cognitive element for a gavel-rules pull request"
                            title="Save this CE in the public library's format, ready to contribute"
                        >
                            <FiDownload />
                            Export
                        </button>
                    )}
                    {canDelete && (
                        <FiTrash2
                            className="delete-icon"
                            onClick={(e) => { e.stopPropagation(); onDelete(ce); }}
                            title="Delete this draft"
                        />
                    )}
                    {isOpen ? <FiChevronUp /> : <FiChevronDown />}
                </div>
            </div>

            {isOpen && (
                <div className="ce-content">
                    {ce.categories && ce.categories.length > 0 && (
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '10px' }}>
                            {ce.categories.map((c) => (
                                <span key={c} className="pill pill-soft">{c}</span>
                            ))}
                        </div>
                    )}
                    {/* Full definition — shown here (not just the clamped header
                      * preview) so the whole explanation is readable on open,
                      * the same way RuleCard surfaces its description. */}
                    {definition && (
                        <div style={{ marginBottom: '18px' }}>
                            <div className="content-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                                <FiFileText /> What this CE means
                            </div>
                            <p
                                style={{
                                    margin: '6px 0 0',
                                    fontSize: '0.85rem',
                                    lineHeight: 1.55,
                                    color: '#f1f5f9',
                                    whiteSpace: 'pre-wrap',
                                    ...(defIsLong && !defExpanded
                                        ? { display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }
                                        : {}),
                                }}
                            >
                                {definition}
                            </p>
                            {defIsLong && (
                                <button
                                    type="button"
                                    onClick={(e) => { e.stopPropagation(); setDefExpanded((v) => !v); }}
                                    style={{
                                        marginTop: '4px', padding: 0, background: 'none', border: 'none',
                                        color: '#6366f1', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                                    }}
                                >
                                    {defExpanded ? 'Show less' : 'Show more'}
                                </button>
                            )}
                        </div>
                    )}

                    <span className="content-label">Examples</span>
                    {(() => {
                        const rawExamples = Array.isArray(ce.examples) ? ce.examples : [];
                        if (rawExamples.length === 0) {
                            return <div style={{ color: '#94a3b8', marginTop: '8px' }}>No examples available.</div>;
                        }
                        return (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                                {rawExamples.map((ex, idx) => {
                                    const input = typeof ex === 'string' ? ex : (ex?.input ?? '');
                                    const output = typeof ex === 'object' ? (ex?.output ?? 'YES') : 'YES';
                                    const isYes = String(output).toUpperCase() === 'YES';
                                    return (
                                        <div key={idx} style={{
                                            background: 'rgba(2, 6, 23, 0.55)',
                                            border: '1px solid rgba(148, 163, 184, 0.18)',
                                            borderRadius: '10px',
                                            padding: '10px 12px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            gap: '12px',
                                        }}>
                                            <span style={{ color: '#e2e8f0', whiteSpace: 'pre-wrap', flex: 1 }}>{input}</span>
                                            <span style={{
                                                fontSize: '0.7rem',
                                                fontWeight: 700,
                                                padding: '3px 10px',
                                                borderRadius: '999px',
                                                color: isYes ? '#6ee7b7' : '#fca5a5',
                                                background: isYes ? 'rgba(16, 185, 129, 0.20)' : 'rgba(239, 68, 68, 0.18)',
                                                border: `1px solid ${isYes ? 'rgba(52, 211, 153, 0.40)' : 'rgba(248, 113, 113, 0.40)'}`,
                                                letterSpacing: '0.5px',
                                            }}>{output}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        );
                    })()}
                </div>
            )}

            {exportOpen && (
                <ExportToRegistryModal
                    open={exportOpen}
                    onClose={() => setExportOpen(false)}
                    kind="ce"
                    entityId={ce.ce_id}
                    defaultTitle={ce?.title || ''}
                />
            )}
        </div>
    );
};

// Inline so the card stays self-contained. If we lift more author UI
// later (avatars, tooltips), promote this into the CSS file.
const authorLinkStyle = {
    fontSize: '0.78rem',
    color: '#a5b4fc',
    textDecoration: 'none',
    padding: '2px 8px',
    borderRadius: '8px',
    background: 'rgba(99, 102, 241, 0.10)',
    border: '1px solid rgba(129, 140, 248, 0.25)',
    fontWeight: 600,
    transition: 'background 180ms ease, color 180ms ease',
};

export default CognitiveElementCard;
