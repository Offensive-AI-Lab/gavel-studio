import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { showAlertDialog } from '../ConfirmDialog/confirmDialog';
import { FiPenTool, FiTrash2, FiChevronDown, FiChevronUp, FiCpu, FiPlus, FiMinus, FiInfo, FiTag, FiUpload, FiFileText, FiBookmark, FiEdit2 } from 'react-icons/fi';
import GlassModal from '../GlassModal/GlassModal';
import RoleLogicGuide from '../RoleLogicGuide/RoleLogicGuide';
import RuleLogicPreview from '../RuleLogicPreview/RuleLogicPreview';
import BuildRuleFromCEsModal from '../../pages/BuildRuleFromCEsModal';
import { extractLogic } from '../../utils/ruleLogic';
import { getCEBookmarks, addCEBookmark, removeCEBookmark } from '../../api';
import { recordRecent } from '../../utils/recents';
import './RuleCard.css'; // Import its own CSS

const RuleCard = ({
    rule,
    isExpanded,
    onToggle,
    onDelete,
    readOnly = false,
    onBookmark,
    bookmarkLabel = 'Save',
    isBookmarked = false,
    onEditLogic,
}) => {
    // Drafts show a DISABLED Publish button: publishing from Studio is gone —
    // library contributions now go through gavel-rules pull requests. The
    // button stays (disabled, with an explanatory tooltip) so the affordance
    // isn't silently missing for users who knew the old flow.
    const isDraftRule = rule?.is_local_draft === true;
    // CE bookmarking from the rule card — which of this rule's CEs the user has
    // saved. Loaded lazily when the card expands (the CE tags are only shown
    // then), so list views don't each fire a fetch.
    const _ruleCardUser = React.useMemo(() => {
        try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
    }, []);
    const [ceBmIds, setCeBmIds] = React.useState(() => new Set());
    const [ceBmLoaded, setCeBmLoaded] = React.useState(false);
    React.useEffect(() => {
        if (!isExpanded || ceBmLoaded || !_ruleCardUser) return;
        let alive = true;
        Promise.resolve(getCEBookmarks?.(_ruleCardUser.user_id))
            .then((res) => {
                if (!alive) return;
                const ids = (res?.data?.bookmarks || []).map((b) => b.ce_id).filter((x) => x != null);
                setCeBmIds(new Set(ids));
                setCeBmLoaded(true);
            })
            .catch(() => { if (alive) setCeBmLoaded(true); });
        return () => { alive = false; };
    }, [isExpanded, ceBmLoaded, _ruleCardUser]);

    const toggleCeBookmark = async (ce) => {
        const id = ce?.ce_id;
        if (id == null || !_ruleCardUser) return;
        const wasSaved = ceBmIds.has(id);
        setCeBmIds((prev) => { const n = new Set(prev); wasSaved ? n.delete(id) : n.add(id); return n; });
        try {
            if (wasSaved) await removeCEBookmark?.(_ruleCardUser.user_id, id);
            else await addCEBookmark?.(_ruleCardUser.user_id, id);
        } catch {
            // revert on failure
            setCeBmIds((prev) => { const n = new Set(prev); wasSaved ? n.add(id) : n.delete(id); return n; });
            showAlertDialog({ title: 'Error', message: 'Could not update the CE bookmark.', variant: 'error' });
        }
    };
    // The rule's explanation (description) can be long and the card is tight,
    // so it's clamped to a few lines with a Show more/less toggle.
    const [descExpanded, setDescExpanded] = React.useState(false);
    // "How is this boolean built?" explainer modal (reuses the wizard's guide).
    const [logicGuideOpen, setLogicGuideOpen] = React.useState(false);
    const description = (rule?.description || '').trim();
    const descIsLong = description.length > 180;
    const navigate = useNavigate();
    const ruleNavId = rule?.source_rule_id || rule?.rule_id;

    // Opening a rule's CARD (expanding it) counts as "recently opened" too — not
    // just visiting its full Rule page. Records the same unique /rules/<id> path
    // so the sidebar Recents highlight stays per-item (no all-active bug).
    React.useEffect(() => {
        if (isExpanded && ruleNavId != null && rule?.custom_name) {
            recordRecent('rule', { id: ruleNavId, name: rule.custom_name, path: `/rules/${ruleNavId}` });
        }
    }, [isExpanded, ruleNavId, rule?.custom_name]);

    // The rule's firing logic in the groups + condition model. `logic.groups`
    // is null for legacy rows — then the derived predicate string is shown.
    const logic = extractLogic(rule);

    // "Edit" = fork this rule into a NEW draft (build-from-CEs prefilled with
    // this rule's CEs/groups/condition/categories, forced to a new name).
    // Available to a signed-in user when the rule has at least one CE we can
    // carry (needs a ce_id to relink).
    const [editOpen, setEditOpen] = React.useState(false);
    const editableCes = (rule?.active_ces || []).filter((c) => c && c.ce_id != null);
    const canEdit = !!_ruleCardUser && editableCes.length > 0;
    const editBase = {
        name: rule?.custom_name,
        ces: editableCes.map((c) => ({ ce_id: c.ce_id, name: c.name, category: c.category })),
        groups: logic.groups,
        condition: logic.condition,
        categories: rule?.categories || [],
    };

    return (
        <div className={`rule-card ${isExpanded ? 'expanded' : ''}`}>
            {/* HEADER */}
            <div className="rule-header" onClick={onToggle}>
                <div className="rule-info">
                    <div className="rule-icon"><FiPenTool /></div>
                    <div className="rule-title">
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {rule.custom_name}
                            {typeof rule.is_local_draft === 'boolean' && (
                                // Status pill — gradient fill + matching shadow so it
                                // reads as a real UI badge, not a flat tag. Draft is
                                // amber (your work, not yet shipped); Public is
                                // emerald (live in the registry).
                                <span style={{
                                    fontSize: '0.65rem',
                                    fontWeight: 800,
                                    textTransform: 'uppercase',
                                    padding: '3px 10px',
                                    borderRadius: '999px',
                                    color: '#ffffff',
                                    background: rule.is_local_draft
                                        ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                                        : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                                    boxShadow: rule.is_local_draft
                                        ? '0 2px 6px -1px rgba(245, 158, 11, 0.40)'
                                        : '0 2px 6px -1px rgba(16, 185, 129, 0.40)',
                                    letterSpacing: '0.06em',
                                    border: 'none',
                                }}>
                                    {rule.is_local_draft ? 'Draft' : 'Public'}
                                </span>
                            )}
                        </h3>
                        <p>{(rule.active_ces || []).length} Cognitive Elements • {readOnly ? 'Public Rule' : 'Private Rule'}</p>
                        {rule.categories && rule.categories.length > 0 && (
                            <div className="rule-categories" style={{ display: 'flex', gap: '6px', marginTop: '4px', flexWrap: 'wrap', alignItems: 'center' }}>
                                {rule.categories.map((cat) => (
                                    <span key={cat} className="pill pill-soft" style={{ fontSize: '0.75rem', padding: '4px 10px' }}>{cat}</span>
                                ))}
                                {rule.created_by_username && (
                                    <Link
                                        to={`/profile/${rule.created_by_username}`}
                                        onClick={(e) => e.stopPropagation()}
                                        style={{
                                            fontSize: '0.78rem',
                                            color: '#a5b4fc',
                                            textDecoration: 'none',
                                            padding: '2px 8px',
                                            borderRadius: '8px',
                                            background: 'rgba(99, 102, 241, 0.10)',
                                            border: '1px solid rgba(129, 140, 248, 0.25)',
                                            fontWeight: 600,
                                        }}
                                    >
                                        by @{rule.created_by_username}
                                    </Link>
                                )}
                            </div>
                        )}
                        {/* If a rule has no categories but DOES have an
                          * author, the link gets its own row so it's not
                          * lost. Categories-less drafts and the public
                          * library both hit this branch. */}
                        {(!rule.categories || rule.categories.length === 0) && rule.created_by_username && (
                            <div style={{ marginTop: '4px' }}>
                                <Link
                                    to={`/profile/${rule.created_by_username}`}
                                    onClick={(e) => e.stopPropagation()}
                                    style={{
                                        fontSize: '0.78rem',
                                        color: '#a5b4fc',
                                        textDecoration: 'none',
                                        padding: '2px 8px',
                                        borderRadius: '8px',
                                        background: 'rgba(99, 102, 241, 0.10)',
                                        border: '1px solid rgba(129, 140, 248, 0.25)',
                                        fontWeight: 600,
                                    }}
                                >
                                    by @{rule.created_by_username}
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
                <div className="rule-actions">
                    {/* Drafts can't be bookmarked here — this bookmark path is
                        keyed by public_id (HF identifier), which drafts don't
                        have yet. Drafts live in "My Drafts" instead. */}
                    {onBookmark && !rule.is_local_draft && rule.public_id && (
                        <button
                            className="bookmark-btn"
                            onClick={(e) => {
                                e.stopPropagation();
                                onBookmark(rule);
                            }}
                            aria-label="Bookmark rule"
                        >
                            {isBookmarked ? <FiMinus /> : <FiPlus />}
                            {isBookmarked ? 'Remove' : bookmarkLabel}
                        </button>
                    )}
                    {isDraftRule && (
                        <button
                            className="bookmark-btn publish-btn"
                            disabled
                            onClick={(e) => e.stopPropagation()}
                            aria-label="Publish rule to library (coming soon)"
                            title="Library contributions move to gavel-rules pull requests — submission from Studio coming soon"
                        >
                            <FiUpload />
                            Publish
                        </button>
                    )}
                    {ruleNavId && (
                        <button
                            className="bookmark-btn"
                            onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/rules/${ruleNavId}`);
                            }}
                            aria-label="Open this rule's page"
                            title="Open the full rule page — CEs, examples, test sets, rating"
                        >
                            <FiFileText />
                            Rule page
                        </button>
                    )}
                    {canEdit && (
                        <button
                            className="bookmark-btn"
                            onClick={(e) => { e.stopPropagation(); setEditOpen(true); }}
                            aria-label="Edit this rule as a new draft"
                            title="Edit — start a new draft from this rule's elements and logic"
                        >
                            <FiEdit2 />
                            Edit
                        </button>
                    )}
                    {typeof onEditLogic === 'function' && !readOnly && (
                        <button
                            className="bookmark-btn"
                            onClick={(e) => { e.stopPropagation(); onEditLogic(rule); }}
                            aria-label="Edit this rule's groups and condition in place"
                            title="Edit logic — change this rule's CE groups and firing condition"
                        >
                            <FiEdit2 />
                            Edit logic
                        </button>
                    )}
                    {!readOnly && (
                        <FiTrash2
                            className="delete-icon"
                            onClick={(e) => {
                                e.stopPropagation();
                                onDelete(rule.setup_id);
                            }}
                        />
                    )}
                    {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
                </div>
            </div>

            {/* EXPANDED CONTENT */}
            {isExpanded && (
                <div className="rule-content">
                    {description && (
                        <div style={{ marginBottom: '16px' }}>
                            <div className="content-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                                <FiFileText /> What this rule detects
                            </div>
                            <p
                                style={{
                                    margin: '6px 0 0',
                                    fontSize: '0.85rem',
                                    lineHeight: 1.55,
                                    color: '#f1f5f9',
                                    ...(descIsLong && !descExpanded
                                        ? { display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }
                                        : {}),
                                }}
                            >
                                {description}
                            </p>
                            {descIsLong && (
                                <button
                                    type="button"
                                    onClick={(e) => { e.stopPropagation(); setDescExpanded((v) => !v); }}
                                    style={{
                                        marginTop: '4px', padding: 0, background: 'none', border: 'none',
                                        color: '#6366f1', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                                    }}
                                >
                                    {descExpanded ? 'Show less' : 'Show more'}
                                </button>
                            )}
                        </div>
                    )}

                    <div className="content-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                        <FiCpu /> Boolean Logic
                        <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setLogicGuideOpen(true); }}
                            title="How is this boolean logic built?"
                            style={{
                                display: 'inline-flex', alignItems: 'center', gap: 5,
                                border: '1px solid rgba(129, 140, 248, 0.40)', background: 'rgba(99, 102, 241, 0.15)',
                                color: '#c7d2fe', borderRadius: 999, padding: '3px 10px', cursor: 'pointer',
                                fontSize: '0.72rem', fontWeight: 600, textTransform: 'none', letterSpacing: 0,
                            }}
                        >
                            <FiInfo size={13} /> How it works
                        </button>
                    </div>

                    {logic.groups ? (
                        <RuleLogicPreview title={null} groups={logic.groups} condition={logic.condition} />
                    ) : (
                        // Legacy row without groups — show the derived predicate string.
                        <div className="code-box">{logic.predicate || rule.predicate}</div>
                    )}

                    <div className="ce-tags-list">
                        <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: '#475569', fontWeight: 600 }}>
                            <span>Cognitive Elements</span>
                            <FiInfo title="Every CE the rule trains on — the groups and condition above decide which combinations make it fire." />
                        </div>
                        {(rule.active_ces || []).map((ce, ci) => (
                            <div key={ce.ce_id || ce.name || ci} className="ce-tag" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', alignSelf: 'stretch' }}>
                                    <span>{ce.name || ce}</span>
                                    {/* Bookmark this CE to your Library. Only for published CEs
                                      * (have a ce_id) when a user is signed in. */}
                                    {ce.ce_id != null && _ruleCardUser && !ce.is_local_draft && (
                                        <button
                                            onClick={(e) => { e.stopPropagation(); toggleCeBookmark(ce); }}
                                            title={ceBmIds.has(ce.ce_id) ? 'Remove CE from your Library' : 'Save CE to your Library'}
                                            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: ceBmIds.has(ce.ce_id) ? '#fcd34d' : '#94a3b8', display: 'inline-flex', alignItems: 'center', padding: 2 }}
                                        >
                                            <FiBookmark size={14} fill={ceBmIds.has(ce.ce_id) ? '#fcd34d' : 'none'} />
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Boolean-logic explainer — the SAME guide shown in the Build-Rule
              * wizard, but here just the content + an X to close (no wizard steps).
              * Lives on the card so it's available wherever a RuleCard renders. */}
            <GlassModal isOpen={logicGuideOpen} onClose={() => setLogicGuideOpen(false)} title="How the boolean logic is built" size="wide">
                <RoleLogicGuide />
            </GlassModal>

            {/* Edit → fork this rule into a new draft via the build-from-CEs wizard. */}
            {canEdit && (
                <BuildRuleFromCEsModal open={editOpen} onClose={() => setEditOpen(false)} baseRule={editBase} />
            )}
        </div>
    );
};

export default RuleCard;