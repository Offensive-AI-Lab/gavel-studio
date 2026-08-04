// Profile.jsx — profile page for a library contributor.
//
// Reachable at /profile/:username. Resolves a username to a profile card;
// the "by [username]" link on every rule and CE card lands here.
//
// Layout:
//   * Header card — display name, @username, member-since, is_team badge,
//     stats (contribution counts), and an
//     "Edit profile" button when you're viewing your own profile.
//   * Bio paragraph (read-only on others; editable inline on own profile).
//   * Tab bar — Rules / CEs.
//   * Paginated list of the user's published contributions of the
//     active tab type.
//

import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    FiArrowLeft, FiHome, FiUsers, FiUser, FiAward, FiEdit2,
    FiSave, FiX,
} from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import Pagination from '../components/Pagination/Pagination';
import RuleCard from '../components/RuleCard/RuleCard';
import CognitiveElementCard from '../components/CognitiveElementCard/CognitiveElementCard';
import {
    getUserProfile, updateMyProfile,
    searchLibrary,
    addRuleBookmark, removeRuleBookmark, getRuleBookmarks,
    addCEBookmark, removeCEBookmark, getCEBookmarks,
    getCognitiveDataset,
} from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';

const Profile = () => {
    const { username } = useParams();
    const navigate = useNavigate();
    const currentUser = JSON.parse(sessionStorage.getItem('user') || 'null');
    const isOwnProfile = !!currentUser && currentUser.username === (username || '').toLowerCase();

    const [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [notFound, setNotFound] = useState(false);

    const [tab, setTab] = useState('rule');
    const [page, setPage] = useState(1);
    const [pageSize] = useState(10);
    const [contribLoading, setContribLoading] = useState(false);
    const [contribData, setContribData] = useState({ items: [], total: 0 });
    // Bumped by useLibraryRefresh to force the contributions effect to
    // re-fetch after any library-mutation event (bookmark, publish, etc.)
    // even when the (profile, tab, page, pageSize) tuple is unchanged.
    const [contribRefreshTick, setContribRefreshTick] = useState(0);

    // Card-expansion state. Single id at a time so opening one rule
    // collapses any previously-open one — matches Browse's behavior.
    const [expandedRuleId, setExpandedRuleId] = useState(null);
    const [expandedCeId, setExpandedCeId] = useState(null);
    // CE expanded view fetches its excitation samples on first open.
    const [previewCache, setPreviewCache] = useState({});
    // True sample count per CE — the preview itself is capped at 10.
    const [previewTotals, setPreviewTotals] = useState({});

    // Bookmark state (for the visitor, not the profile owner). Lets
    // someone "save" an artist's rule/CE straight from the profile.
    const [ruleBookmarkIds, setRuleBookmarkIds] = useState(new Set());
    const [ceBookmarkIds, setCeBookmarkIds] = useState(new Set());

    // Edit-mode state (only meaningful on own profile).
    const [editing, setEditing] = useState(false);
    const [editDisplayName, setEditDisplayName] = useState('');
    const [editBio, setEditBio] = useState('');
    const [saving, setSaving] = useState(false);

    const pageHelp = {
        title: 'Profile',
        summary:
            'Profile pages show what an author has contributed to the library — their published rules and CEs. Click any "by [username]" link in Browse to get here.',
        sections: [
            {
                heading: 'On your own profile',
                bullets: [
                    'Click Edit Profile to update your display name and bio.',
                    'Your username is permanent — it can\'t be changed.',
                    'Your stats update automatically as people rate your contributions.',
                ],
            },
            {
                heading: 'On another user\'s profile',
                bullets: [
                    'Browse their published rules and CEs in the tabs below.',
                    'Click any item to view it in the public library.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    // refetchProfile is callable from both the URL-change effect AND
    // the library-changed listener below, so a publish from this page
    // updates the header stats (rules / CEs counts)
    // immediately without a page refresh.
    //
    // The `silent` flag skips the full-page spinner — used by the
    // library-change refresh, which should never blink the screen.
    const refetchProfile = useCallback(async (silent = false) => {
        if (!silent) {
            setLoading(true);
            setNotFound(false);
        }
        try {
            const res = await getUserProfile(username);
            setProfile(res.data);
            // Only seed edit-mode values on the first / non-silent
            // load. We don't want a background refetch to clobber
            // the user's in-progress edits.
            if (!silent) {
                setEditDisplayName(res.data.display_name || '');
                setEditBio(res.data.bio || '');
            }
        } catch (err) {
            if (err.response?.status === 404) {
                setNotFound(true);
            } else if (!silent) {
                showAlertDialog({
                    title: 'Could not load profile',
                    message: err.response?.data?.detail || err.message || 'Unknown error',
                    variant: 'error',
                });
            }
        } finally {
            if (!silent) setLoading(false);
        }
    }, [username]);

    // Initial fetch + reset card state whenever the URL :username changes.
    useEffect(() => {
        setProfile(null);
        setPage(1);
        setTab('rule');
        refetchProfile(false);
    }, [username, refetchProfile]);

    // Listen for any library-mutation event in the app (bookmark toggled,
    // draft published, etc.) and refresh both the profile header AND the
    // contributions list silently.
    useLibraryRefresh(useCallback(() => {
        if (!username) return;
        refetchProfile(true);
        // The contributions effect below already keys on `profile`,
        // so once setProfile fires from refetchProfile the list will
        // re-fetch automatically. But for cases where profile is
        // stable (e.g., only a rating changed, no counts moved), we
        // also force the contributions list to re-fetch by bumping
        // the page state... actually a simpler approach: bump a
        // refresh counter the contributions effect depends on.
        setContribRefreshTick((t) => t + 1);
    }, [username, refetchProfile]));

    // Re-fetch contributions whenever the profile or tab changes.
    // Uses the same /library/search endpoint Browse uses (with the
    // ?author=… filter from Phase 4), so we get the rich shape RuleCard
    // and CognitiveElementCard expect — active_ces membership plus the
    // rule's groups/condition, is_local_draft, public_id, categories, etc.
    useEffect(() => {
        if (!profile) return;
        let cancelled = false;
        setContribLoading(true);
        setExpandedRuleId(null);
        setExpandedCeId(null);
        searchLibrary({
            q: '',
            author: profile.username,
            asset_types: tab,                 // 'rule' or 'ce'
            page,
            page_size: pageSize,
        }).then((res) => {
            if (cancelled) return;
            const items = res.data?.results || [];
            // Backend returns search-shaped rows. Normalize for the
            // existing card components: rules want predicate +
            // active_ces; CEs want definition + examples.
            const mapped = items.map((it) => {
                if (it.asset_type === 'rule') {
                    return {
                        ...it,
                        rule_id: it.id,
                        setup_id: it.id,
                        custom_name: it.name,
                        predicate: it.content,
                        // active_ces is already in the right shape from
                        // _hydrate_results (post-fix).
                        active_ces: it.active_ces || (it.ces || []).map((n) => ({ name: n })),
                    };
                }
                return {
                    ...it,
                    ce_id: it.id,
                    definition: it.content,
                    examples: it.examples || [],
                };
            });
            setContribData({
                items: mapped,
                total: res.data?.total_results ?? mapped.length,
            });
        }).catch(() => {
            if (!cancelled) setContribData({ items: [], total: 0 });
        }).finally(() => {
            if (!cancelled) setContribLoading(false);
        });
        return () => { cancelled = true; };
    // Keyed on profile?.username (stable string) instead of `profile`
    // (object reference that changes on every re-fetch). This prevents
    // the contributions list from re-fetching when only the profile
    // re-render → expanded card collapses.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [profile?.username, tab, page, pageSize, contribRefreshTick]);

    // Seed bookmark state once we know who's viewing. Lets the cards
    // render the "Saved" badge for items the viewer has already
    // bookmarked elsewhere.
    useEffect(() => {
        if (!currentUser?.user_id) return;
        let cancelled = false;
        Promise.all([
            getRuleBookmarks(currentUser.user_id),
            getCEBookmarks(currentUser.user_id),
        ]).then(([rRes, cRes]) => {
            if (cancelled) return;
            const ruleIds = new Set(
                (rRes.data?.bookmarks || []).map((b) => b.rule_id || b.id).filter(Boolean),
            );
            const ceIds = new Set(
                (cRes.data?.bookmarks || []).map((b) => b.ce_id || b.id).filter(Boolean),
            );
            setRuleBookmarkIds(ruleIds);
            setCeBookmarkIds(ceIds);
        }).catch(() => { /* ignore — bookmarks are best-effort */ });
        return () => { cancelled = true; };
    }, [currentUser?.user_id]);

    const handleRuleBookmark = async (rule) => {
        if (!currentUser?.user_id) return;
        const id = rule.rule_id || rule.id;
        try {
            if (ruleBookmarkIds.has(id)) {
                await removeRuleBookmark(currentUser.user_id, id);
                setRuleBookmarkIds((p) => { const n = new Set(p); n.delete(id); return n; });
            } else {
                await addRuleBookmark(currentUser.user_id, id);
                setRuleBookmarkIds((p) => { const n = new Set(p); n.add(id); return n; });
            }
        } catch (err) {
            showAlertDialog({ title: 'Bookmark failed', message: err.message || 'Could not update bookmark.', variant: 'error' });
        }
    };

    const handleCeBookmark = async (ce) => {
        if (!currentUser?.user_id) return;
        const id = ce.ce_id || ce.id;
        try {
            if (ceBookmarkIds.has(id)) {
                await removeCEBookmark(currentUser.user_id, id);
                setCeBookmarkIds((p) => { const n = new Set(p); n.delete(id); return n; });
            } else {
                await addCEBookmark(currentUser.user_id, id);
                setCeBookmarkIds((p) => { const n = new Set(p); n.add(id); return n; });
            }
        } catch (err) {
            showAlertDialog({ title: 'Bookmark failed', message: err.message || 'Could not update bookmark.', variant: 'error' });
        }
    };

    const toggleCeExpand = async (ceId) => {
        const next = expandedCeId === ceId ? null : ceId;
        setExpandedCeId(next);
        if (next && !previewCache[ceId]) {
            try {
                const res = await getCognitiveDataset(ceId);
                const raw = res.data?.training_data_preview || res.data?.training_data || [];
                setPreviewCache((p) => ({ ...p, [ceId]: raw }));
                setPreviewTotals((p) => ({ ...p, [ceId]: res.data?.samples_count ?? null }));
            } catch {
                setPreviewCache((p) => ({ ...p, [ceId]: [] }));
            }
        }
    };

    const readonlyNotice = () => {
        showAlertDialog({
            title: 'Read-only on profile',
            message: 'Open the rule from Browse to edit it.',
            variant: 'info',
        });
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const res = await updateMyProfile({
                display_name: editDisplayName,
                bio: editBio,
            });
            // Reflect the server's canonical response.
            setProfile((p) => p && ({
                ...p,
                display_name: res.data.display_name,
                bio: res.data.bio,
            }));
            // Keep the localStorage user in sync so other parts of the
            // app see the new display name without a full reload.
            if (currentUser) {
                const updated = {
                    ...currentUser,
                    display_name: res.data.display_name,
                    bio: res.data.bio,
                };
                sessionStorage.setItem('user', JSON.stringify(updated));
            }
            setEditing(false);
        } catch (err) {
            showAlertDialog({
                title: 'Save failed',
                message: err.response?.data?.detail || err.message || 'Could not save profile.',
                variant: 'error',
            });
        } finally {
            setSaving(false);
        }
    };

    const handleCancel = () => {
        setEditDisplayName(profile?.display_name || '');
        setEditBio(profile?.bio || '');
        setEditing(false);
    };

    if (loading) {
        return (
            <Layout>
                <div style={{ textAlign: 'center', padding: '80px', color: '#94a3b8' }}>
                    Loading profile…
                </div>
            </Layout>
        );
    }

    if (notFound) {
        return (
            <Layout>
                <div style={{ textAlign: 'center', padding: '60px', color: '#cbd5e1' }}>
                    <h2 style={{ color: '#f1f5f9', marginBottom: '8px' }}>User not found</h2>
                    <p style={{ color: '#94a3b8' }}>
                        No user with the username <code style={{ color: '#fcd34d' }}>{username}</code>.
                    </p>
                    <div style={{ display: 'inline-flex', gap: '10px', marginTop: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
                        <button onClick={() => navigate('/community')} style={notFoundPrimaryBtn}>
                            <FiUsers /> Browse Community
                        </button>
                        <button
                            onClick={() => navigate('/workspace')}
                            style={{ ...backBtnStyle, justifyContent: 'center' }}
                        >
                            <FiArrowLeft /> Back to Workspace
                        </button>
                    </div>
                </div>
            </Layout>
        );
    }

    const displayName = profile.display_name || profile.username;

    return (
        <Layout>
            <Breadcrumb items={[
                { label: 'Hub', icon: FiHome, to: '/workspace' },
                { label: 'Community', icon: FiUsers, to: '/community' },
                { label: profile.username },
            ]} style={{ marginBottom: 16 }} />

            {/* Header card */}
            <div style={headerCardStyle}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '20px', flexWrap: 'wrap' }}>
                    <div style={avatarStyle}>
                        <FiUser size={36} />
                    </div>
                    <div style={{ flex: 1, minWidth: '240px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                            <h1 style={{ margin: 0, color: '#f8fafc', fontSize: '1.85rem', letterSpacing: '-0.02em' }}>
                                {displayName}
                            </h1>
                            {profile.is_team && <span style={teamBadgeStyle}><FiAward /> Team</span>}
                        </div>
                        <p style={{ margin: '4px 0 10px 0', color: '#94a3b8', fontSize: '1rem' }}>
                            {/* Public profile — show the handle, never the email
                              * (the API no longer returns another user's email). */}
                            @{profile.username}
                        </p>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', color: '#cbd5e1', fontSize: '0.9rem' }}>
                            <span><strong style={{ color: '#e2e8f0' }}>{profile.contribution_count_rules}</strong> rules</span>
                            <span><strong style={{ color: '#e2e8f0' }}>{profile.contribution_count_ces}</strong> CEs</span>
                        </div>
                        {profile.member_since && (
                            <p style={{ margin: '10px 0 0 0', color: '#64748b', fontSize: '0.85rem' }}>
                                Member since {new Date(profile.member_since).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                            </p>
                        )}
                    </div>
                    {isOwnProfile && !editing && (
                        <button onClick={() => setEditing(true)} style={editBtnStyle}>
                            <FiEdit2 /> Edit Profile
                        </button>
                    )}
                </div>

                {/* Bio (read or edit mode) */}
                <div style={{ marginTop: '20px', borderTop: '1px solid rgba(148, 163, 184, 0.14)', paddingTop: '16px' }}>
                    {editing ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <label style={editLabelStyle}>
                                Display name
                                <input
                                    type="text"
                                    value={editDisplayName}
                                    onChange={(e) => setEditDisplayName(e.target.value)}
                                    maxLength={255}
                                    style={editInputStyle}
                                    placeholder={profile.username}
                                />
                            </label>
                            <label style={editLabelStyle}>
                                Bio
                                <textarea
                                    value={editBio}
                                    onChange={(e) => setEditBio(e.target.value)}
                                    maxLength={2000}
                                    rows={4}
                                    style={{ ...editInputStyle, resize: 'vertical' }}
                                    placeholder="A few sentences about you and what you build."
                                />
                            </label>
                            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                                <button onClick={handleCancel} disabled={saving} style={cancelBtnStyle}>
                                    <FiX /> Cancel
                                </button>
                                <button onClick={handleSave} disabled={saving} style={saveBtnStyle}>
                                    <FiSave /> {saving ? 'Saving…' : 'Save'}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <p style={{
                            margin: 0,
                            color: profile.bio ? '#cbd5e1' : '#64748b',
                            lineHeight: 1.6,
                            whiteSpace: 'pre-wrap',
                            fontStyle: profile.bio ? 'normal' : 'italic',
                        }}>
                            {profile.bio || (isOwnProfile ? 'Add a bio to tell others what you build.' : 'This user hasn\'t added a bio yet.')}
                        </p>
                    )}
                </div>
            </div>

            {/* Tab bar */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '24px', marginBottom: '12px' }}>
                <button onClick={() => { setTab('rule'); setPage(1); }} style={tabBtnStyle(tab === 'rule')}>
                    Rules ({profile.contribution_count_rules})
                </button>
                <button onClick={() => { setTab('ce'); setPage(1); }} style={tabBtnStyle(tab === 'ce')}>
                    CEs ({profile.contribution_count_ces})
                </button>
            </div>

            {/* Contributions list */}
            {contribLoading ? (
                <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>Loading…</div>
            ) : contribData.items.length === 0 ? (
                <div style={emptyStateStyle}>
                    {isOwnProfile
                        ? `You haven't published any ${tab === 'rule' ? 'rules' : 'CEs'} yet. Publish one from Drafts to see it here.`
                        : `${displayName} hasn't published any ${tab === 'rule' ? 'rules' : 'CEs'}.`}
                </div>
            ) : (
                <>
                    {/* Render contributions using the same RuleCard /
                      * CognitiveElementCard components Browse uses, so
                      * the profile view looks identical to the
                      * canonical card (role badges, ratings, examples
                      * on expand, "by @user" link, save button). Cards
                      * are read-only on the profile page — edit /
                      * delete actions don't make sense here. */}
                    <div className="rules-list">
                        {tab === 'rule'
                            ? contribData.items.map((rule) => (
                                <RuleCard
                                    key={`profile-rule-${rule.rule_id}`}
                                    rule={rule}
                                    isExpanded={expandedRuleId === rule.rule_id}
                                    onToggle={() => setExpandedRuleId(
                                        expandedRuleId === rule.rule_id ? null : rule.rule_id,
                                    )}
                                    onDelete={readonlyNotice}
                                    onRemoveCE={readonlyNotice}
                                    onAddCE={readonlyNotice}
                                    readOnly
                                    onBookmark={currentUser ? handleRuleBookmark : undefined}
                                    bookmarkLabel="Save"
                                    isBookmarked={ruleBookmarkIds.has(rule.rule_id)}
                                />
                            ))
                            : contribData.items.map((ce) => (
                                <CognitiveElementCard
                                    key={`profile-ce-${ce.ce_id}`}
                                    ce={ce}
                                    isOpen={expandedCeId === ce.ce_id}
                                    onToggle={toggleCeExpand}
                                    samples={previewCache[ce.ce_id]}
                                        samplesTotal={previewTotals[ce.ce_id]}
                                    onBookmark={currentUser ? handleCeBookmark : undefined}
                                    isBookmarked={ceBookmarkIds.has(ce.ce_id)}
                                />
                            ))}
                    </div>
                    <Pagination
                        currentPage={page}
                        totalItems={contribData.total}
                        pageSize={pageSize}
                        onPageChange={setPage}
                    />
                </>
            )}
        </Layout>
    );
};

// --- Style fragments. Inline so the page is self-contained; if these
// repeat anywhere else we can hoist them into a shared CSS later. ---

const backBtnStyle = {
    background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer',
    display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500, padding: 0,
};

// Body CTA on the "user not found" view — matches the page's gradient accent
// (same indigo→blue as the Save / tab pills) so it reads as a clear primary
// way-forward rather than a new design.
const notFoundPrimaryBtn = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '10px 18px',
    borderRadius: '10px',
    border: 'none',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    cursor: 'pointer',
    fontWeight: 700,
    boxShadow: '0 6px 18px -2px rgba(99, 102, 241, 0.55)',
};

const headerCardStyle = {
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.55) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '16px',
    padding: '24px',
    backdropFilter: 'blur(14px)',
    boxShadow: '0 8px 24px -8px rgba(2, 6, 23, 0.50), 0 4px 12px rgba(99, 102, 241, 0.12)',
};

const avatarStyle = {
    width: '78px', height: '78px', borderRadius: '20px',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    boxShadow: '0 6px 18px -2px rgba(99, 102, 241, 0.55)',
    flexShrink: 0,
};

const teamBadgeStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '5px',
    padding: '4px 12px',
    borderRadius: '999px',
    background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(5, 150, 105, 0.22) 100%)',
    color: '#6ee7b7',
    border: '1px solid rgba(52, 211, 153, 0.45)',
    fontSize: '0.78rem',
    fontWeight: 700,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
};

const editBtnStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '8px 14px',
    borderRadius: '10px',
    border: '1px solid rgba(129, 140, 248, 0.42)',
    background: 'rgba(99, 102, 241, 0.18)',
    color: '#c7d2fe',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '0.88rem',
    transition: 'background 180ms ease, color 180ms ease',
};

const editLabelStyle = {
    display: 'flex', flexDirection: 'column', gap: '6px',
    color: '#cbd5e1', fontSize: '0.85rem', fontWeight: 600,
};

const editInputStyle = {
    padding: '10px 12px',
    borderRadius: '10px',
    border: '1.5px solid rgba(148, 163, 184, 0.22)',
    background: 'rgba(2, 6, 23, 0.55)',
    color: '#f1f5f9',
    fontSize: '0.95rem',
    fontFamily: 'inherit',
    outline: 'none',
};

const saveBtnStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '10px 18px',
    borderRadius: '10px',
    border: 'none',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    cursor: 'pointer',
    fontWeight: 700,
    boxShadow: '0 6px 18px -2px rgba(99, 102, 241, 0.55)',
};

const cancelBtnStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '10px 16px',
    borderRadius: '10px',
    border: '1px solid rgba(148, 163, 184, 0.22)',
    background: 'rgba(15, 23, 42, 0.55)',
    color: '#cbd5e1',
    cursor: 'pointer',
    fontWeight: 600,
};

const tabBtnStyle = (active) => ({
    padding: '8px 18px',
    borderRadius: '999px',
    border: active ? '1px solid transparent' : '1px solid rgba(148, 163, 184, 0.18)',
    background: active
        ? 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)'
        : 'rgba(15, 23, 42, 0.55)',
    color: active ? '#ffffff' : '#cbd5e1',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '0.9rem',
    boxShadow: active
        ? '0 6px 18px -2px rgba(99, 102, 241, 0.55)'
        : '0 2px 6px rgba(2, 6, 23, 0.30)',
    backdropFilter: active ? 'none' : 'blur(8px)',
    transition: 'transform 120ms ease, box-shadow 180ms ease',
});

const emptyStateStyle = {
    padding: '40px 20px',
    textAlign: 'center',
    color: '#94a3b8',
    background: 'rgba(15, 23, 42, 0.40)',
    border: '1px dashed rgba(148, 163, 184, 0.20)',
    borderRadius: '12px',
};

export default Profile;
