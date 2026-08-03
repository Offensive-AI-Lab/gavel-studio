import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import CommunityTabs from '../components/CommunityTabs/CommunityTabs';
import RuleSetCard from '../components/RuleSetCard/RuleSetCard';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import Pagination from '../components/Pagination/Pagination';
import {
    getPublicRuleSets,
    getRuleSetBookmarks,
    addRuleSetBookmark,
    removeRuleSetBookmark,
    getAllCategories,
} from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import { normalizeCategoryValue } from '../utils/categoryUtils';
import '../css/RulesManager.css';

const PAGE_SIZE = 10;

const BrowseRuleSets = () => {
    const navigate = useNavigate();
    const [ruleSets, setRuleSets] = useState([]);
    // Starts TRUE so the first paint is a loading state, not "No public
    // rule sets" — an unresolved fetch is not an empty result (#11).
    const [loading, setLoading] = useState(true);
    const [expandedId, setExpandedId] = useState(null);
    const [bookmarkIds, setBookmarkIds] = useState(new Set());
    const [bookmarks, setBookmarks] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [page, setPage] = useState(1);
    const [availableCategories, setAvailableCategories] = useState([]);
    const user = JSON.parse(sessionStorage.getItem('user'));

    useEffect(() => {
        fetchRuleSets();
        fetchCategories();
        fetchBookmarks();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [navigate]);

    // Reset to page 1 when filters change.
    useEffect(() => { setPage(1); }, [searchQuery, searchCategories]);

    // Stay in sync with library mutations (publishes, syncs, bookmark toggles).
    useLibraryRefresh(() => {
        fetchRuleSets();
        fetchBookmarks();
    });

    const fetchRuleSets = async () => {
        try {
            const res = await getPublicRuleSets();
            const data = res.data?.rule_sets || [];
            // Public space: never show drafts (belt-and-suspenders; the backend
            // already filters is_local_draft = FALSE).
            setRuleSets((Array.isArray(data) ? data : [])
                .map((rs) => ({ ...rs, is_local_draft: rs.is_local_draft ?? false }))
                .filter((rs) => !rs.is_local_draft));
        } catch {
            setRuleSets([]);
        } finally {
            setLoading(false);
        }
    };

    const fetchCategories = async () => {
        try {
            const res = await getAllCategories();
            setAvailableCategories(res.data || []);
        } catch {
            setAvailableCategories([]);
        }
    };

    const fetchBookmarks = async () => {
        if (!user?.user_id) return;
        try {
            const res = await getRuleSetBookmarks(user.user_id);
            const list = res.data?.bookmarks || [];
            setBookmarks(list);
            setBookmarkIds(new Set(list.map((b) => b.rule_set_id)));
        } catch {
            setBookmarks([]);
            setBookmarkIds(new Set());
        }
    };

    const handleBookmark = async (ruleSet) => {
        if (!user?.user_id) {
            return showAlertDialog({ title: 'Sign in', message: 'Sign in to bookmark rule sets.', variant: 'info' });
        }
        const id = ruleSet.rule_set_id;
        if (!id) {
            return showAlertDialog({ title: 'Missing ID', message: 'This rule set cannot be bookmarked.', variant: 'warning' });
        }
        try {
            if (bookmarkIds.has(id)) {
                await removeRuleSetBookmark(user.user_id, id);
                setBookmarkIds((prev) => { const n = new Set(prev); n.delete(id); return n; });
                setBookmarks((prev) => prev.filter((b) => b.rule_set_id !== id));
                showAlertDialog({ title: 'Removed', message: 'Rule set removed from your bookmarks.', variant: 'success' });
            } else {
                await addRuleSetBookmark(user.user_id, id);
                setBookmarkIds((prev) => { const n = new Set(prev); n.add(id); return n; });
                setBookmarks((prev) => [{ rule_set_id: id, name: ruleSet.name }, ...prev]);
                showAlertDialog({ title: 'Saved', message: 'Rule set added to your bookmarks.', variant: 'success' });
            }
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not bookmark this rule set.', variant: 'error' });
        }
    };

    useTutorialContent({
        title: 'Community · Rule Sets',
        summary: 'Rule sets are model-agnostic collections of published rules shared by others. Bookmark the ones you like — then fork a bookmarked set into your own workspace from the Rule Sets page.',
        sections: [
            {
                heading: 'Right now',
                bullets: ruleSets.length === 0
                    ? ['No public rule sets yet, or still syncing. Rule sets arrive with library sync from the gavel-rules repository.']
                    : [
                        `${ruleSets.length} rule set${ruleSets.length === 1 ? '' : 's'} available. Search by name or filter by category.`,
                        'Open a card to see its member rules. Bookmark a set, then fork it from your Rule Sets page.',
                    ],
            },
        ],
    });

    // Client-side search + category filter (rule sets are a small set; no
    // server-side hybrid search is wired for this asset type in v1).
    const q = searchQuery.trim().toLowerCase();
    const filtered = ruleSets.filter((rs) => {
        if (q) {
            const hay = `${rs.name || ''} ${rs.description || ''}`.toLowerCase();
            if (!hay.includes(q)) return false;
        }
        if (searchCategories.length > 0) {
            const cats = new Set((rs.categories || []).map(normalizeCategoryValue).filter(Boolean));
            if (!searchCategories.some((c) => cats.has(c))) return false;
        }
        return true;
    });
    const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <CommunityTabs active="rule-sets" />
                    <h1>Public Rule Sets</h1>
                    <p>Browse shared rule sets. Fork one into your workspace, or bookmark it for later.</p>
                </div>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: '20px', alignItems: 'flex-start', width: '100%' }}>
                <div style={{ minWidth: 0 }}>
                    <SearchPanel
                        query={searchQuery}
                        onQueryChange={setSearchQuery}
                        categories={searchCategories}
                        onCategoriesChange={setSearchCategories}
                        onSearch={() => {}}
                        onReset={() => { setSearchQuery(''); setSearchCategories([]); setPage(1); }}
                        onTopKChange={() => {}}
                        showAssetTypeFilter={false}
                        searchPlaceholder="Search public rule sets..."
                        availableCategories={availableCategories}
                    />

                    {loading ? (
                        <div className="empty-state" style={{ textAlign: 'center', padding: '40px 20px', color: '#94a3b8' }} role="status">
                            Loading rule sets…
                        </div>
                    ) : filtered.length === 0 ? (
                        <div className="empty-state" style={{ textAlign: 'center', padding: '40px 20px', color: '#94a3b8' }}>
                            <h2 style={{ fontSize: '1.4rem', marginBottom: '10px', color: '#cbd5e1' }}>No public rule sets</h2>
                            <p style={{ marginBottom: '20px' }}>
                                {ruleSets.length === 0
                                    ? 'Nothing shared yet. Public rule sets arrive with library sync from the gavel-rules repository.'
                                    : 'No rule sets match your filters.'}
                            </p>
                            <button className="primary-btn" onClick={() => navigate('/guardrails')}>Go to my Rule Sets</button>
                        </div>
                    ) : (
                        <div>
                            <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '2px solid rgba(148, 163, 184, 0.18)' }}>
                                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                    All Public Rule Sets ({filtered.length})
                                </h2>
                            </div>
                            <div className="rules-list">
                                {paged.map((rs) => (
                                    <RuleSetCard
                                        key={rs.rule_set_id}
                                        ruleSet={rs}
                                        isExpanded={expandedId === rs.rule_set_id}
                                        onToggle={() => setExpandedId(expandedId === rs.rule_set_id ? null : rs.rule_set_id)}
                                        onBookmark={handleBookmark}
                                        bookmarkLabel="Save"
                                        isBookmarked={bookmarkIds.has(rs.rule_set_id)}
                                    />
                                ))}
                            </div>
                            <Pagination
                                currentPage={page}
                                totalItems={filtered.length}
                                pageSize={PAGE_SIZE}
                                onPageChange={setPage}
                            />
                        </div>
                    )}
                </div>

                <aside style={asideStyle}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                        <h3 style={{ margin: 0, fontSize: '0.95rem', color: '#f1f5f9', fontWeight: 700 }}>My Bookmarked Rule Sets</h3>
                    </div>
                    {bookmarks.length === 0 ? (
                        <p style={{ color: '#94a3b8', margin: 0, fontSize: '0.88rem' }}>No bookmarks yet.</p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {bookmarks.map((b) => (
                                <div key={b.rule_set_id} style={bookmarkRowStyle}>
                                    <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.88rem', wordBreak: 'break-word', flex: 1 }}>{b.name || 'Rule set'}</span>
                                    <button
                                        className="bookmark-btn"
                                        onClick={() => handleBookmark({ rule_set_id: b.rule_set_id, name: b.name })}
                                        aria-label="Remove bookmark"
                                        style={{ flexShrink: 0 }}
                                    >
                                        Remove
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </aside>
            </div>
        </Layout>
    );
};

const asideStyle = {
    width: '100%',
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.72) 0%, rgba(15, 23, 42, 0.62) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '14px',
    padding: '18px',
    position: 'sticky',
    top: '20px',
    alignSelf: 'flex-start',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    boxShadow: '0 8px 24px -8px rgba(2, 6, 23, 0.50)',
};
const bookmarkRowStyle = {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
    padding: '10px 12px',
    border: '1px solid rgba(148, 163, 184, 0.14)',
    borderRadius: '10px',
    background: 'rgba(2, 6, 23, 0.55)',
    width: '100%',
    boxSizing: 'border-box',
};

export default BrowseRuleSets;
