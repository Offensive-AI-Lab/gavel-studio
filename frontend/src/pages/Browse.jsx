import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import CommunityTabs from '../components/CommunityTabs/CommunityTabs';
import RuleCard from '../components/RuleCard/RuleCard';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import Pagination from '../components/Pagination/Pagination';
import { getPublicRules, addRuleBookmark, getRuleBookmarks, removeRuleBookmark, getAllCategories } from '../api';
import useLibrarySearch from '../hooks/useLibrarySearch';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import '../css/RulesManager.css';
import { normalizeCategoryValue } from '../utils/categoryUtils';

const Browse = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const [rules, setRules] = useState([]);
    const [expandedRule, setExpandedRule] = useState(null);
    const [bookmarkIds, setBookmarkIds] = useState(new Set());
    const [bookmarks, setBookmarks] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [searchTopK, setSearchTopK] = useState(10);
    const [page, setPage] = useState(1);
    const [availableCategories, setAvailableCategories] = useState([]);
    const [searchReloadKey, setSearchReloadKey] = useState(0);
    const user = JSON.parse(sessionStorage.getItem('user'));

    // Phase 4: ?author=<username> in the URL filters the listing to one
    // contributor's work. Clicking "Browse by author" on a profile card
    // sets this; clearing the chip below removes it.
    const authorFilter = (searchParams.get('author') || '').trim().toLowerCase() || null;
    const clearAuthorFilter = () => {
        const next = new URLSearchParams(searchParams);
        next.delete('author');
        setSearchParams(next, { replace: true });
    };

    // Stable identity for the assetTypes prop so the hook's deps don't churn.
    const assetTypes = useMemo(() => ['rule'], []);

    // Single shared hook drives live search for both Browse pages — the only
    // difference here is the assetTypes filter. See useLibrarySearch.js.
    const {
        results: rawSearchResults,
        totalResults: searchResultCount,
        loading: searchLoading,
        error: searchError,
        hasSearched,
    } = useLibrarySearch({
        query: searchQuery,
        categories: searchCategories,
        page,
        pageSize: searchTopK,
        assetTypes,
        author: authorFilter,
        reloadKey: searchReloadKey,
    });

    useEffect(() => {
        fetchRules();
        fetchCategories();
        fetchBookmarks();
    }, [navigate]);

    const normalizeRule = (rule) => {
        const ceNames = Array.isArray(rule.required_ces) ? rule.required_ces : [];
        const activeCes = Array.isArray(rule.active_ces) && rule.active_ces.length > 0
            ? rule.active_ces
            : ceNames.map((name) => ({ name }));

        return {
            ...rule,
            setup_id: rule.setup_id || rule.id || rule.rule_id || Math.random().toString(36).slice(2),
            custom_name: rule.custom_name || rule.name || rule.title || 'Rule',
            predicate: rule.predicate || rule.logic || 'No predicate available',
            active_ces: activeCes,
        };
    };

    const mapSearchResultToRule = (item) => {
        // Prefer the membership list from the backend (`active_ces`,
        // { ce_id, name } per member). The rule's LOGIC — groups +
        // condition — rides along on the row itself (spread below) and
        // RuleCard renders it via extractLogic; `ces` (names only) is
        // the fallback for older response shapes.
        const richActive = Array.isArray(item.active_ces) && item.active_ces.length > 0
            ? item.active_ces
            : null;
        const ceList = richActive
            || (Array.isArray(item.ces) ? item.ces : item.categories);
        const normalized = normalizeRule({
            ...item,
            setup_id: item.setup_id || item.id || item.rule_id,
            rule_id: item.id || item.rule_id,
            custom_name: item.name || item.custom_name,
            predicate: item.logic || item.predicate || item.content || item.description,
            required_ces: Array.isArray(ceList)
                ? ceList.map((ce) => (typeof ce === 'string' ? ce : ce.name))
                : ceList,
            active_ces: Array.isArray(ceList)
                ? ceList.map((ce) => (typeof ce === 'string' ? { name: ce } : ce))
                : item.active_ces,
        });
        return normalized;
    };

    // Hook returns raw API rows; map to RuleCard's expected shape, asset_type-filtered.
    // mapSearchResultToRule is a stable closure over only its argument — re-running
    // when it re-defines wouldn't change behavior, so excluding it from deps is safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    const searchResults = useMemo(
        () => (rawSearchResults || [])
            .filter((item) => (item.asset_type || item.type) === 'rule')
            .map(mapSearchResultToRule),
        [rawSearchResults],
    );


    const fetchRules = async () => {
        try {
            // Community/Browse is the PUBLIC space: it shows only synced
            // library rules. A user's local drafts never appear here — they
            // live in "Your Library" (drafts join the shared library via
            // gavel-rules pull requests).
            const publicRes = await getPublicRules();
            const publicData = publicRes.data.rules || publicRes.data || [];
            const publicRules = (Array.isArray(publicData) ? publicData : [])
                .map(normalizeRule)
                .map(r => ({ ...r, is_local_draft: r.is_local_draft ?? false }))
                .filter(r => !r.is_local_draft);   // belt-and-suspenders: never show drafts publicly
            setRules(publicRules);
        } catch {
            setRules([]);
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

    // Reset to page 1 whenever the user changes their filters — otherwise a
    // typed query on page 3 would search page 3 of the new query and look empty.
    useEffect(() => {
        setPage(1);
    }, [searchQuery, searchCategories, searchTopK]);

    // Stay in sync with library mutations from anywhere in the app —
    // a new public rule arriving via gavel-rules sync, a bookmark
    // toggle in another tab, etc. — all flow through
    // gavel:libraryChanged.
    useLibraryRefresh(() => {
        fetchRules();
        fetchBookmarks();
    });

    const pageHelp = {
        title: 'Community · Rules',
        summary: 'Curated rules published by other users. Bookmark the ones you want to reuse — they\'ll show up in My Rule Bookmarks and become available when adding rules to a rule set.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    rules.length === 0
                        ? ['Library is empty or still syncing. New content is pushed to you automatically the moment it is published.']
                        : [
                            `${rules.length} rule${rules.length === 1 ? '' : 's'} available. Use the search bar + category filters to narrow down.`,
                            'Click a rule to expand its details. Click the bookmark icon to save it.',
                            'Categories are clickable chips — click them to filter to that category.',
                        ],
            },
            {
                heading: 'Notes',
                bullets: [
                    'Click "Browse CEs" in the search panel to switch to the cognitive-elements view.',
                    'Bookmarks are per-user and persist across sessions. Remove via the un-bookmark icon or My Rule Bookmarks.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    const fetchBookmarks = async () => {
        if (!user?.user_id) return;
        try {
            const res = await getRuleBookmarks(user.user_id);
            const list = res.data?.bookmarks || [];
            setBookmarks(list);
            const ids = new Set(list.map((b) => b.rule_id));
            setBookmarkIds(ids);
        } catch {
            setBookmarkIds(new Set());
            setBookmarks([]);
        }
    };

    const handleBookmark = async (rule) => {
        if (!user?.user_id) return readonlyNotice();
        const ruleId = rule.rule_id || rule.id;
        if (!ruleId) return showAlertDialog({ title: 'Missing ID', message: 'This rule cannot be bookmarked because it lacks an id.', variant: 'warning' });
        try {
            if (bookmarkIds.has(ruleId)) {
                await removeRuleBookmark(user.user_id, ruleId);
                setBookmarkIds((prev) => {
                    const next = new Set(prev);
                    next.delete(ruleId);
                    return next;
                });
                setBookmarks((prev) => prev.filter((b) => b.rule_id !== ruleId));
                showAlertDialog({ title: 'Removed', message: 'Rule removed from your bookmarks.', variant: 'success' });
            } else {
                await addRuleBookmark(user.user_id, ruleId);
                setBookmarkIds((prev) => {
                    const next = new Set(prev);
                    next.add(ruleId);
                    return next;
                });
                setBookmarks((prev) => [{ rule_id: ruleId, name: rule.custom_name || rule.name }, ...prev]);
                showAlertDialog({ title: 'Saved', message: 'Rule added to your bookmarks.', variant: 'success' });
            }
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not bookmark this rule.', variant: 'error' });
        }
    };

    const readonlyNotice = () => showAlertDialog({ title: 'Read-only', message: 'Public rules cannot be modified here.', variant: 'info' });

    // Filter rules locally based on search categories when not performing a text search
    const filteredRules = rules.filter((rule) => {
        if (searchCategories.length === 0) return true;
        
        const ruleCats = new Set();
        if (Array.isArray(rule.categories)) {
            rule.categories.forEach(c => {
                const n = normalizeCategoryValue(c);
                if (n) ruleCats.add(n);
            });
        }
        [rule.category, rule.primary_category].forEach(c => {
            const n = normalizeCategoryValue(c);
            if (n) ruleCats.add(n);
        });

        // Filter: Show rule if it matches ANY of the selected categories
        return searchCategories.some(cat => ruleCats.has(cat));
    });

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <CommunityTabs active="rules" />
                    <h1>Public Rules</h1>
                    <p>Review shared rules. Open any rule to inspect its logic and cognitive elements.</p>
                </div>
            </header>

            {/* Author filter chip — visible only when ?author=… is in
              * the URL. Clicking the × clears the filter and lands back
              * on the unfiltered Browse view. */}
            {authorFilter && (
                <div style={authorChipBarStyle}>
                    <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Filtered to author:</span>
                    <span style={authorChipStyle}>
                        @{authorFilter}
                        <button
                            onClick={clearAuthorFilter}
                            aria-label="Clear author filter"
                            style={authorChipCloseStyle}
                        >×</button>
                    </span>
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: '20px', alignItems: 'flex-start', width: '100%' }}>
                <div style={{ minWidth: 0 }}>
                    <SearchPanel
                        query={searchQuery}
                        onQueryChange={setSearchQuery}
                        categories={searchCategories}
                        onCategoriesChange={setSearchCategories}
                        topK={searchTopK}
                        onTopKChange={setSearchTopK}
                        // Live search runs via useLibrarySearch — the button just
                        // forces an immediate refresh, which is already what the
                        // current debounced state will produce.
                        onSearch={() => {}}
                        onReset={() => {
                            setSearchQuery('');
                            setSearchCategories([]);
                            setPage(1);
                        }}
                        loading={searchLoading}
                        assetTypes={assetTypes}
                        showAssetTypeFilter={false}
                        searchPlaceholder="Search public rules..."
                        availableCategories={availableCategories}
                    />

                    {searchError && (
                        <div style={{
                            background: 'rgba(239, 68, 68, 0.18)',
                            border: '1px solid #fecaca',
                            color: '#991b1b',
                            padding: '12px',
                            borderRadius: '8px',
                            marginBottom: '16px',
                            fontSize: '14px'
                        }}>
                            <div style={{ marginBottom: '10px' }}>{searchError}</div>
                            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                                <button className="primary-btn" onClick={() => setSearchReloadKey((k) => k + 1)}>Try again</button>
                                <button className="primary-btn" onClick={() => navigate('/workspace')}>Go to Hub</button>
                            </div>
                        </div>
                    )}

                    {searchResults.length > 0 && !searchLoading && (
                        <div style={{ marginBottom: '16px' }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px',
                                marginBottom: '12px',
                                paddingBottom: '12px',
                                borderBottom: '2px solid rgba(148, 163, 184, 0.18)'
                            }}>
                                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                    Search Results ({searchResultCount} found)
                                </h2>
                            </div>
                            <div className="rules-list">
                                {searchResults.map((rule) => (
                                    <RuleCard
                                        key={`search-${rule.setup_id}`}
                                        rule={rule}
                                        isExpanded={expandedRule === rule.setup_id}
                                        onToggle={() => setExpandedRule(expandedRule === rule.setup_id ? null : rule.setup_id)}
                                        onDelete={readonlyNotice}
                                        onRemoveCE={readonlyNotice}
                                        onAddCE={readonlyNotice}
                                        readOnly
                                        onBookmark={handleBookmark}
                                        bookmarkLabel="Save"
                                        isBookmarked={bookmarkIds.has(rule.rule_id || rule.id)}
                                    />
                                ))}
                            </div>

                            {!searchLoading && searchResultCount > 0 && (
                                <Pagination 
                                    currentPage={page}
                                    totalItems={searchResultCount}
                                    pageSize={searchTopK}
                                    onPageChange={setPage}
                                />
                            )}
                        </div>
                    )}

                    {searchLoading && (

                        <div style={{
                            textAlign: 'center',
                            padding: '60px 20px',
                            color: '#94a3b8'
                        }}>
                            <div style={{
                                display: 'inline-block',
                                width: '40px',
                                height: '40px',
                                border: '4px solid #e5e7eb',
                                borderTopColor: '#3b82f6',
                                borderRadius: '50%',
                                animation: 'spin 0.8s linear infinite',
                                marginBottom: '12px'
                            }}></div>
                            <p>Searching...</p>
                        </div>
                    )}

                    {!searchLoading && searchResults.length === 0 && hasSearched && (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px 20px',
                            color: '#94a3b8'
                        }}>
                            <p>
                                {(() => {
                                    const q = searchQuery.trim();
                                    const cats = searchCategories;
                                    if (q && cats.length > 0) {
                                        return `No results found for "${q}" in ${cats.join(', ')}.`;
                                    }
                                    if (q) {
                                        return `No results found for "${q}".`;
                                    }
                                    if (cats.length > 0) {
                                        return cats.length === 1
                                            ? `No rules in the ${cats[0]} category.`
                                            : `No rules in the selected categories: ${cats.join(', ')}.`;
                                    }
                                    return 'No rules match your filters.';
                                })()}
                            </p>
                        </div>
                    )}

                    {!searchLoading && rules.length === 0 && !hasSearched && (
                        <div className="empty-state">
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>No Public Rules</h2>
                            <p style={{marginBottom: '20px', color: '#94a3b8'}}>Start by searching for rules or creating a new one.</p>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                                <button className="primary-btn" onClick={() => navigate('/bookmarks/rules')}>Create a Rule</button>
                                <button className="primary-btn" onClick={() => navigate('/workspace')}>Go to Hub</button>
                            </div>
                        </div>
                    )}

                    {!searchLoading && rules.length > 0 && filteredRules.length === 0 && !hasSearched && (
                         <div style={{
                            textAlign: 'center',
                            padding: '40px 20px',
                            color: '#94a3b8'
                         }}>
                            <p>No rules match the selected categories.</p>
                         </div>
                    )}

                    {!hasSearched && !searchLoading && filteredRules.length > 0 && (
                        <div>
                            <div style={{
                                marginBottom: '12px',
                                paddingBottom: '12px',
                                borderBottom: '2px solid rgba(148, 163, 184, 0.18)'
                            }}>
                                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                    All Public Rules
                                </h2>
                            </div>
                            <div className="rules-list">
                                {filteredRules.slice((page - 1) * 10, page * 10).map((rule) => (
                                    <RuleCard
                                        key={rule.setup_id}
                                        rule={rule}
                                        isExpanded={expandedRule === rule.setup_id}
                                        onToggle={() => setExpandedRule(expandedRule === rule.setup_id ? null : rule.setup_id)}
                                        onDelete={readonlyNotice}
                                        onRemoveCE={readonlyNotice}
                                        onAddCE={readonlyNotice}
                                        readOnly
                                        onBookmark={handleBookmark}
                                        bookmarkLabel="Save"
                                        isBookmarked={bookmarkIds.has(rule.rule_id || rule.id)}
                                    />
                                ))}
                            </div>

                            <Pagination
                                currentPage={page}
                                totalItems={filteredRules.length}
                                pageSize={searchTopK}
                                onPageChange={setPage}
                            />
                        </div>
                    )}
                </div>
                <aside style={{
                    width: '100%',
                    // Dark glass panel matching SearchPanel + the rest of the
                    // dashboard chrome. Slightly lighter than the bookmark
                    // tiles inside so they read as nested.
                    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.72) 0%, rgba(15, 23, 42, 0.62) 100%)',
                    border: '1px solid rgba(148, 163, 184, 0.18)',
                    borderRadius: '14px',
                    padding: '18px',
                    position: 'sticky',
                    top: '20px',
                    alignSelf: 'flex-start',
                    backdropFilter: 'blur(14px)',
                    WebkitBackdropFilter: 'blur(14px)',
                    boxShadow: '0 8px 24px -8px rgba(2, 6, 23, 0.50), 0 4px 12px rgba(99, 102, 241, 0.12)',
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                        <h3 style={{ margin: 0, fontSize: '0.95rem', color: '#f1f5f9', fontWeight: 700, letterSpacing: '-0.01em' }}>My Bookmarked Rules</h3>
                        <button
                            style={{ background: 'none', border: 'none', color: '#a5b4fc', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, padding: 0 }}
                            onClick={() => navigate('/bookmarks/rules')}
                        >
                            View all
                        </button>
                    </div>
                    {bookmarks.length === 0 ? (
                        <p style={{ color: '#94a3b8', margin: 0, fontSize: '0.88rem' }}>No bookmarks yet.</p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {bookmarks.map((b) => (
                                <div key={b.rule_id} style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
                                    padding: '10px 12px',
                                    border: '1px solid rgba(148, 163, 184, 0.14)',
                                    borderRadius: '10px',
                                    background: 'rgba(2, 6, 23, 0.55)',
                                    width: '100%',
                                    boxSizing: 'border-box',
                                    transition: 'border-color 180ms ease, box-shadow 180ms ease',
                                }}>
                                    <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.88rem', wordBreak: 'break-word', flex: 1 }}>{b.name || 'Rule'}</span>
                                    <button
                                        className="bookmark-btn"
                                        onClick={() => handleBookmark({ rule_id: b.rule_id })}
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

const backBtnStyle = {
    background: 'none', border: 'none', color: '#64748b', cursor: 'pointer',
    display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500,
    transition: 'color 150ms ease',
};
// Tab pill: solid gradient + glow when active, soft hover when not.
// The active pill matches the sidebar's accent gradient (indigo→blue) so
// the navigation chrome reads as one coherent system.
// Phase 4 author-filter chip styles (consumed by the bar at the top
// of the Browse list when ?author=… is set).
const authorChipBarStyle = {
    display: 'flex', alignItems: 'center', gap: '10px',
    padding: '10px 14px', marginBottom: '14px',
    background: 'rgba(15, 23, 42, 0.55)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: '12px',
};
const authorChipStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '4px 6px 4px 12px',
    background: 'rgba(99, 102, 241, 0.18)',
    border: '1px solid rgba(129, 140, 248, 0.45)',
    borderRadius: '999px',
    color: '#c7d2fe',
    fontWeight: 600,
    fontSize: '0.9rem',
};
const authorChipCloseStyle = {
    width: '22px', height: '22px',
    borderRadius: '50%',
    border: 'none',
    background: 'rgba(2, 6, 23, 0.55)',
    color: '#cbd5e1',
    cursor: 'pointer',
    fontSize: '1rem',
    lineHeight: 1,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
};

const pillStyle = (active) => ({
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

export default Browse;
