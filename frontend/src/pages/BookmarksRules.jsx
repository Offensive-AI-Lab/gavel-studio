import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import RuleCard from '../components/RuleCard/RuleCard';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import Pagination from '../components/Pagination/Pagination';
import { getRuleBookmarks, removeRuleBookmark, getPublicRules, searchBookmarks, listLocalDrafts, deleteDraftRule } from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import Swal from 'sweetalert2';
import { showAlertDialog, showConfirmDialog } from '../components/ConfirmDialog/confirmDialog';
import { FiArrowLeft, FiInbox } from 'react-icons/fi';
import '../css/RulesManager.css';

const BookmarksRules = ({ embedded = false, mineOnly = false }) => {
    const navigate = useNavigate();
    const [bookmarks, setBookmarks] = useState([]);
    const [filteredBookmarks, setFilteredBookmarks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [availableCategories, setAvailableCategories] = useState([]);
    const [topK, setTopK] = useState(10);
    const [page, setPage] = useState(1);
    const [totalResults, setTotalResults] = useState(0);
    const [expandedRule, setExpandedRule] = useState(null);
    const [hasSearched, setHasSearched] = useState(false);

    const user = JSON.parse(sessionStorage.getItem('user'));

    useEffect(() => {
        fetchBookmarks();
    }, [navigate]);

    // Auto-refresh on bookmark toggles, library sync, or any other
    // mutation that might change the user's bookmarked rules.
    useLibraryRefresh(() => { if (user) fetchBookmarks(); });

    const pageHelp = {
        title: 'My Rule Bookmarks',
        summary: 'The rules you saved from Browse. They\'re available when adding rules to a rule set ("Add from Bookmarked Rules" tile in the Rule Set Logic Manager).',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    bookmarks.length === 0
                        ? ['No bookmarks yet. Go to Browse, find rules you like, and click the bookmark icon to save them here.']
                        : [
                            `${bookmarks.length} bookmark${bookmarks.length === 1 ? '' : 's'}.`,
                            'The search bar + categories work the same as on Browse — semantic search across names, definitions, and predicates.',
                            'Click the bookmark icon to remove from this list.',
                        ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    // Keep filter and pagination triggers separate to avoid overlapping requests.
    useEffect(() => {
        const hasFilters = !!searchQuery.trim() || searchCategories.length > 0;

        if (!hasFilters) {
            setHasSearched(false);
            setFilteredBookmarks(bookmarks);
            setTotalResults(bookmarks.length);
            setPage(1);
            return;
        }

        setHasSearched(true);
        if (page !== 1) {
            setPage(1);
        } else {
            filterBookmarks();
        }
    }, [searchQuery, searchCategories, bookmarks, topK]); 
    
    // Pagination effect
    useEffect(() => {
        if (hasSearched) {
            filterBookmarks();
        }
    }, [page]);

    const normalizeRule = (rule) => {
        // Handle case where bookmark API might return minimal info vs full info
        // We assume here that the bookmark API should ideally return full rule info.
        // If not, we might only see what's available.
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

    const deriveCategories = (list) => {
        const categorySet = new Set();
        list.forEach((rule) => {
            const ruleCategories = Array.isArray(rule.categories) && rule.categories.length > 0
                ? rule.categories
                : (rule.active_ces || []).map((ce) => ce.name || ce);
            ruleCategories.forEach((cat) => {
                if (cat) categorySet.add(cat);
            });
        });
        return Array.from(categorySet).sort();
    };

    const applyFilters = (source) => {
        let result = Array.isArray(source) ? [...source] : [];

        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            result = result.filter((r) => {
                const nameMatch = (r.custom_name || '').toLowerCase().includes(q);
                const predicateMatch = (r.predicate || '').toLowerCase().includes(q);
                
                // Check categories
                const categories = Array.isArray(r.categories) ? r.categories : [];
                const categoryMatch = categories.some(cat => 
                    (cat || '').toLowerCase().includes(q)
                );

                // Check active CEs (tags)
                const ces = r.active_ces || [];
                const ceMatch = ces.some(ce => 
                    (ce.name || '').toLowerCase().includes(q)
                );
                
                return nameMatch || predicateMatch || categoryMatch || ceMatch;
            });
        }

        if (searchCategories.length > 0) {
            result = result.filter((r) => {
                const rCats = Array.isArray(r.categories) && r.categories.length > 0
                    ? r.categories
                    : (r.active_ces || []).map((ce) => ce.name || ce);
                return searchCategories.some((c) => rCats.includes(c));
            });
        }

        return result.slice(0, topK);
    };

    const fetchBookmarks = async () => {
        try {
            // Your own DRAFT rules — shown first (your unpublished work).
            let draftRules = [];
            try {
                const dRes = await listLocalDrafts();
                draftRules = (dRes.data?.rules || []).map((r) => ({
                    ...r,
                    setup_id: `draft-rule-${r.rule_id}`,
                    source_rule_id: r.rule_id,
                    custom_name: r.name,
                    is_local_draft: true,
                    predicate: r.predicate || r.logic || 'No predicate available',
                    active_ces: r.active_ces || [],
                    categories: r.categories || [],
                }));
            } catch { /* no drafts is fine */ }

            const res = await getRuleBookmarks(user.user_id);
            const bookmarkList = res.data?.bookmarks || [];
            const bookmarkIds = bookmarkList.map((item) => item.rule_id || item.id).filter(Boolean);

            let matches = [];
            if (bookmarkIds.length > 0) {
                const publicRes = await getPublicRules();
                const allRules = publicRes.data.rules || publicRes.data || [];
                const myName = user?.username;
                matches = allRules
                    .filter((rule) => bookmarkIds.includes(rule.rule_id || rule.id))
                    .map(normalizeRule)
                    .sort((a, b) => {
                        // Your own authored bookmarks first, then newest-added.
                        const mineA = a.created_by_username === myName ? 1 : 0;
                        const mineB = b.created_by_username === myName ? 1 : 0;
                        if (mineA !== mineB) return mineB - mineA;
                        const idA = a.rule_id || a.id;
                        const idB = b.rule_id || b.id;
                        return bookmarkIds.indexOf(idB) - bookmarkIds.indexOf(idA);
                    });
            }

            const composed = [...draftRules, ...matches];   // drafts (yours) first
            setBookmarks(composed);
            setFilteredBookmarks(composed);
            setTotalResults(composed.length);
            setAvailableCategories(deriveCategories(composed));
        } catch {
            setBookmarks([]);
            setFilteredBookmarks([]);
            setAvailableCategories([]);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteDraftRule = async (rule) => {
        const ok = await showConfirmDialog({
            title: 'Delete draft rule?',
            message: `"${rule.custom_name}" will be removed from your library.`,
            confirmText: 'Delete', cancelText: 'Keep', variant: 'danger',
        });
        if (!ok) return;
        try {
            await deleteDraftRule(rule.source_rule_id);
            fetchBookmarks();
        } catch (err) {
            showAlertDialog({ title: 'Could not delete', message: err.response?.data?.detail || err.message || 'Try again', variant: 'error' });
        }
    };

    const filterBookmarks = async () => {
        setLoading(true);
        try {
            const res = await searchBookmarks({
                user_id: user.user_id,
                q: searchQuery, // Can be empty string
                categories: searchCategories.join(','),
                asset_types: 'rule',
                page: page,
                page_size: topK
            });

            const results = res.data.results || [];
            const mapped = results.map(r => ({
                ...r,
                rule_id: r.id,
                setup_id: r.id,
                custom_name: r.name,
                predicate: r.content,
                active_ces: (r.ces || []).map(c => ({ name: c })),
                categories: r.categories
            })).map(normalizeRule);
            
            setTotalResults(res.data.total_results || mapped.length);
            setFilteredBookmarks(mapped);
        } catch {
            // Fallback to local filter if backend fails (or if feature not fully ready)
            // Note: Local fallback won't support server-side pagination efficiently
            const local = applyFilters(bookmarks);
            setFilteredBookmarks(local);
            setTotalResults(local.length);
        } finally {
            setLoading(false);
        }
    };

    const handleRemoveBookmark = async (rule) => {
        const ruleId = rule.rule_id || rule.id;
        try {
            await removeRuleBookmark(user.user_id, ruleId);
            setBookmarks(prev => {
                const next = prev.filter(b => (b.rule_id || b.id) !== ruleId);
                setAvailableCategories(deriveCategories(next));
                setFilteredBookmarks(applyFilters(next));
                return next;
            });
            showAlertDialog({ title: 'Removed', message: 'Rule removed from your bookmarks.', variant: 'success' });
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not remove bookmark.', variant: 'error' });
        }
    };

    // Same pill chrome as Browse / BrowseCEs — gradient indigo→blue when
    // active, translucent white with blur when not. Keeps the navigation
    // language consistent across every tab-pair in the app.
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

    // The page body (search + results + pagination). When embedded in the
    // unified Bookmarks page, only this is rendered — the parent provides the
    // Layout, header, tabs and CreateActions. When standalone, we wrap it in
    // a Layout + page-header below so direct /bookmarks/rules rendering and
    // the existing tests keep working unchanged.
    // "Created by you" filter — your drafts + rules you authored.
    const myName = user?.username;
    const isMine = (r) => r.is_local_draft || (myName && r.created_by_username === myName);
    const visibleBookmarks = mineOnly ? filteredBookmarks.filter(isMine) : filteredBookmarks;

    const body = (
        <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                    <SearchPanel
                        query={searchQuery}
                        onQueryChange={setSearchQuery}
                        categories={searchCategories}
                        onCategoriesChange={setSearchCategories}
                        topK={topK}
                        onTopKChange={(value) => setTopK(value)}
                        onSearch={filterBookmarks} 
                        onReset={() => {
                            setSearchQuery('');
                            setSearchCategories([]);
                            setTopK(10);
                            setHasSearched(false);
                            setFilteredBookmarks(bookmarks); // Show all on reset
                        }}
                        loading={false}
                        showAssetTypeFilter={false}
                        searchPlaceholder="Search in your bookmarks..."
                        availableCategories={availableCategories}
                        allowEmptyQuery={true}
                    />

                    {loading ? (
                        <div style={{ textAlign: 'center', padding: '60px', color: '#94a3b8' }}>Loading...</div>
                    ) : visibleBookmarks.length === 0 ? (
                         <div className="empty-state">
                            <FiInbox size={64} style={{ color: '#64748b', marginBottom: '20px' }} />
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>{mineOnly ? 'Nothing yours yet' : 'No Rules Found'}</h2>
                            <p style={{marginBottom: '20px', color: '#94a3b8'}}>
                                {mineOnly
                                    ? "You have no draft rules or rules you authored yet."
                                    : bookmarks.length === 0
                                        ? "You haven't bookmarked or drafted any rules yet."
                                        : "No rules match your search."}
                            </p>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                                {bookmarks.length === 0 && !mineOnly && (
                                    <button className="primary-btn" onClick={() => navigate('/browse')}>Browse Public Rules</button>
                                )}
                                <button className="primary-btn" onClick={() => navigate('/community')}>Browse Community</button>
                            </div>
                        </div>
                    ) : (
                        <>
                            {hasSearched && (
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px',
                                    marginBottom: '12px',
                                    paddingBottom: '12px',
                                    borderBottom: '2px solid rgba(148, 163, 184, 0.18)'
                                }}>
                                    <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                        Search Results ({totalResults} found)
                                    </h2>
                                </div>
                            )}
                            <div className="rules-list">
                                {(!hasSearched ? visibleBookmarks.slice((page - 1) * topK, page * topK) : visibleBookmarks).map((rule) => (
                                    rule.is_local_draft ? (
                                        <RuleCard
                                            key={rule.setup_id}
                                            rule={rule}
                                            isExpanded={expandedRule === rule.setup_id}
                                            onToggle={() => setExpandedRule(expandedRule === rule.setup_id ? null : rule.setup_id)}
                                            onDelete={() => handleDeleteDraftRule(rule)}
                                            onRemoveCE={() => {}}
                                            onAddCE={() => {}}
                                        />
                                    ) : (
                                        <RuleCard
                                            key={rule.setup_id}
                                            rule={rule}
                                            isExpanded={expandedRule === rule.setup_id}
                                            onToggle={() => setExpandedRule(expandedRule === rule.setup_id ? null : rule.setup_id)}
                                            readOnly
                                            onBookmark={handleRemoveBookmark}
                                            bookmarkLabel="Remove"
                                            isBookmarked={true}
                                            onDelete={() => {}}
                                            onRemoveCE={() => {}}
                                            onAddCE={() => {}}
                                        />
                                    )
                                ))}
                            </div>
                        </>
                    )}
                    
                    {!loading && !mineOnly && visibleBookmarks.length > 0 && (
                        <Pagination
                            currentPage={page}
                            totalItems={totalResults}
                            pageSize={topK}
                            onPageChange={setPage}
                        />
                    )}
                </div>
        </div>
    );

    if (embedded) return body;

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '8px', flexWrap: 'wrap' }}>
                        <button onClick={() => navigate('/community')} style={backBtnStyle}>
                            <FiArrowLeft /> Back to Community
                        </button>
                        <button onClick={() => navigate('/bookmarks/rules')} style={pillStyle(true)}>
                            My Rules
                        </button>
                        <button onClick={() => navigate('/bookmarks/ces')} style={pillStyle(false)}>
                            My CEs
                        </button>
                    </div>
                    <h1>My Bookmarked Rules</h1>
                    <p>Manage and search your saved rules.</p>
                </div>
            </header>

            {body}
        </Layout>
    );
};

const backBtnStyle = { background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500 };

export default BookmarksRules;
