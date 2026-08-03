import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import Pagination from '../components/Pagination/Pagination';
import { getCEBookmarks, removeCEBookmark, getCognitiveDataset, getCognitiveElements, searchBookmarks, listLocalDrafts, deleteDraftCE, getCeDependentDraftRules } from '../api';
import { forgetRecent } from '../utils/recents';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import CognitiveElementCard from '../components/CognitiveElementCard/CognitiveElementCard';
import { FiArrowLeft, FiInbox } from 'react-icons/fi';
import Swal from 'sweetalert2';
import { showAlertDialog, showConfirmDialog } from '../components/ConfirmDialog/confirmDialog';

const BookmarksCEs = ({ embedded = false, mineOnly = false }) => {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    // The ?ce= value we've already auto-expanded, so a re-render doesn't force
    // a card the user just collapsed back open.
    const autoExpandedRef = useRef(null);
    const [bookmarks, setBookmarks] = useState([]);
    const [filteredBookmarks, setFilteredBookmarks] = useState([]);
    const [expandedCe, setExpandedCe] = useState(null);
    const [loading, setLoading] = useState(true);
    const [previewCache, setPreviewCache] = useState({});
    // True sample count per CE — the preview itself is capped at 10.
    const [previewTotals, setPreviewTotals] = useState({});
    
    // Search state
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [availableCategories, setAvailableCategories] = useState([]);
    const [topK, setTopK] = useState(10);
    const [page, setPage] = useState(1);
    const [totalResults, setTotalResults] = useState(0);
    const [hasSearched, setHasSearched] = useState(false);

    const user = JSON.parse(sessionStorage.getItem('user'));

    useEffect(() => {
        fetchBookmarks();
    }, [navigate]);

    // Auto-refresh on any library mutation app-wide.
    useLibraryRefresh(() => { if (user) fetchBookmarks(); });

    const pageHelp = {
        title: 'My CE Bookmarks',
        summary: 'The cognitive elements you saved from Browse · CEs. These are the building blocks for new rules — pick from this list when composing a rule from scratch.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    bookmarks.length === 0
                        ? ['No CE bookmarks yet. Go to Browse → CEs and bookmark the atomic detectors you want to compose into rules.']
                        : [
                            `${bookmarks.length} bookmarked CE${bookmarks.length === 1 ? '' : 's'}.`,
                            'These appear in the "Build from Bookmarked CEs" picker on the Rule Set Logic Manager page.',
                            'Click the bookmark icon to remove a CE from this list.',
                        ],
            },
        ],
    };
    useTutorialContent(pageHelp);

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

    const deriveCategories = (list) => {
        const categorySet = new Set();
        list.forEach((ce) => {
            (ce.categories || []).forEach((cat) => {
                if (cat) categorySet.add(cat);
            });
        });
        return Array.from(categorySet).sort();
    };

    const applyFilters = (source) => {
        let result = Array.isArray(source) ? [...source] : [];

        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            result = result.filter((ce) => {
                const nameMatch = (ce.name || '').toLowerCase().includes(q);
                const defMatch = (ce.definition || '').toLowerCase().includes(q);
                
                // Check categories
                const categories = ce.categories || [];
                const catMatch = categories.some(cat => 
                    (cat || '').toLowerCase().includes(q)
                );
                
                return nameMatch || defMatch || catMatch;
            });
        }

        if (searchCategories.length > 0) {
            result = result.filter((ce) => {
                const ceCats = ce.categories || [];
                return searchCategories.some((c) => ceCats.includes(c));
            });
        }

        return result.slice(0, topK);
    };

    const fetchBookmarks = async () => {
        try {
            // Your own DRAFT CEs — shown first (your unpublished work).
            let draftCes = [];
            try {
                const dRes = await listLocalDrafts();
                draftCes = (dRes.data?.ces || []).map((ce) => mapCeData({ ...ce, is_local_draft: true }));
            } catch { /* no drafts is fine */ }

            const res = await getCEBookmarks(user.user_id);
            const data = res.data?.bookmarks || [];
            const bookmarkIds = data.map((item) => item.ce_id || item.id).filter(Boolean);

            let matches = [];
            if (bookmarkIds.length > 0) {
                const allRes = await getCognitiveElements(user.user_id);
                const raw = allRes.data || [];
                const allCes = Array.isArray(raw) ? raw : (raw.results || []);
                const myName = user?.username;
                matches = allCes
                    .filter((ce) => bookmarkIds.includes(ce.ce_id || ce.id))
                    .map(mapCeData)
                    .sort((a, b) => {
                        const mineA = a.created_by_username === myName ? 1 : 0;
                        const mineB = b.created_by_username === myName ? 1 : 0;
                        if (mineA !== mineB) return mineB - mineA;
                        return bookmarkIds.indexOf(b.ce_id) - bookmarkIds.indexOf(a.ce_id);
                    });
            }

            const composed = [...draftCes, ...matches];   // drafts (yours) first
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

    const handleDeleteDraftCe = async (ce) => {
        const ok = await showConfirmDialog({
            title: 'Delete draft CE?',
            message: `"${ce.name}" and its training samples will be removed from your library.`,
            confirmText: 'Delete', cancelText: 'Keep', variant: 'danger',
        });
        if (!ok) return;
        // Deleting a draft CE cascade-deletes the draft rules that use it, so
        // collect their ids BEFORE the delete — their Recents entries have to
        // go too, and afterwards there's no way to look them up.
        let cascadedRuleIds = [];
        try {
            const dep = await getCeDependentDraftRules(ce.ce_id);
            cascadedRuleIds = (dep.data?.rules || []).map(r => r.rule_id);
        } catch {
            // Best-effort: if this fails, the navigation-time prune still
            // cleans the entries up the first time one is clicked.
        }
        try {
            await deleteDraftCE(ce.ce_id);
            forgetRecent('ce', ce.ce_id);
            cascadedRuleIds.forEach(id => forgetRecent('rule', id));
            fetchBookmarks();
        } catch (err) {
            showAlertDialog({ title: 'Could not delete', message: err.response?.data?.detail || err.message || 'Try again', variant: 'error' });
        }
    };

    const mapCeData = (item) => ({
        // Spread the raw item first so backend fields the card depends on
        // (public_id, created_by_username) and future additions survive into
        // it. The named keys below override for normalization. Without the
        // spread, public_id was dropped
        ...item,
        ce_id: item.ce_id || item.id,
        name: item.name,
        definition: item.definition || item.content || '',
        category: (item.categories && item.categories[0]) || item.category || item.type || 'Context',
        categories: item.categories || (item.category ? [item.category] : []),
        is_local_draft: item.is_local_draft,
        examples: item.examples || [],
    });

    const filterBookmarks = async () => {
        setLoading(true);
        try {
            const res = await searchBookmarks({
                user_id: user.user_id,
                q: searchQuery,
                categories: searchCategories.join(','),
                asset_types: 'ce',
                page: page,
                page_size: topK
            });

            const results = res.data.results || [];
            // Spread `r` first so fields the backend returns (like
            // is_local_draft) survive into mapCeData. The Public / Draft
            // badge in CognitiveElementCard reads is_local_draft and
            // hides itself if the value isn't a boolean — enumerating
            // fields by hand here used to drop it, so filtered results
            // rendered without the badge while unfiltered results showed it.
            const mapped = results.map(r => ({
                ...r,
                ce_id: r.id,
                id: r.id,
                name: r.name,
                definition: r.content,
                categories: r.categories,
            })).map(mapCeData);

            setTotalResults(res.data.total_results || mapped.length);
            setFilteredBookmarks(mapped);
        } catch {
            const local = applyFilters(bookmarks);
            setFilteredBookmarks(local);
            setTotalResults(local.length);
        } finally {
            setLoading(false);
        }
    };

    const handleRemoveBookmark = async (ce) => {
        try {
            await removeCEBookmark(user.user_id, ce.ce_id);
            setBookmarks(prev => {
                const next = prev.filter(b => b.ce_id !== ce.ce_id);
                setAvailableCategories(deriveCategories(next));
                setFilteredBookmarks(applyFilters(next));
                return next;
            });
            showAlertDialog({ title: 'Removed', message: 'CE removed from your bookmarks.', variant: 'success' });
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not remove bookmark.', variant: 'error' });
        }
    };

    const normalizeSamples = (raw) => {
        const arr = Array.isArray(raw) ? raw : (raw ? [raw] : []);
        return arr.map((item) => {
            if (Array.isArray(item)) return item.map((msg) => toMessage(msg));
            if (item && typeof item === 'object' && ('role' in item || 'content' in item)) return [toMessage(item)];
            return [toMessage({ role: 'sample', content: String(item) })];
        });
    };

    const toMessage = (msg) => ({
        role: msg?.role || 'message',
        content: msg?.content || (typeof msg === 'string' ? msg : ''),
    });

    const toggleExpand = async (ceId) => {
        setExpandedCe(expandedCe === ceId ? null : ceId);
        if (expandedCe === ceId) return;
        if (previewCache[ceId]) return;
        try {
            const res = await getCognitiveDataset(ceId);
            const raw = res.data?.training_data_preview || res.data?.training_data || [];
            const normalized = normalizeSamples(raw);
            setPreviewCache((prev) => ({ ...prev, [ceId]: normalized }));
            setPreviewTotals((prev) => ({ ...prev, [ceId]: res.data?.samples_count ?? null }));
        } catch {
            setPreviewCache((prev) => ({ ...prev, [ceId]: [] }));
        }
    };

    // Same pill chrome as Browse / BrowseCEs / BookmarksRules — gradient
    // indigo→blue when active, translucent white with blur when not.
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
    // a Layout + page-header below so direct /bookmarks/ces rendering and the
    // existing tests keep working unchanged.
    // "Created by you" filter — your drafts + CEs you authored.
    const myName = user?.username;
    const isMine = (c) => c.is_local_draft || (myName && c.created_by_username === myName);
    const visibleBookmarks = mineOnly ? filteredBookmarks.filter(isMine) : filteredBookmarks;

    // Deep link: /bookmarks/ces?ce=<id> auto-expands that CE once the list has
    // loaded — the counterpart of BrowseCEs' ?ce= handling, and where a DRAFT
    // CE's Recents entry points: the public browse page filters drafts out of
    // its list, so a /community/ces?ce=<draft id> link could never open one.
    // Keyed on the param so navigating to a DIFFERENT recent re-expands, while
    // a manual collapse on the same URL is respected.
    useEffect(() => {
        const ceParam = searchParams.get('ce');
        if (!ceParam || loading || autoExpandedRef.current === ceParam) return;
        const idx = visibleBookmarks.findIndex((c) => String(c.ce_id) === String(ceParam));
        if (idx < 0) return;
        autoExpandedRef.current = ceParam;
        if (!hasSearched) setPage(Math.floor(idx / topK) + 1);   // jump to its page
        toggleExpand(visibleBookmarks[idx].ce_id);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchParams, visibleBookmarks, loading, hasSearched, topK]);

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
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>{mineOnly ? 'Nothing yours yet' : 'No CEs Found'}</h2>
                            <p style={{marginBottom: '20px', color: '#94a3b8'}}>
                                {mineOnly
                                    ? "You have no draft CEs or CEs you authored yet."
                                    : bookmarks.length === 0
                                        ? "You haven't bookmarked or drafted any CEs yet."
                                        : "No CEs match your search."}
                            </p>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                                {bookmarks.length === 0 && !mineOnly && (
                                    <button className="primary-btn" onClick={() => navigate('/browse/ces')}>Browse Public CEs</button>
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
                                {(!hasSearched ? visibleBookmarks.slice((page - 1) * topK, page * topK) : visibleBookmarks).map((ce) => (
                                    ce.is_local_draft ? (
                                        <CognitiveElementCard
                                            key={ce.ce_id}
                                            ce={ce}
                                            isOpen={expandedCe === ce.ce_id}
                                            onToggle={toggleExpand}
                                            samples={previewCache[ce.ce_id]}
                                        samplesTotal={previewTotals[ce.ce_id]}
                                            onDelete={(c) => handleDeleteDraftCe(c)}
                                        />
                                    ) : (
                                        <CognitiveElementCard
                                            key={ce.ce_id}
                                            ce={ce}
                                            isOpen={expandedCe === ce.ce_id}
                                            onToggle={toggleExpand}
                                            samples={previewCache[ce.ce_id]}
                                        samplesTotal={previewTotals[ce.ce_id]}
                                            onBookmark={handleRemoveBookmark}
                                            isBookmarked={true}
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
                        <button onClick={() => navigate('/community/ces')} style={backBtnStyle}>
                            <FiArrowLeft /> Back to Community
                        </button>
                        <button onClick={() => navigate('/bookmarks/rules')} style={pillStyle(false)}>
                            My Rules
                        </button>
                        <button onClick={() => navigate('/bookmarks/ces')} style={pillStyle(true)}>
                            My CEs
                        </button>
                    </div>
                    <h1>My Bookmarked CEs</h1>
                    <p>Manage and search your saved cognitive elements.</p>
                </div>
            </header>

            {body}
        </Layout>
    );
};

const backBtnStyle = { background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500 };

export default BookmarksCEs;
