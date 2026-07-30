import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import CommunityTabs from '../components/CommunityTabs/CommunityTabs';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import Pagination from '../components/Pagination/Pagination';
import { getCognitiveElements, getCognitiveDataset, addCEBookmark, getCEBookmarks, removeCEBookmark, getAllCategories } from '../api';
import useLibrarySearch from '../hooks/useLibrarySearch';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorialContent } from '../contexts/TutorialContext';
import CognitiveElementCard from '../components/CognitiveElementCard/CognitiveElementCard';
import { FiArrowLeft, FiInbox } from 'react-icons/fi';
import Swal from 'sweetalert2';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import { normalizeCategoryValue } from '../utils/categoryUtils';
import { recordRecent } from '../utils/recents';

const BrowseCEs = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();

    // Phase 4: ?author=<username> filters the CE listing to one
    // contributor's work. Clicking the × on the chip clears it.
    const authorFilter = (searchParams.get('author') || '').trim().toLowerCase() || null;
    const clearAuthorFilter = () => {
        const next = new URLSearchParams(searchParams);
        next.delete('author');
        setSearchParams(next, { replace: true });
    };
    const [ces, setCes] = useState([]);
    const [expandedCe, setExpandedCe] = useState(null);
    const [loading, setLoading] = useState(true);
    const [previewCache, setPreviewCache] = useState({});
    const [bookmarkIds, setBookmarkIds] = useState(new Set());
    const [bookmarks, setBookmarks] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [searchTopK, setSearchTopK] = useState(10);
    const [page, setPage] = useState(1);
    const [availableCategories, setAvailableCategories] = useState([]);
    const [searchReloadKey, setSearchReloadKey] = useState(0);
    const user = JSON.parse(sessionStorage.getItem('user'));

    // Stable identity for the assetTypes prop so the hook's deps don't churn.
    const assetTypes = useMemo(() => ['ce'], []);

    // Same hook as Browse.jsx — single source of truth for live library search.
    const {
        results: rawSearchResults,
        totalResults,
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
        // CEs allow categories-only browse — useful before the user types anything.
        allowEmptyQuery: true,
        reloadKey: searchReloadKey,
    });


    useEffect(() => {
        fetchCes();
        fetchCategories();
        fetchBookmarks();
    }, [navigate]);

    // Auto-refresh on any library mutation app-wide (AI-pipeline finishes,
    // HF sync pulls in new CEs, bookmark toggles in another tab, etc.).
    useLibraryRefresh(() => {
        if (user) {
            fetchCes();
            fetchBookmarks();
        }
    });

    const pageHelp = {
        title: 'Community · CEs',
        summary: 'Cognitive Elements (CEs) are the atomic concepts the rule set learns to detect — things like "user_asks_for_medical_advice" or "assistant_provides_disclaimer". Bookmark useful ones to reuse when building rules.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    ces.length === 0
                        ? ['No CEs in the library yet, or still syncing.']
                        : [
                            `${ces.length} CE${ces.length === 1 ? '' : 's'} available. Each has a definition + categories.`,
                            'Click a CE to expand and see its in-scope examples and out-of-scope notes.',
                            'Bookmark CEs you want to reuse when composing rules.',
                        ],
            },
            {
                heading: 'How CEs differ from rules',
                bullets: [
                    'A CE is ONE detector — single concept, single signal.',
                    'A rule organizes multiple CEs into named groups with a firing condition (e.g. "all of required and 1 of option_1").',
                    'Building a new rule from your bookmarked CEs is the typical workflow.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);


    const parseArrayString = (str) => {
        if (!str) return [];
        
        // If already an array, return it
        if (Array.isArray(str)) {
            return str.map(item => normalizeCategoryValue(item)).filter(Boolean);
        }
        
        if (typeof str !== 'string') return [];
        
        const trimmed = str.trim();
        let values = [];
        
        // Handle PostgreSQL array format: {value1,value2} or {"value1","value2"}
        if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
            const inner = trimmed.slice(1, -1);
            // Split by comma and clean each value
            values = inner.split(',').map(v => {
                let cleaned = v.trim().replace(/^"|"$/g, '').replace(/^'|'$/g, '');
                // Remove any remaining brackets
                cleaned = cleaned.replace(/^[{[]+/, '').replace(/[\]}]+$/, '').trim();
                return cleaned;
            }).filter(v => v.length > 0);
        }
        
        return values;
    }

    const fetchCategories = async () => {
        try {
            const res = await getAllCategories();
            setAvailableCategories(res.data || []);
           } catch {
             setAvailableCategories([]);
        }
    };

    const fetchCes = async () => {
        try {
            const res = await getCognitiveElements(user.user_id);
            const data = res.data || [];
            // Community/Browse is the PUBLIC space — never surface local
            // drafts here; they live in "Your Library".
            const list = (Array.isArray(data) ? data : []).filter(ce => !ce.is_local_draft);
            setCes(list);
        } catch {
            setCes([]);
        } finally {
            setLoading(false);
        }
    };

    const fetchBookmarks = async () => {
        if (!user?.user_id) return;
        try {
            const res = await getCEBookmarks(user.user_id);
            const list = res.data?.bookmarks || [];
            setBookmarks(list);
            const ids = new Set(list.map((b) => b.ce_id));
            setBookmarkIds(ids);
        } catch {
            setBookmarkIds(new Set());
            setBookmarks([]);
        }
    };

    // Reset to page 1 when filters change so a new query doesn't open on page N
    // of stale results. The hook itself handles the actual fetch.
    useEffect(() => {
        setPage(1);
    }, [searchQuery, searchCategories, searchTopK]);

    const normalizeSamples = (raw) => {
        // Accepts strings, objects, or arrays; returns array of conversations, each conversation is array of {role, content}
        const arr = Array.isArray(raw) ? raw : (raw ? [raw] : []);
        return arr.map((item) => {
            // If already an array of messages
            if (Array.isArray(item)) {
                return item.map((msg) => toMessage(msg));
            }
            // If object-like, convert to a single-message conversation
            if (item && typeof item === 'object') {
                return [toMessage(item)];
            }
            // Fallback string
            return [toMessage({ role: 'sample', content: String(item) })];
        });
    };

    const toMessage = (msg) => {
        if (typeof msg === 'string') {
            return { role: 'sample', content: msg };
        }

        if (msg && typeof msg === 'object') {
            if (msg.role || msg.content) {
                return { role: (msg.role || 'message').toLowerCase(), content: msg.content || '' };
            }
            if (msg.input) {
                return { role: 'user', content: msg.input };
            }
            if (msg.output || msg.response) {
                return { role: 'assistant', content: msg.output || msg.response };
            }
            if (msg.system) {
                return { role: 'system', content: msg.system };
            }
            // Fallback: render the object for visibility
            try {
                return { role: 'sample', content: JSON.stringify(msg, null, 2) };
            } catch {
                return { role: 'sample', content: String(msg) };
            }
        }

        return { role: 'sample', content: '' };
    };

    const handleBookmark = async (ce) => {
        if (!user?.user_id) return;
        try {
            if (bookmarkIds.has(ce.ce_id)) {
                await removeCEBookmark(user.user_id, ce.ce_id);
                setBookmarkIds((prev) => {
                    const next = new Set(prev);
                    next.delete(ce.ce_id);
                    return next;
                });
                setBookmarks((prev) => prev.filter((b) => b.ce_id !== ce.ce_id));
                showAlertDialog({ title: 'Removed', message: 'CE removed from your bookmarks.', variant: 'success' });
            } else {
                await addCEBookmark(user.user_id, ce.ce_id);
                setBookmarkIds((prev) => {
                    const next = new Set(prev);
                    next.add(ce.ce_id);
                    return next;
                });
                setBookmarks((prev) => [{ ce_id: ce.ce_id, name: ce.name }, ...prev]);
                showAlertDialog({ title: 'Saved', message: 'CE added to your bookmarks.', variant: 'success' });
            }
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not bookmark this CE.', variant: 'error' });
        }
    };

    const ensurePreview = async (ceId) => {
        if (previewCache[ceId]) return;
        try {
            const res = await getCognitiveDataset(ceId);
            const raw = res.data?.training_data_preview || res.data?.training_data || [];
            setPreviewCache((prev) => ({ ...prev, [ceId]: normalizeSamples(raw) }));
        } catch {
            setPreviewCache((prev) => ({ ...prev, [ceId]: [] }));
        }
    };

    const toggleExpand = async (ceId, ceName) => {
        setExpandedCe(expandedCe === ceId ? null : ceId);
        if (expandedCe === ceId) return;
        // Opening a CE → record it for the sidebar's Recents. CEs have no detail
        // route, so the recent deep-links back here with ?ce=<id>, which auto-
        // expands this exact CE (and keeps each recent's "active" highlight unique).
        if (ceName) recordRecent('ce', { id: ceId, name: ceName, path: `/community/ces?ce=${ceId}` });
        ensurePreview(ceId);
    };

    // Deep link: /community/ces?ce=<id> auto-expands that CE once it's loaded.
    // Keyed on the param value so navigating to a DIFFERENT recent re-expands,
    // but a manual collapse (same URL) is respected — we don't force it back open.
    const autoExpandedRef = useRef(null);
    useEffect(() => {
        const ceParam = searchParams.get('ce');
        if (!ceParam || loading || autoExpandedRef.current === ceParam) return;
        const idx = ces.findIndex((c) => String(c.ce_id) === String(ceParam));
        if (idx < 0) return;
        const found = ces[idx];
        autoExpandedRef.current = ceParam;
        setPage(Math.floor(idx / 10) + 1);   // jump to the page that holds this CE
        setExpandedCe(found.ce_id);
        ensurePreview(found.ce_id);
        // Scroll once the right page has rendered the card.
        setTimeout(() => {
            const el = document.getElementById(`ce-card-${found.ce_id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 120);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchParams, ces, loading]);

    const mapSearchCe = (item) => {
        const categories = item.categories || [];
        // If categories is a string, parse it
        const parsedCategories = typeof categories === 'string' ? parseArrayString(categories) : categories;
        return {
            ce_id: item.id,
            name: item.name,
            definition: item.content || '',
            category: (parsedCategories && parsedCategories[0]) || item.type || 'Context',
            categories: parsedCategories,
            is_local_draft: item.is_local_draft,
            examples: item.examples || [],
            // v2 CE fields — role is the primary badge, title the display
            // name; tags render as muted pills.
            role: item.role,
            title: item.title,
            tags: item.tags,
        };
    };

    // Hook returns raw API rows; map to the shape CognitiveElementCard expects.
    // mapSearchCe is a stable closure over only its argument; excluded from deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    const searchResults = useMemo(
        () => (rawSearchResults || []).map(mapSearchCe),
        [rawSearchResults],
    );

    // Filter CEs locally based on search categories
    const filteredSearchResults = searchResults.filter((ce) => {
        if (searchCategories.length === 0) return true;
        
        const ceCats = new Set();
        
        // Extract all categories from the CE
        if (Array.isArray(ce.categories)) {
            ce.categories.forEach(c => {
                const n = normalizeCategoryValue(c);
                if (n) ceCats.add(n);
            });
        } else if (typeof ce.categories === 'string') {
            // If it's a string, parse it
            const parsed = parseArrayString(ce.categories);
            parsed.forEach(c => {
                const n = normalizeCategoryValue(c);
                if (n) ceCats.add(n);
            });
        }
        
        // Also add the single category field
        if (ce.category) {
            const n = normalizeCategoryValue(ce.category);
            if (n) ceCats.add(n);
        }
        
        // Filter: Show CE if it matches ANY of the selected categories
        const matches = searchCategories.some(cat => ceCats.has(cat));
        return matches;
    });

    // Match Browse.jsx's pill styling so the two tabs feel like the
    // same navigation surface — see Browse.jsx for design rationale.
    const backBtnStyle = {
        background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500,
        transition: 'color 150ms ease',
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

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <CommunityTabs active="ces" />
                    <h1>Cognitive Elements</h1>
                    <p>Explore public CEs and inspect sample excitation data.</p>
                </div>
            </header>

            {/* Phase 4 author chip — only visible when ?author=… is set. */}
            {authorFilter && (
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '10px',
                    padding: '10px 14px', marginBottom: '14px',
                    background: 'rgba(15, 23, 42, 0.55)',
                    border: '1px solid rgba(148, 163, 184, 0.18)',
                    borderRadius: '12px',
                }}>
                    <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Filtered to author:</span>
                    <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: '6px',
                        padding: '4px 6px 4px 12px',
                        background: 'rgba(99, 102, 241, 0.18)',
                        border: '1px solid rgba(129, 140, 248, 0.45)',
                        borderRadius: '999px',
                        color: '#c7d2fe', fontWeight: 600, fontSize: '0.9rem',
                    }}>
                        @{authorFilter}
                        <button
                            onClick={clearAuthorFilter}
                            aria-label="Clear author filter"
                            style={{
                                width: '22px', height: '22px',
                                borderRadius: '50%',
                                border: 'none',
                                background: 'rgba(2, 6, 23, 0.55)',
                                color: '#cbd5e1', cursor: 'pointer',
                                fontSize: '1rem', lineHeight: 1,
                                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                            }}
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
                        // Search is live via useLibrarySearch; the button is a no-op.
                        onSearch={() => {}}
                        onReset={() => {
                            setSearchQuery('');
                            setSearchCategories([]);
                            setPage(1);
                        }}
                        loading={searchLoading}
                        showAssetTypeFilter={false}
                        allowEmptyQuery={true}
                        searchPlaceholder="Search cognitive elements..."
                        availableCategories={availableCategories}
                    />

                    {searchError && (
                        <div className="alert" style={{ marginBottom: '16px' }}>
                            <div style={{ marginBottom: '10px' }}>{searchError}</div>
                            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                                <button className="primary-btn" onClick={() => setSearchReloadKey((k) => k + 1)}>Try again</button>
                                <button className="primary-btn" onClick={() => navigate('/workspace')}>Go to Hub</button>
                            </div>
                        </div>
                    )}

                    {searchLoading && <div className="skeleton" style={{ marginBottom: '12px' }}>Searching cognitive elements…</div>}

                    {!searchLoading && searchResults.length === 0 && hasSearched && (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px 20px',
                            color: '#94a3b8'
                        }}>
                            <p>
                                {(() => {
                                    const q = (searchQuery || '').trim();
                                    const cats = searchCategories || [];
                                    if (q && cats.length > 0) {
                                        return `No cognitive elements found for "${q}" in ${cats.join(', ')}.`;
                                    }
                                    if (q) {
                                        return `No cognitive elements found for "${q}".`;
                                    }
                                    if (cats.length > 0) {
                                        return cats.length === 1
                                            ? `No cognitive elements in the ${cats[0]} category.`
                                            : `No cognitive elements in the selected categories: ${cats.join(', ')}.`;
                                    }
                                    return 'No cognitive elements match your filters.';
                                })()}
                            </p>
                        </div>
                    )}

                    {!searchLoading && searchResults.length > 0 && hasSearched && (
                        <div className="section-block" style={{ marginBottom: '16px' }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px',
                                marginBottom: '12px',
                                paddingBottom: '12px',
                                borderBottom: '2px solid rgba(148, 163, 184, 0.18)'
                            }}>
                                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                    Search Results ({filteredSearchResults.length} found)
                                </h2>
                            </div>
                            <div className="rules-list">
                                {filteredSearchResults.map((ce) => (
                                    <CognitiveElementCard
                                        key={`ce-${ce.ce_id}`}
                                        ce={ce}
                                        isOpen={expandedCe === ce.ce_id}
                                        onToggle={() => toggleExpand(ce.ce_id, ce.name)}
                                        samples={previewCache[ce.ce_id]}
                                        onBookmark={handleBookmark}
                                        isBookmarked={bookmarkIds.has(ce.ce_id)}
                                    />
                                ))}
                            </div>
                            
                            {!loading && searchResults.length > 0 && (
                                <Pagination 
                                    currentPage={page}
                                    totalItems={totalResults}
                                    pageSize={searchTopK}
                                    onPageChange={setPage}
                                />
                            )}

                        </div>
                    )}

                    {loading ? (
                        <div style={{ textAlign: 'center', padding: '60px', color: '#94a3b8' }}>Loading...</div>
                    ) : ces.length === 0 && !hasSearched ? (
                        <div className="empty-state">
                            <FiInbox size={64} style={{ color: '#d1d5db', marginBottom: '20px' }} />
                            <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>No Cognitive Elements</h2>
                            <p style={{marginBottom: '20px', color: '#94a3b8'}}>Public CEs will appear here when available.</p>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                                <button className="primary-btn" onClick={() => navigate('/bookmarks/ces')}>Create a CE</button>
                                <button className="primary-btn" onClick={() => navigate('/workspace')}>Go to Hub</button>
                            </div>
                        </div>
                    ) : !hasSearched ? (
                        <div>
                            <div style={{
                                marginBottom: '12px',
                                paddingBottom: '12px',
                                borderBottom: '2px solid rgba(148, 163, 184, 0.18)'
                            }}>
                                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#e2e8f0' }}>
                                    All Public CEs
                                </h2>
                            </div>
                            <div className="rules-list">
                                {ces.slice((page - 1) * 10, page * 10).map((ce) => (
                                    <div key={ce.ce_id} id={`ce-card-${ce.ce_id}`}>
                                        <CognitiveElementCard
                                            ce={ce}
                                            isOpen={expandedCe === ce.ce_id}
                                            onToggle={() => toggleExpand(ce.ce_id, ce.name)}
                                            samples={previewCache[ce.ce_id]}
                                            onBookmark={handleBookmark}
                                            isBookmarked={bookmarkIds.has(ce.ce_id)}
                                        />
                                    </div>
                                ))}
                                {ces.length > 0 && (
                                    <Pagination
                                        currentPage={page}
                                        totalItems={ces.length}
                                        pageSize={10}
                                        onPageChange={setPage}
                                    />
                                )}
                            </div>
                        </div>
                    ) : null}
                </div>
                <aside style={{
                    width: '100%',
                    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.72) 0%, rgba(15, 23, 42, 0.62) 100%)',
                    border: '1px solid rgba(148, 163, 184, 0.18)',
                    borderRadius: '14px', padding: '18px',
                    position: 'sticky', top: '20px', alignSelf: 'flex-start',
                    backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                    boxShadow: '0 8px 24px -8px rgba(2, 6, 23, 0.50), 0 4px 12px rgba(99, 102, 241, 0.12)',
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                        <h3 style={{ margin: 0, fontSize: '0.95rem', color: '#f1f5f9', fontWeight: 700, letterSpacing: '-0.01em' }}>My Bookmarked CEs</h3>
                        <button
                            style={{ background: 'none', border: 'none', color: '#a5b4fc', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, padding: 0 }}
                            onClick={() => navigate('/bookmarks/ces')}
                        >
                            View all
                        </button>
                    </div>
                    {bookmarks.length === 0 ? (
                        <p style={{ color: '#94a3b8', margin: 0, fontSize: '0.88rem' }}>No bookmarks yet.</p>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {bookmarks.map((b) => (
                                <div key={b.ce_id} style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
                                    padding: '10px 12px',
                                    border: '1px solid rgba(148, 163, 184, 0.14)',
                                    borderRadius: '10px',
                                    background: 'rgba(2, 6, 23, 0.55)',
                                    width: '100%',
                                    boxSizing: 'border-box',
                                    transition: 'border-color 180ms ease, box-shadow 180ms ease',
                                }}>
                                    <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.88rem', wordBreak: 'break-word', flex: 1 }}>{b.name || 'CE'}</span>
                                    <button
                                        className="bookmark-btn"
                                        onClick={() => handleBookmark({ ce_id: b.ce_id, name: b.name })}
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

export default BrowseCEs;
