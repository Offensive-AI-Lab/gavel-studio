import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import RuleSetCard from '../components/RuleSetCard/RuleSetCard';
import SearchPanel from '../components/SearchPanel/SearchPanel';
import {
    getRuleSetBookmarks, removeRuleSetBookmark, getPublicRuleSets, getAllCategories,
} from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { showAlertDialog } from '../components/ConfirmDialog/confirmDialog';
import { normalizeCategoryValue } from '../utils/categoryUtils';
import { FiInbox } from 'react-icons/fi';
import '../css/RulesManager.css';

// "Your Library" → Rule Sets tab. Lists the public rule sets the user has
// bookmarked from the Community (the user's OWN rule sets live on the /guardrails
// page, not here). Remove un-bookmarks; forking a bookmarked set into the
// workspace happens on the Rule Sets page.
const BookmarksRuleSets = ({ embedded = false, mineOnly = false }) => {
    const navigate = useNavigate();
    const [ruleSets, setRuleSets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedId, setExpandedId] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchCategories, setSearchCategories] = useState([]);
    const [availableCategories, setAvailableCategories] = useState([]);
    const user = JSON.parse(sessionStorage.getItem('user'));

    useEffect(() => {
        fetchBookmarks();
        fetchCategories();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [navigate]);

    useLibraryRefresh(() => { if (user) fetchBookmarks(); });

    const fetchCategories = async () => {
        try {
            const res = await getAllCategories();
            setAvailableCategories(res.data || []);
        } catch {
            setAvailableCategories([]);
        }
    };

    const fetchBookmarks = async () => {
        try {
            const bk = await getRuleSetBookmarks(user.user_id);
            const ids = new Set((bk.data?.bookmarks || []).map((b) => b.rule_set_id));
            if (ids.size === 0) {
                setRuleSets([]);
                return;
            }
            const res = await getPublicRuleSets();
            const all = res.data?.rule_sets || [];
            setRuleSets(all.filter((rs) => ids.has(rs.rule_set_id)));
        } catch {
            setRuleSets([]);
        } finally {
            setLoading(false);
        }
    };

    const handleRemove = async (ruleSet) => {
        const id = ruleSet.rule_set_id;
        try {
            await removeRuleSetBookmark(user.user_id, id);
            setRuleSets((prev) => prev.filter((rs) => rs.rule_set_id !== id));
            showAlertDialog({ title: 'Removed', message: 'Rule set removed from your bookmarks.', variant: 'success' });
        } catch {
            showAlertDialog({ title: 'Error', message: 'Could not remove bookmark.', variant: 'error' });
        }
    };

    const myName = user?.username;
    const q = searchQuery.trim().toLowerCase();
    const filtered = ruleSets.filter((rs) => {
        if (mineOnly && !(myName && rs.created_by_username === myName)) return false;
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

    const body = (
        <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
            <div style={{ flex: 1, minWidth: 0 }}>
                <SearchPanel
                    query={searchQuery}
                    onQueryChange={setSearchQuery}
                    categories={searchCategories}
                    onCategoriesChange={setSearchCategories}
                    onSearch={() => {}}
                    onReset={() => { setSearchQuery(''); setSearchCategories([]); }}
                    onTopKChange={() => {}}
                    showAssetTypeFilter={false}
                    searchPlaceholder="Search in your bookmarked rule sets..."
                    availableCategories={availableCategories}
                    allowEmptyQuery
                />

                {loading ? (
                    <div style={{ textAlign: 'center', padding: '60px', color: '#94a3b8' }}>Loading...</div>
                ) : filtered.length === 0 ? (
                    <div className="empty-state">
                        <FiInbox size={64} style={{ color: '#64748b', marginBottom: '20px' }} />
                        <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>
                            {mineOnly ? 'Nothing yours yet' : 'No bookmarked rule sets'}
                        </h2>
                        <p style={{ marginBottom: '20px', color: '#94a3b8' }}>
                            {ruleSets.length === 0
                                ? "You haven't bookmarked any rule sets yet."
                                : 'No rule sets match your search.'}
                        </p>
                        <button className="primary-btn" onClick={() => navigate('/community/rule-sets')}>
                            Browse Community Rule Sets
                        </button>
                    </div>
                ) : (
                    <div className="rules-list">
                        {filtered.map((rs) => (
                            <RuleSetCard
                                key={rs.rule_set_id}
                                ruleSet={rs}
                                isExpanded={expandedId === rs.rule_set_id}
                                onToggle={() => setExpandedId(expandedId === rs.rule_set_id ? null : rs.rule_set_id)}
                                onBookmark={handleRemove}
                                bookmarkLabel="Remove"
                                isBookmarked
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );

    if (embedded) return body;

    return (
        <Layout>
            <header className="page-header">
                <div>
                    <h1>My Bookmarked Rule Sets</h1>
                    <p>The rule sets you saved from the Community.</p>
                </div>
            </header>
            {body}
        </Layout>
    );
};

export default BookmarksRuleSets;
