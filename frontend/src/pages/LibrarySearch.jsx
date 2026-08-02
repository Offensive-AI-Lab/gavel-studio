import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { FiFilter, FiRefreshCcw, FiSearch, FiHome, FiUsers } from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import { searchLibrary } from '../api';
import RuleCard from '../components/RuleCard/RuleCard';
import CognitiveElementCard from '../components/CognitiveElementCard/CognitiveElementCard';
import { useTutorialContent } from '../contexts/TutorialContext';
import '../css/LibrarySearch.css';

const toCeCard = (item) => ({
  ...item,
  ce_id: item.id,
  name: item.name,
  definition: item.content || '',
  category: (item.categories && item.categories[0]) || item.type || 'Context',
  categories: item.categories || [],
});

const LibrarySearch = () => {
  const user = useMemo(() => JSON.parse(sessionStorage.getItem('user')), []);
  const [query, setQuery] = useState('');
  const [categories, setCategories] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [topK, setTopK] = useState(10);
  const [expandedRule, setExpandedRule] = useState(null);
  const [expandedCe, setExpandedCe] = useState(null);
  const assetParam = 'rule,ce';

  const pageHelp = {
    title: 'Library Search',
    summary: 'Hybrid search across the whole library — rules and cognitive elements together. It blends meaning-based (semantic) matching with keyword and name matching, so close paraphrases and exact terms both surface.',
    sections: [
      {
        heading: 'How to use it',
        bullets: [
          'Type a query (3+ characters auto-searches); results mix matching rules and CEs.',
          'Optional Categories filter narrows results to specific topics.',
          'Top-K controls how many results come back.',
          'Open any rule or CE card to expand its details, or bookmark it for reuse.',
        ],
      },
    ],
  };
  useTutorialContent(pageHelp);

  const runSearch = async () => {
    if (!query.trim()) {
      setError('Enter a query to search');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const res = await searchLibrary({
        q: query.trim(),
        categories: categories.trim() || undefined,
        asset_types: assetParam,
        top_k: topK,
        candidate_limit: 80,
      });
      const list = res.data?.results || [];
      setResults(list);
    } catch {
      setError('Search failed. Please try again.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (query.trim().length >= 3) {
      runSearch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ruleResults = results.filter((r) => r.asset_type === 'rule');
  const ceResults = results.filter((r) => r.asset_type !== 'rule');

  const toRuleCard = (r) => ({
    ...r,
    setup_id: `rule-${r.id}`,
    rule_id: r.id,
    custom_name: r.name,
    predicate: r.content || 'No predicate available',
    active_ces: (r.ces && r.ces.length > 0 ? r.ces : []).map((c) =>
      typeof c === 'string' ? { name: c } : c
    ),
  });

  return (
    <Layout>
      <div className="library-page">
        <Breadcrumb items={[
          { label: 'Hub', icon: FiHome, to: '/workspace' },
          { label: 'Search' },
        ]} />
        <header className="library-header">
          <div>
            <p className="eyebrow">Unified Library</p>
            <h1>Search Rules & Cognitive Elements</h1>
            <p className="subhead">Hybrid retrieval with cross-encoder re-ranking and shared categories.</p>
          </div>
          <div className="header-actions">
            {user && <span className="pill pill-ghost">Signed in as {user.username || user.email}</span>}
          </div>
        </header>

        <div className="search-panel">
          <div className="input-group">
            <FiSearch className="icon" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search intents, policies, detectors..."
              onKeyDown={(e) => e.key === 'Enter' && runSearch()}
            />
            <button className="ghost-btn" onClick={() => setQuery('')}>Clear</button>
          </div>

          <div className="filters">
            <div className="filter-group">
              <span className="filter-label">Categories (comma-separated)</span>
              <div className="input-group small">
                <FiFilter className="icon" />
                <input
                  value={categories}
                  onChange={(e) => setCategories(e.target.value)}
                  placeholder="Finance, Security, Safety"
                />
                <button className="ghost-btn" onClick={() => setCategories('')}>Reset</button>
              </div>
            </div>

            <div className="filter-group">
              <span className="filter-label">Results count</span>
              <div className="input-group small">
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value) || 10)}
                />
              </div>
            </div>

            <div className="actions">
              <button className="secondary" onClick={() => { setQuery(''); setCategories(''); setResults([]); setError(''); }}>
                <FiRefreshCcw /> Reset
              </button>
              <button className="primary" onClick={runSearch} disabled={loading}>
                <FiSearch /> {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="alert">
            <div style={{ marginBottom: '10px' }}>{error}</div>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button className="primary" onClick={runSearch} disabled={loading}>
                <FiRefreshCcw /> Try again
              </button>
              <Link to="/community" className="pill pill-ghost" style={{ textDecoration: 'none', gap: '6px' }}>
                <FiUsers /> Browse the Community
              </Link>
            </div>
          </div>
        )}

        <section className="results">
          {loading && <div className="skeleton">Running hybrid search…</div>}
          {!loading && results.length > 0 && (
              <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  marginBottom: '12px',
                  paddingBottom: '12px',
                  borderBottom: '2px solid #e5e7eb'
              }}>
                  <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: 'black' }}>
                      Search Results ({results.length} found)
                  </h2>
              </div>
          )}
          {!loading && results.length === 0 && !error && (
            <div className="empty">
              <div>No results yet. Try a query like "finance policy" or "medical detector".</div>
              <Link to="/community" className="pill pill-ghost" style={{ textDecoration: 'none', gap: '6px', marginTop: '14px' }}>
                <FiUsers /> Browse the Community
              </Link>
            </div>
          )}

          {!loading && ruleResults.length > 0 && (
            <div className="section-block">
              <div className="section-header">Rules</div>
              <div className="rules-stack">
                {ruleResults.map((r) => (
                  <RuleCard
                    key={`rule-${r.id}`}
                    rule={toRuleCard(r)}
                    isExpanded={expandedRule === r.id}
                    onToggle={() => setExpandedRule(expandedRule === r.id ? null : r.id)}
                    onDelete={() => {}}
                    onRemoveCE={() => {}}
                    onAddCE={() => {}}
                    readOnly
                  />
                ))}
              </div>
            </div>
          )}

          {!loading && ceResults.length > 0 && (
            <div className="section-block">
              <div className="section-header">Cognitive Elements</div>
              <div className="rules-stack">
                {ceResults.map((item) => (
                  <CognitiveElementCard
                    key={`ce-${item.id}`}
                    ce={toCeCard(item)}
                    isOpen={expandedCe === item.id}
                    onToggle={() => setExpandedCe(expandedCe === item.id ? null : item.id)}
                  />
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
};

export default LibrarySearch;