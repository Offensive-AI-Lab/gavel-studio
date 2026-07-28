// Community.jsx — Phase 4 discovery page.
//
// Reachable at /community. Two tabs:
//   * Browse — search artists by username / display name; empty query
//     shows the recently-active list.
//   * Leaderboard — top contributors. Two sub-orderings: by average
//     total contributions.
//
// The artist gate is enforced server-side, so this page never shows
// zero-contribution users. Every card links to /profile/<username>.

import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
    FiUsers, FiSearch, FiAward, FiArrowLeft, FiHome,
} from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import CommunityTabs from '../components/CommunityTabs/CommunityTabs';
import Pagination from '../components/Pagination/Pagination';
import { searchArtists, getLeaderboard } from '../api';
import { useTutorialContent } from '../contexts/TutorialContext';

const PAGE_SIZE = 12;

const Community = () => {
    const [searchParams, setSearchParams] = useSearchParams();

    // URL-driven state so the back button works and a "share this view"
    // link is just the current URL. mode/orderBy live in the query string.
    const mode = searchParams.get('mode') || 'search';   // 'search' | 'leaderboard'
    // Kept so old /community/people?by=…&min=… links still resolve. The
    // backend accepts both for shape compatibility but ignores them: with
    // ratings gone there is only one ranking (contribution volume).
    const orderBy = searchParams.get('by') || 'avg_rating';
    const minRatings = parseInt(searchParams.get('min') || '0', 10) || 0;
    const urlQuery = searchParams.get('q') || '';

    const [query, setQuery] = useState(urlQuery);
    const [page, setPage] = useState(1);
    const [data, setData] = useState({ items: [], total: 0 });
    const [loading, setLoading] = useState(false);

    const pageHelp = {
        title: 'Community',
        summary: 'Discover the people behind the public library. Search by username or browse the leaderboard to find contributors whose work you might want to fork, rate, or follow.',
        sections: [
            {
                heading: 'Two views',
                bullets: [
                    'Search — find an artist by username or display name. Empty query shows recently-active artists.',
                    'Leaderboard — top contributors by how much they have published into the library.',
                ],
            },
            {
                heading: 'Who shows up here',
                bullets: [
                    'Only contributors whose published work is in your synced library appear — a new author shows up after you Sync and their rule/CE lands locally, not before.',
                    'Listeners (registered but no publications) have profile pages by direct URL but don\'t appear in discovery.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    // Debounce the search input so we don't fire on every keystroke.
    useEffect(() => {
        const t = setTimeout(() => {
            // Reflect the query in the URL — debounced so the URL bar
            // doesn't thrash on every keystroke.
            if (mode === 'search') {
                const next = new URLSearchParams(searchParams);
                if (query.trim()) next.set('q', query.trim());
                else next.delete('q');
                setSearchParams(next, { replace: true });
            }
        }, 350);
        return () => clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [query]);

    // Fetch data whenever mode / orderBy / debounced-query / page changes.
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        const fetcher = mode === 'leaderboard'
            ? getLeaderboard(orderBy, page, PAGE_SIZE, minRatings)
            : searchArtists(urlQuery, page, PAGE_SIZE);
        fetcher.then((res) => {
            if (!cancelled) setData(res.data);
        }).catch(() => {
            if (!cancelled) setData({ items: [], total: 0 });
        }).finally(() => {
            if (!cancelled) setLoading(false);
        });
        return () => { cancelled = true; };
    }, [mode, orderBy, minRatings, urlQuery, page]);

    const setMode = (newMode) => {
        const next = new URLSearchParams();
        next.set('mode', newMode);
        if (newMode === 'leaderboard') next.set('by', orderBy);
        setSearchParams(next);
        setPage(1);
    };

    return (
        <Layout>
            <CommunityTabs active="people" />

            <header style={headerStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={iconBoxStyle}><FiUsers size={22} /></div>
                    <div>
                        <h1 style={{ margin: 0, color: '#f8fafc', letterSpacing: '-0.02em', fontSize: '1.75rem' }}>Community</h1>
                        <p style={{ margin: '4px 0 0 0', color: '#94a3b8' }}>
                            Discover the people behind the public library.
                        </p>
                    </div>
                </div>
            </header>

            {/* Mode tabs */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px', marginBottom: '16px' }}>
                <button onClick={() => setMode('search')} style={tabBtnStyle(mode === 'search')}>
                    <FiSearch /> Search
                </button>
                <button onClick={() => setMode('leaderboard')} style={tabBtnStyle(mode === 'leaderboard')}>
                    <FiAward /> Leaderboard
                </button>
            </div>

            {/* Mode body */}
            {mode === 'search' ? (
                <div>
                    <div style={inputWrapStyle}>
                        <FiSearch style={{ color: '#94a3b8' }} />
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => { setQuery(e.target.value); setPage(1); }}
                            placeholder="Search by username or display name…"
                            style={inputStyle}
                        />
                    </div>
                </div>
            ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px 18px', marginBottom: '16px', color: '#94a3b8', fontSize: '0.85rem' }}>
                    Contributors in your synced library, most published first.
                </div>
            )}

            {/* Results grid */}
            {loading ? (
                <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>Loading…</div>
            ) : data.items.length === 0 ? (
                <div style={emptyStateStyle}>
                    <div style={{ marginBottom: '16px' }}>
                        {mode === 'search'
                            ? (urlQuery ? `No artists match "${urlQuery}".` : 'No artists yet. Be the first to publish!')
                            : 'No leaderboard data yet — needs more contributions.'}
                    </div>
                    <Link to="/workspace" style={communityCtaStyle}>
                        <FiHome /> Go to Hub
                    </Link>
                </div>
            ) : (
                <>
                    <div style={gridStyle}>
                        {data.items.map((artist) => (
                            <ArtistCard key={artist.username} artist={artist} />
                        ))}
                    </div>
                    {data.total > PAGE_SIZE && (
                        <Pagination
                            currentPage={page}
                            totalItems={data.total}
                            pageSize={PAGE_SIZE}
                            onPageChange={setPage}
                        />
                    )}
                </>
            )}
        </Layout>
    );
};

// --- Single artist card --------------------------------------------------

const ArtistCard = ({ artist }) => {
    const total = artist.contribution_count_rules + artist.contribution_count_ces;
    return (
        <Link to={`/profile/${artist.username}`} style={cardStyle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={avatarStyle}>
                    {(artist.display_name || artist.username)[0].toUpperCase()}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                        color: '#f8fafc', fontWeight: 700, fontSize: '1rem',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                        {artist.display_name || artist.username}
                        {artist.is_team && <FiAward style={{ color: '#6ee7b7' }} title="Team account" />}
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                        @{artist.username}
                    </div>
                </div>
            </div>
            {artist.bio && (
                <p style={{
                    margin: 0,
                    color: '#cbd5e1',
                    fontSize: '0.88rem',
                    lineHeight: 1.45,
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    marginBottom: '12px',
                }}>{artist.bio}</p>
            )}
            <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                color: '#cbd5e1', fontSize: '0.85rem',
                paddingTop: '12px', borderTop: '1px solid rgba(148, 163, 184, 0.14)',
            }}>
                <span><strong style={{ color: '#e2e8f0' }}>{total}</strong> contributions</span>
            </div>
        </Link>
    );
};

// --- Style fragments -----------------------------------------------------

const backBtnStyle = {
    background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer',
    display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500, padding: 0,
};

const headerStyle = {
    paddingBottom: '20px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.14)',
};

const iconBoxStyle = {
    width: '48px', height: '48px', borderRadius: '14px',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    boxShadow: '0 6px 18px -2px rgba(99, 102, 241, 0.55)',
};

const tabBtnStyle = (active) => ({
    padding: '10px 18px',
    borderRadius: '999px',
    border: active ? '1px solid transparent' : '1px solid rgba(148, 163, 184, 0.18)',
    background: active
        ? 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)'
        : 'rgba(15, 23, 42, 0.55)',
    color: active ? '#ffffff' : '#cbd5e1',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '0.9rem',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    boxShadow: active
        ? '0 6px 18px -2px rgba(99, 102, 241, 0.55)'
        : '0 2px 6px rgba(2, 6, 23, 0.30)',
});


const inputWrapStyle = {
    display: 'flex', alignItems: 'center', gap: '10px',
    padding: '12px 16px',
    background: 'rgba(2, 6, 23, 0.55)',
    border: '1.5px solid rgba(148, 163, 184, 0.22)',
    borderRadius: '12px',
    marginBottom: '16px',
};

const inputStyle = {
    flex: 1,
    background: 'transparent',
    border: 'none',
    outline: 'none',
    color: '#f1f5f9',
    fontSize: '0.95rem',
    fontFamily: 'inherit',
};

const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '14px',
};

const cardStyle = {
    display: 'block',
    padding: '16px',
    borderRadius: '14px',
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.55) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.16)',
    boxShadow: '0 4px 12px -4px rgba(2, 6, 23, 0.40)',
    textDecoration: 'none',
    transition: 'transform 120ms ease, border-color 180ms ease, box-shadow 180ms ease',
};

const avatarStyle = {
    width: '44px', height: '44px', borderRadius: '12px',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    fontWeight: 700,
    fontSize: '1.1rem',
    flexShrink: 0,
};

const emptyStateStyle = {
    padding: '40px 20px',
    textAlign: 'center',
    color: '#94a3b8',
    background: 'rgba(15, 23, 42, 0.40)',
    border: '1px dashed rgba(148, 163, 184, 0.20)',
    borderRadius: '12px',
};

// Body CTA on the empty/error discovery view — same indigo→blue accent as the
// mode tabs, rendered as an inline-flex link to /workspace.
const communityCtaStyle = {
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '10px 18px',
    borderRadius: '999px',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    color: '#ffffff',
    fontWeight: 600,
    fontSize: '0.9rem',
    textDecoration: 'none',
    boxShadow: '0 6px 18px -2px rgba(99, 102, 241, 0.55)',
};

export default Community;
