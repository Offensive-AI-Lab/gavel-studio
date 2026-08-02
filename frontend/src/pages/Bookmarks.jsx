import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import BookmarksRules from './BookmarksRules';
import BookmarksCEs from './BookmarksCEs';
import BookmarksRuleSets from './BookmarksRuleSets';
import { useTutorialContent } from '../contexts/TutorialContext';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import { FiHome, FiBookmark, FiUser } from 'react-icons/fi';
import '../css/RulesManager.css';

// "Your Library" — one hub with Rules / CEs tabs. Each tab shows the user's
// own DRAFTS first, then the items they've bookmarked from the Community. A
// "Created by you" toggle filters to just the user's own work (drafts + items
// they authored). There is no separate Drafts page — drafts live here.
const Bookmarks = () => {
    const navigate = useNavigate();
    const location = useLocation();
    // Deep-link aware: /bookmarks/ces → CEs tab; /bookmarks/rule-sets → Rule Sets
    // tab; everything else → Rules tab.
    const initialTab = location.pathname.endsWith('/ces')
        ? 'ces'
        : location.pathname.endsWith('/rule-sets')
            ? 'rule-sets'
            : 'rules';
    const [tab, setTab] = useState(initialTab);
    const [mineOnly, setMineOnly] = useState(false);

    const pageHelp = {
        title: 'Your Library',
        summary: 'Your saved rules and cognitive elements, plus your own unpublished drafts — all in the Rules / CEs tabs (your drafts come first). Saved items are available when composing rule sets ("Add from your Library" tile).',
        sections: [
            {
                heading: 'Right now',
                bullets: [
                    'Your own drafts appear first in each tab; publish or delete them right here.',
                    'Toggle "Created by you" to see only your drafts and the items you authored.',
                    'Search + categories work the same as Community — semantic search across names, definitions, and firing logic.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    const selectTab = (next) => {
        setTab(next);
        const path = next === 'ces'
            ? '/bookmarks/ces'
            : next === 'rule-sets'
                ? '/bookmarks/rule-sets'
                : '/bookmarks/rules';
        navigate(path);
    };

    const tabBtnStyle = (active) => ({
        padding: '8px 18px',
        borderRadius: '999px',
        border: active ? '1px solid transparent' : '1px solid rgba(148, 163, 184, 0.18)',
        background: active
            ? 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)'
            : 'rgba(15, 23, 42, 0.55)',
        color: active ? '#ffffff' : '#cbd5e1',
        fontWeight: 600,
        fontSize: '0.9rem',
        cursor: 'pointer',
        boxShadow: active
            ? '0 6px 18px -2px rgba(99, 102, 241, 0.55)'
            : '0 2px 6px rgba(2, 6, 23, 0.30)',
        backdropFilter: active ? 'none' : 'blur(8px)',
        transition: 'transform 120ms ease, box-shadow 180ms ease',
    });

    return (
        <Layout>
            <Breadcrumb items={[
                { label: 'Hub', icon: FiHome, to: '/workspace' },
                { label: 'Your Library', icon: FiBookmark },
            ]} />
            <div style={{ display: 'flex', gap: '10px', marginBottom: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                <button style={tabBtnStyle(tab === 'rules')} onClick={() => selectTab('rules')}>
                    Rules
                </button>
                <button style={tabBtnStyle(tab === 'rule-sets')} onClick={() => selectTab('rule-sets')}>
                    Rule Sets
                </button>
                <button style={tabBtnStyle(tab === 'ces')} onClick={() => selectTab('ces')}>
                    CEs
                </button>
                {/* "Created by you" filter — your drafts + items you authored. */}
                <button
                    onClick={() => setMineOnly((v) => !v)}
                    style={{ ...tabBtnStyle(mineOnly), marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                    title="Show only your drafts and the items you created"
                >
                    <FiUser size={14} /> Created by you
                </button>
            </div>

            <h1 style={{ margin: '4px 0 0', fontSize: '1.6rem', color: '#f1f5f9' }}>Your Library</h1>
            <p style={{ margin: '6px 0 20px', color: '#64748b' }}>
                Your own drafts (shown first) plus the Rules and Cognitive Elements you saved from the Community. Available when adding rules or building from CEs in the Rule Set Manager.
            </p>

            {tab === 'rules' && <BookmarksRules embedded mineOnly={mineOnly} />}
            {tab === 'rule-sets' && <BookmarksRuleSets embedded mineOnly={mineOnly} />}
            {tab === 'ces' && <BookmarksCEs embedded mineOnly={mineOnly} />}
        </Layout>
    );
};

export default Bookmarks;
