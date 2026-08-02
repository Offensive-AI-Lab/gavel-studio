import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    FiShield, FiChevronRight, FiLogOut, FiBookmark, FiUsers,
    FiFileText, FiBox, FiCheckCircle, FiDownload, FiRefreshCw,
} from 'react-icons/fi';
import { syncLibrary } from '../../api';
import { useSyncStatus } from '../../contexts/SyncStatusContext';
import SidebarCreate from '../SidebarCreate/SidebarCreate';
import { getRecents } from '../../utils/recents';
import './Sidebar.css';

// Recents groups (client-side, recently-opened items per type).
const RECENT_GROUPS = [
    { type: 'guardrail', label: 'Rule Sets', Icon: FiShield },
    { type: 'rule', label: 'Rules', Icon: FiFileText },
    { type: 'ce', label: 'CEs', Icon: FiBox },
];

// Persisted collapse state for the recents groups (so a group the user closed
// stays closed across navigations / sessions).
const RECENTS_OPEN_KEY = 'gavel_recents_open';

const Sidebar = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const user = JSON.parse(sessionStorage.getItem('user'));
    const { status: syncStatus, pulling: syncPulling, setStatus: setSyncStatus, setPulling: setSyncPulling } = useSyncStatus();

    // Manual "Sync now" fallback. Ongoing updates are pushed and applied
    // automatically (LibrarySyncStream); this only runs on an explicit force.
    const handlePullUpdates = async () => {
        if (syncPulling) return;
        setSyncPulling(true);
        try {
            await syncLibrary({ force: true });
            setSyncStatus('synced');
        } catch (err) {
            console.warn('[Sidebar] pull updates failed:', err);
        } finally {
            setSyncPulling(false);
        }
    };

    // Recents — read from localStorage, refresh on the gavel:recents event and
    // on navigation (a newly-opened item is recorded by the destination page).
    const [recents, setRecents] = useState({ guardrail: [], rule: [], ce: [] });
    const [openRecents, setOpenRecents] = useState(() => {
        try {
            const raw = localStorage.getItem(RECENTS_OPEN_KEY);
            if (raw) return new Set(JSON.parse(raw));
        } catch { /* fall through to default */ }
        return new Set(['guardrail']);
    });

    const loadRecents = () => setRecents({
        guardrail: getRecents('guardrail'),
        rule: getRecents('rule'),
        ce: getRecents('ce'),
    });

    useEffect(() => {
        loadRecents();
        const onRec = () => loadRecents();
        window.addEventListener('gavel:recents', onRec);
        return () => window.removeEventListener('gavel:recents', onRec);
    }, []);

    useEffect(() => { loadRecents(); }, [location.pathname]);

    const toggleRecent = (type) => setOpenRecents(prev => {
        const next = new Set(prev);
        next.has(type) ? next.delete(type) : next.add(type);
        try { localStorage.setItem(RECENTS_OPEN_KEY, JSON.stringify([...next])); } catch { /* best-effort */ }
        return next;
    });

    const isActive = (path) => location.pathname === path;
    const startsWith = (p) => location.pathname.startsWith(p);
    // Recents match the FULL url (path + query). Without the query, every CE
    // recent (which shares /community/ces) would light up at once; with it, only
    // the one whose ?ce=<id> is open is marked active.
    const recentActive = (p) => `${location.pathname}${location.search}` === p;

    return (
        <aside className="sidebar">
            <div className="sidebar-header" onClick={() => navigate('/workspace')}>
                <div className="logo-icon">
                    <img src="/gavel-icon.png" alt="" width={30} height={30} />
                </div>
                <h2 className="brand-text">GAVEL</h2>
            </div>

            {/* Library sync indicator (pulling / available / synced). */}
            <div
                className={`sidebar-sync sidebar-sync-${syncPulling ? 'pulling' : (syncStatus === 'available' ? 'available' : 'synced')}`}
                onClick={syncStatus === 'available' && !syncPulling ? handlePullUpdates : undefined}
                role={syncStatus === 'available' ? 'button' : undefined}
                aria-label={
                    syncPulling ? 'Pulling library updates' :
                    syncStatus === 'available' ? 'Updates available — click to pull' :
                    'Library is up to date'
                }
                title={
                    syncPulling ? 'Pulling…' :
                    syncStatus === 'available' ? 'Click to apply the latest library updates from the gavel-rules repository.' :
                    'Local library is up to date with the public registry.'
                }
            >
                <span className="sidebar-sync-icon">
                    {syncPulling ? (
                        <FiRefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />
                    ) : syncStatus === 'available' ? (
                        <FiDownload size={14} />
                    ) : (
                        <FiCheckCircle size={14} />
                    )}
                </span>
                <span className="sidebar-sync-text">
                    {syncPulling ? 'Updating…' :
                     syncStatus === 'available' ? 'Updates available' :
                     'Library synced'}
                </span>
            </div>

            <nav className="sidebar-nav">
                {/* EXPLORE — discover and author content. */}
                <div className="nav-section-label">EXPLORE</div>
                <div
                    className={`nav-item ${(startsWith('/community') || startsWith('/browse')) ? 'active' : ''}`}
                    onClick={() => navigate('/community')}
                >
                    <FiUsers className="nav-icon" />
                    <span>Community</span>
                </div>
                <SidebarCreate />

                {/* MY WORKSPACE — the user's own space. */}
                <div className="nav-section-label" style={{ marginTop: '20px' }}>MY WORKSPACE</div>
                <div
                    className={`nav-item ${startsWith('/bookmarks') ? 'active' : ''}`}
                    onClick={() => navigate('/bookmarks')}
                >
                    <FiBookmark className="nav-icon" />
                    <span>Your Library</span>
                </div>
                <div
                    className={`nav-item ${isActive('/guardrails') ? 'active' : ''}`}
                    onClick={() => navigate('/guardrails')}
                >
                    <FiShield className="nav-icon" />
                    <span>Rule Sets</span>
                </div>

                {/* RECENTS — recently-opened items, per type (client-side). */}
                <div className="nav-section-label" style={{ marginTop: '20px' }}>RECENTS</div>
                {RECENT_GROUPS.map(g => {
                    const items = recents[g.type] || [];
                    const open = openRecents.has(g.type);
                    return (
                        <div key={g.type}>
                            <div className="nav-item" onClick={() => toggleRecent(g.type)}>
                                <g.Icon className="nav-icon" />
                                <span>{g.label}</span>
                                <FiChevronRight
                                    style={{ marginLeft: 'auto', transition: 'transform 150ms ease', transform: open ? 'rotate(90deg)' : 'none' }}
                                />
                            </div>
                            {open && (
                                items.length === 0 ? (
                                    <div style={recentEmptyStyle}>Nothing recent</div>
                                ) : (
                                    items.map(it => (
                                        <div
                                            key={`${g.type}-${it.id}`}
                                            className={`nav-item ${it.path && recentActive(it.path) ? 'active' : ''}`}
                                            style={recentItemStyle}
                                            onClick={() => it.path && navigate(it.path)}
                                            title={it.name}
                                        >
                                            <span className="truncate-text">{it.name}</span>
                                        </div>
                                    ))
                                )
                            )}
                        </div>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                {user && (
                    <div className="user-info" title={user.email || ''}>
                        <div className="user-avatar">
                            {(user.username || user.email || '?').toString().charAt(0).toUpperCase()}
                        </div>
                        <div className="user-meta">
                            <span className="user-name">{user.username || user.email || 'Account'}</span>
                            {user.email && <span className="user-email">{user.email}</span>}
                        </div>
                    </div>
                )}
            </div>
        </aside>
    );
};

const recentItemStyle = { paddingLeft: '34px', fontSize: '0.85rem' };
const recentEmptyStyle = { paddingLeft: '34px', fontSize: '0.78rem', color: '#64748b', fontStyle: 'italic', padding: '6px 0 6px 34px' };

export default Sidebar;
