// CommunityTabs — the shared tab bar for the Community hub.
//
// The public space is one section ("Community") with three tabs: Rules, CEs,
// and People. This bar sits at the top of all three pages so they read as a
// single area, and replaces the old per-page "Browse Rules / Browse CEs" pills.

import { useNavigate } from 'react-router-dom';
import { FiHome, FiFileText, FiCpu, FiUsers, FiLayers } from 'react-icons/fi';
import Breadcrumb from '../Breadcrumb/Breadcrumb';

const TABS = [
    { key: 'rules',     label: 'Rules',     path: '/community',           Icon: FiFileText },
    { key: 'rule-sets', label: 'Rule Sets', path: '/community/rule-sets', Icon: FiLayers },
    { key: 'ces',       label: 'CEs',       path: '/community/ces',       Icon: FiCpu },
    { key: 'people',    label: 'Contributors', path: '/community/people', Icon: FiUsers },
];

// Solid gradient + glow when active (matches the sidebar accent), soft when not.
const pillStyle = (active) => ({
    display: 'inline-flex', alignItems: 'center', gap: '7px',
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
});

const CommunityTabs = ({ active }) => {
    const navigate = useNavigate();
    return (
        <>
            <Breadcrumb items={[
                { label: 'Hub', icon: FiHome, to: '/workspace' },
                { label: 'Community', icon: FiUsers },
            ]} />
            <div style={{ display: 'flex', gap: '10px', marginBottom: '8px', flexWrap: 'wrap' }}>
                {TABS.map(({ key, label, path, Icon }) => (
                    <button key={key} onClick={() => navigate(path)} style={pillStyle(active === key)}>
                        <Icon size={15} /> {label}
                    </button>
                ))}
            </div>
        </>
    );
};

export default CommunityTabs;
