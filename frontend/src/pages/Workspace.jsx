import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    FiCompass, FiLayers, FiArrowRight, FiBookOpen, FiCpu, FiCheckCircle,
    FiBarChart2, FiActivity, FiShield,
} from 'react-icons/fi';
import { getDashboardData } from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import { useTutorial, useTutorialContent } from '../contexts/TutorialContext';
import InlineHelp from '../components/InlineHelp/InlineHelp';
import { aboutGavel } from '../components/InlineHelp/instructorHelp';

// ---------------------------------------------------------------------------
// Workspace — landing page right after login. Modernized to match the
// authenticated-app gradient + card chrome (same atmosphere as Layout.css,
// same accent gradient as the sidebar / tab pills). The page is plain
// inline-style only (no shared CSS file) so each visual decision lives
// next to where it's used.
// ---------------------------------------------------------------------------

const Workspace = () => {
    const navigate = useNavigate();
    const [stats, setStats] = useState(null);
    const [guardrailSummary, setGuardrailSummary] = useState([]);
    const [recentActivity, setRecentActivity] = useState([]);

    // Pulled out of the mount-effect so the library-refresh hook can call
    // it on `gavel:libraryChanged`. Reads `user` from localStorage each
    // time so it works both on initial mount (before useState commits)
    // and on refresh-after-mutation calls.
    const refresh = useCallback(() => {
        const storedUser = JSON.parse(sessionStorage.getItem('user') || 'null');
        if (!storedUser) return;
        getDashboardData(storedUser.user_id)
            .then(res => {
                setStats(res.data.stats);
                setGuardrailSummary(res.data.classifier_summary || []);
                setRecentActivity(res.data.recent_activity || []);
            })
            .catch(() => {});
    }, []);

    const { showWelcome } = useTutorial();

    useEffect(() => {
        const storedUser = JSON.parse(sessionStorage.getItem('user') || 'null');
        refresh();

        // First-login auto-fire of the onboarding modal. Backed by the
        // per-user tutorial_seen column; the Tutorial component flips
        // it to TRUE on finish/skip so this won't re-trigger after a
        // refresh. Older sessions whose stored user blob predates the
        // column are treated as "not seen" — they get the tutorial
        // once on next /workspace mount, which is the intended UX.
        if (!storedUser.tutorial_seen) {
            // Forces the 5-slide welcome regardless of any per-page
            // content registration — first-login auto-fire wants the
            // platform overview, not page-mode help.
            showWelcome();
        }
    }, [navigate, refresh, showWelcome]);

    useLibraryRefresh(refresh);

    // Per-page tutorial — what to do RIGHT NOW based on hub state.
    // The bullets shift as the user adds models, then guardrails,
    // then trains them.
    const totalModels = stats?.total_models || 0;
    const totalGuardrails = stats?.total_classifiers || 0;
    const activeGuardrails = stats?.active_classifiers || 0;
    const pageHelp = {
        title: 'Workspace',
        summary: 'Your hub. A rule set is a reusable collection of rules — build its rules, then pick the model it runs on when you train. Use the cards above to start a rule set, browse the public library, or manage your models.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    totalGuardrails === 0
                        ? ['No rule sets yet. Click the Rule Sets tile (or Rule Sets in the sidebar) to create one — name it, build rules, then pick a model to train on.']
                        : activeGuardrails === 0
                            ? [`${totalGuardrails} rule set${totalGuardrails === 1 ? '' : 's'} created — none trained yet. Open one to add rules, then Train (you'll pick its model there).`]
                            : [`${activeGuardrails} of ${totalGuardrails} rule set${totalGuardrails === 1 ? '' : 's'} are trained. Run evaluation or real-time monitoring on them.`],
            },
            {
                heading: 'Sidebar at a glance',
                bullets: [
                    'GAVEL logo (top) — click to return to this page from anywhere.',
                    'Library status pill — green "synced" means you have the latest. New content is pushed to you and applied automatically the moment it is published, so this stays green on its own.',
                    'Rule Sets — your rule sets; the primary place you work.',
                    'Models — manage your registered LLMs and see which rule sets are attached to each.',
                    'Community — public library of vetted rules and CEs.',
                    'My Bookmarks — saved for reuse. My Drafts — your local rules and CEs (not yet in the shared library).',
                    'Tutorial — re-opens this help (page-aware on every page).',
                    'Models (lower section) — your model tree, expandable to show the rule sets attached underneath.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    // Greeting that adapts to the time of day. No name is appended: there is no
    // login, so the only name available is the fixed local user's — addressing
    // the single operator by it reads as odd rather than personal.
    const greeting = (() => {
        const h = new Date().getHours();
        if (h < 12) return 'Good morning';
        if (h < 18) return 'Good afternoon';
        return 'Good evening';
    })();

    return (
        <div style={pageStyle}>
            <div style={shellStyle}>
                {/* ----- Hero header ----- */}
                <div style={heroStyle}>
                    <div>
                        <div style={kickerStyle}>
                            <FiShield size={14} /> GAVEL Cloud Platform
                        </div>
                        <h1 style={heroTitleStyle}>{greeting}.</h1>
                        <p style={heroSubtitleStyle}>
                            Pick where you want to work today. You can switch anytime.
                        </p>
                    </div>
                </div>

                <InlineHelp content={aboutGavel} />

                {/* ----- Action tiles (Rule Sets is the primary flow, shown centered) ----- */}
                <div style={actionsGridStyle}>
                    <ActionTile
                        icon={FiBookOpen}
                        title="Browse"
                        subtitle="Explore the public library, bookmark rules and CEs you want to reuse."
                        accent={['#0ea5e9', '#2563eb']}
                        onClick={() => navigate('/browse')}
                    />
                    <ActionTile
                        icon={FiShield}
                        title="Rule Sets"
                        subtitle="Build a rule set, then pick the model it runs on at train time."
                        accent={['#6366f1', '#8b5cf6']}
                        onClick={() => navigate('/guardrails')}
                    />
                </div>

                {/* ----- Stats grid ----- */}
                {stats && (
                    <Section title="Your statistics" icon={FiActivity}>
                        <div style={statsGridStyle}>
                            <StatCard icon={FiCpu}         label="Models"      value={stats.total_models}        accent={['#6366f1', '#4f46e5']} />
                            <StatCard icon={FiShield}      label="Rule Sets"   value={stats.total_classifiers}   accent={['#0ea5e9', '#2563eb']} />
                            <StatCard icon={FiCheckCircle} label="Active"      value={stats.active_classifiers}  accent={['#10b981', '#059669']} />
                            <StatCard icon={FiLayers}      label="Rules"       value={stats.total_rules}         accent={['#8b5cf6', '#7c3aed']} />
                            <StatCard icon={FiActivity}    label="CEs"         value={stats.total_ces}           accent={['#f59e0b', '#d97706']} />
                            <StatCard icon={FiBarChart2}   label="Evaluations" value={stats.total_evaluations}   accent={['#3b82f6', '#1d4ed8']} />
                        </div>
                    </Section>
                )}

                {/* ----- Rule set overview ----- */}
                {guardrailSummary.length > 0 && (
                    <Section title="Rule set overview" icon={FiShield}>
                        <div style={tableWrapStyle}>
                            <table style={tableStyle}>
                                <thead>
                                    <tr>
                                        <th style={thStyle}>Rule Set</th>
                                        <th style={thStyle}>Model</th>
                                        <th style={thStyle}>Status</th>
                                        <th style={{ ...thStyle, textAlign: 'center' }}>Rules</th>
                                        <th style={{ ...thStyle, textAlign: 'center' }}>CEs</th>
                                        <th style={thStyle}>Last eval</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {guardrailSummary.map((c, i) => (
                                        <tr key={c.classifier_id}
                                            style={i % 2 === 0 ? trAltStyle : undefined}>
                                            <td style={tdStrongStyle}>{c.classifier_name}</td>
                                            <td style={tdMutedStyle}>{c.model_name || 'No model yet'}</td>
                                            <td style={tdStyle}>
                                                <StatusPill status={c.status} />
                                            </td>
                                            <td style={{ ...tdStyle, textAlign: 'center' }}>{c.rule_count}</td>
                                            <td style={{ ...tdStyle, textAlign: 'center' }}>{c.ce_count}</td>
                                            <td style={tdMutedStyle}>
                                                {c.last_evaluation
                                                    ? new Date(c.last_evaluation).toLocaleDateString('en-US')
                                                    : '—'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </Section>
                )}

                {/* ----- Recent activity ----- */}
                {recentActivity.length > 0 && (
                    <Section title="Recent activity" icon={FiBarChart2}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            {recentActivity.slice(0, 5).map((a, i) => (
                                <div key={i} style={activityRowStyle}>
                                    <div style={activityDotStyle} />
                                    <span style={{ fontSize: '0.9rem', color: '#f1f5f9' }}>
                                        <strong>{a.classifier_name}</strong>{' — '}
                                        <span style={{ color: '#cbd5e1' }}>{a.detail || a.event_type}</span>
                                    </span>
                                    <span style={activityDateStyle}>
                                        {a.created_at ? new Date(a.created_at).toLocaleDateString('en-US') : ''}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </Section>
                )}

                <div style={footerHintStyle}>
                    You can return here from any page using the back-to-hub button in the sidebar.
                </div>
            </div>
        </div>
    );
};


// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const ActionTile = ({ icon: Icon, title, subtitle, accent, onClick }) => {
    const [hover, setHover] = useState(false);
    return (
        <button
            onClick={onClick}
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            style={{
                ...tileStyle,
                transform: hover ? 'translateY(-3px)' : 'translateY(0)',
                borderColor: hover ? `${accent[0]}66` : 'rgba(148, 163, 184, 0.14)',
                boxShadow: hover
                    ? `0 22px 44px -10px ${accent[0]}66, 0 8px 18px -6px rgba(2, 6, 23, 0.55)`
                    : '0 4px 12px rgba(2, 6, 23, 0.30)',
            }}
        >
            <div
                style={{
                    width: 52,
                    height: 52,
                    borderRadius: 14,
                    background: `linear-gradient(135deg, ${accent[0]} 0%, ${accent[1]} 100%)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#fff',
                    boxShadow: `0 6px 16px -4px ${accent[0]}66`,
                    flexShrink: 0,
                }}
            >
                <Icon size={24} />
            </div>
            <div style={{ textAlign: 'left', flex: 1, minWidth: 0 }}>
                <div style={tileTitle}>{title}</div>
                <div style={tileSubtitle}>{subtitle}</div>
            </div>
            <div style={{
                color: '#64748b',
                transform: hover ? 'translateX(4px)' : 'translateX(0)',
                transition: 'transform 200ms cubic-bezier(0.16, 1, 0.3, 1)',
            }}>
                <FiArrowRight size={20} />
            </div>
        </button>
    );
};

const Section = ({ title, icon: Icon, children }) => (
    <div style={sectionStyle}>
        <div style={sectionHeaderStyle}>
            <Icon size={16} color="#a5b4fc" />
            <h3 style={sectionTitleStyle}>{title}</h3>
        </div>
        {children}
    </div>
);

const StatCard = ({ icon: Icon, label, value, accent }) => (
    <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '14px 16px',
        borderRadius: 14,
        background: 'rgba(15, 23, 42, 0.55)',
        border: '1px solid rgba(148, 163, 184, 0.14)',
        backdropFilter: 'blur(8px)',
        boxShadow: '0 4px 12px rgba(2, 6, 23, 0.30)',
        color: '#e2e8f0',
    }}>
        <div style={{
            width: 38,
            height: 38,
            borderRadius: 10,
            background: `linear-gradient(135deg, ${accent[0]} 0%, ${accent[1]} 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            flexShrink: 0,
            boxShadow: `0 4px 12px -3px ${accent[0]}66`,
        }}>
            <Icon size={18} />
        </div>
        <div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em' }}>
                {value ?? 0}
            </div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: 2, fontWeight: 500 }}>
                {label}
            </div>
        </div>
    </div>
);

const StatusPill = ({ status }) => {
    const cfg = {
        active:           { bg: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', label: 'Active' },
        training:         { bg: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', label: 'Training' },
        needs_retraining: { bg: 'linear-gradient(135deg, #f97316 0%, #c2410c 100%)', label: 'Needs Retrain' },
        error:            { bg: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', label: 'Error' },
        untrained:        { bg: 'linear-gradient(135deg, #94a3b8 0%, #64748b 100%)', label: 'Untrained' },
    }[status] || { bg: 'linear-gradient(135deg, #94a3b8 0%, #64748b 100%)', label: status };
    return (
        <span style={{
            display: 'inline-block',
            padding: '3px 10px',
            borderRadius: 999,
            fontSize: '0.7rem',
            fontWeight: 700,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            background: cfg.bg,
            color: '#fff',
        }}>
            {cfg.label}
        </span>
    );
};


// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const pageStyle = {
    minHeight: '100vh',
    width: '100vw',
    display: 'flex',
    justifyContent: 'center',
    padding: '40px 24px 60px',
    boxSizing: 'border-box',
    overflowY: 'auto',
    fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
    color: '#e2e8f0',
    background:
        'radial-gradient(ellipse 60% 50% at 12% 0%, rgba(99, 102, 241, 0.30) 0%, transparent 60%),' +
        'radial-gradient(ellipse 50% 50% at 100% 30%, rgba(59, 130, 246, 0.20) 0%, transparent 55%),' +
        'radial-gradient(ellipse 70% 50% at 50% 105%, rgba(139, 92, 246, 0.22) 0%, transparent 60%),' +
        'linear-gradient(180deg, #060a1a 0%, #0c1226 50%, #0a0f23 100%)',
};

const shellStyle = {
    width: 'min(1100px, 100%)',
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
};

const heroStyle = {
    padding: '32px 36px',
    borderRadius: 20,
    background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.50) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.14)',
    backdropFilter: 'blur(14px)',
    boxShadow:
        '0 1px 0 rgba(255, 255, 255, 0.06) inset, ' +
        '0 16px 36px -12px rgba(2, 6, 23, 0.50), ' +
        '0 6px 18px -6px rgba(99, 102, 241, 0.18)',
};

const kickerStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 10px',
    borderRadius: 999,
    background: 'rgba(99, 102, 241, 0.20)',
    color: '#c7d2fe',
    fontSize: '0.72rem',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    marginBottom: 12,
};

const heroTitleStyle = {
    margin: 0,
    fontSize: '2.1rem',
    fontWeight: 800,
    letterSpacing: '-0.02em',
    color: '#f8fafc',
    lineHeight: 1.15,
};

const heroSubtitleStyle = {
    margin: '8px 0 0 0',
    fontSize: '1rem',
    color: '#94a3b8',
    lineHeight: 1.5,
};

const actionsGridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 16,
};

const tileStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    padding: '20px 22px',
    borderRadius: 18,
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.55) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.14)',
    cursor: 'pointer',
    width: '100%',
    fontFamily: 'inherit',
    color: '#e2e8f0',
    backdropFilter: 'blur(12px)',
    transition: 'transform 220ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1), border-color 200ms ease',
    textAlign: 'left',
};

const tileTitle = {
    fontWeight: 700,
    fontSize: '1.1rem',
    color: '#f1f5f9',
    letterSpacing: '-0.01em',
    marginBottom: 4,
};

const tileSubtitle = {
    color: '#94a3b8',
    fontSize: '0.88rem',
    lineHeight: 1.45,
};

const sectionStyle = {
    padding: '20px 24px 22px',
    borderRadius: 18,
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.55) 100%)',
    border: '1px solid rgba(148, 163, 184, 0.14)',
    backdropFilter: 'blur(12px)',
    boxShadow: '0 4px 12px rgba(2, 6, 23, 0.30)',
};

const sectionHeaderStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 14,
};

const sectionTitleStyle = {
    margin: 0,
    fontSize: '0.78rem',
    fontWeight: 800,
    color: '#cbd5e1',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
};

const statsGridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: 12,
};

const tableWrapStyle = {
    overflowX: 'auto',
    borderRadius: 12,
    border: '1px solid rgba(148, 163, 184, 0.12)',
};

const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.88rem',
    color: '#e2e8f0',
};

const thStyle = {
    textAlign: 'left',
    padding: '10px 14px',
    color: '#94a3b8',
    fontWeight: 700,
    fontSize: '0.72rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    borderBottom: '1px solid rgba(148, 163, 184, 0.14)',
    background: 'rgba(99, 102, 241, 0.10)',
};

const tdStyle = {
    padding: '10px 14px',
    color: '#cbd5e1',
    borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
};

const tdStrongStyle = { ...tdStyle, fontWeight: 600, color: '#f1f5f9' };
const tdMutedStyle  = { ...tdStyle, color: '#94a3b8', fontSize: '0.82rem' };

const trAltStyle = { background: 'rgba(99, 102, 241, 0.04)' };

const activityRowStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 14px',
    borderRadius: 10,
    background: 'rgba(99, 102, 241, 0.10)',
    border: '1px solid rgba(129, 140, 248, 0.18)',
    color: '#e2e8f0',
};

const activityDotStyle = {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)',
    flexShrink: 0,
};

const activityDateStyle = {
    marginLeft: 'auto',
    fontSize: '0.78rem',
    color: '#94a3b8',
    fontWeight: 500,
};

const footerHintStyle = {
    marginTop: 8,
    color: '#64748b',
    fontSize: '0.85rem',
    textAlign: 'center',
};

export default Workspace;
