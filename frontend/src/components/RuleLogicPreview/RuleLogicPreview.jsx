// RuleLogicPreview — renders a rule's firing logic in the groups + condition
// model: one chip-cluster per named CE group (group name header + member CE
// pills) and the condition expression verbatim in monospace underneath.
//
// Used by RuleCard / RulePage / RuleSetPage to display stored rules, by the
// Build-Rule wizard as a live preview, and by the logic guide's worked
// examples. `groups` members may be CE names or { ce_id, name } objects.

import { normalizeGroups } from '../../utils/ruleLogic';

const GROUP_COLOR = '#a78bfa';

const memberChip = {
    display: 'inline-flex', alignItems: 'center',
    padding: '3px 10px', borderRadius: 999,
    fontSize: '0.8rem', fontWeight: 700, color: '#f8fafc',
    background: `${GROUP_COLOR}26`, border: `1px solid ${GROUP_COLOR}`,
    whiteSpace: 'nowrap',
};

const groupBox = {
    border: '1px solid rgba(148, 163, 184, 0.22)',
    borderRadius: 10, padding: '8px 10px',
    background: 'rgba(15, 23, 42, 0.45)',
    display: 'flex', flexDirection: 'column', gap: 6,
    minWidth: 0,
};

const groupNameStyle = {
    fontSize: '0.72rem', fontWeight: 800, letterSpacing: '0.04em',
    color: '#c7d2fe', fontFamily: 'monospace',
};

const conditionStyle = {
    display: 'block', marginTop: 10,
    background: 'rgba(2, 6, 23, 0.6)', border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: 8, padding: '8px 10px',
    fontFamily: 'monospace', fontSize: '0.82rem', color: '#e2e8f0',
    wordBreak: 'break-word', whiteSpace: 'pre-wrap',
};

export default function RuleLogicPreview({
    groups,
    condition,
    title = 'Firing logic',
    emptyHint = 'Add cognitive elements to a group to form the firing logic.',
    style = {},
}) {
    const normalized = normalizeGroups(groups);
    const entries = Object.entries(normalized).filter(([, members]) => members.length > 0);
    const cond = String(condition || '').trim();

    return (
        <div style={{
            background: 'rgba(2, 6, 23, 0.55)',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            borderRadius: 12, padding: '12px 14px', ...style,
        }}>
            {title && (
                <div style={{
                    fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase',
                    letterSpacing: '0.05em', color: '#94a3b8', marginBottom: 10,
                }}>{title}</div>
            )}
            {entries.length > 0 ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {entries.map(([gname, members]) => (
                        <div key={gname} style={groupBox}>
                            <span style={groupNameStyle}>{gname}</span>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                {members.map((m, i) => (
                                    <span key={m.ce_id ?? `${m.name}-${i}`} style={memberChip}>{m.name}</span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div style={{ fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic' }}>{emptyHint}</div>
            )}
            {cond && (
                <code style={conditionStyle} title="Firing condition over the groups above">{cond}</code>
            )}
        </div>
    );
}
