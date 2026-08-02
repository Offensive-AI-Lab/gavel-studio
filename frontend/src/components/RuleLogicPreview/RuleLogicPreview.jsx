// RuleLogicPreview — renders a rule's firing logic the same way the
// gavel-rules viewer does: one chip-cluster per named CE group and the
// condition expression underneath. Each group gets a color used for its name
// pill, its member CE chips, and the group name inside the condition text, so
// the rule reads back to its groups at a glance.
//
// Used by RuleCard / RulePage / RuleSetPage to display stored rules, by the
// Build-Rule wizard as a live preview, and by the logic guide's worked
// examples. `groups` members may be CE names or { ce_id, name } objects.

import { normalizeGroups } from '../../utils/ruleLogic';

// Group color cycle, mirroring the gavel-rules viewer's dark-theme palette
// (--g0..--g3): amber, teal, violet, pink.
const GROUP_PALETTE = [
    { fg: '#e3b25e', bg: '#35290f' },
    { fg: '#5fd1c0', bg: '#10302b' },
    { fg: '#b79cf5', bg: '#2a2140' },
    { fg: '#ef8bb4', bg: '#3a1a29' },
];

const memberChip = (color) => ({
    display: 'inline-flex', alignItems: 'center',
    padding: '3px 10px', borderRadius: 999,
    fontSize: '0.8rem', fontWeight: 700, color: color.fg,
    background: color.bg, border: `1px solid ${color.fg}55`,
    whiteSpace: 'nowrap',
});

const groupBox = {
    border: '1px solid rgba(148, 163, 184, 0.22)',
    borderRadius: 10, padding: '8px 10px',
    background: 'rgba(15, 23, 42, 0.45)',
    display: 'flex', flexDirection: 'column', gap: 6,
    minWidth: 0,
};

const groupNamePill = (color) => ({
    alignSelf: 'flex-start',
    fontSize: '0.72rem', fontWeight: 800, letterSpacing: '0.04em',
    color: color.fg, background: color.bg,
    borderRadius: 6, padding: '2px 7px',
    fontFamily: 'monospace', whiteSpace: 'nowrap',
});

const conditionStyle = {
    display: 'block', marginTop: 10,
    background: 'rgba(2, 6, 23, 0.6)', border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: 8, padding: '8px 10px',
    fontFamily: 'monospace', fontSize: '0.82rem', color: '#e2e8f0',
    wordBreak: 'break-word', whiteSpace: 'pre-wrap', lineHeight: 1.9,
};

// Split the condition on identifiers and tint every group name with its
// group's color; keywords, quantifiers and parentheses stay plain text.
function renderCondition(cond, colorOf) {
    return cond.split(/([A-Za-z_][A-Za-z0-9_]*)/g).map((part, i) => {
        const color = colorOf(part);
        if (!color) return part;
        return (
            <span
                key={i}
                style={{ color: color.fg, background: color.bg, borderRadius: 6, padding: '2px 6px' }}
            >{part}</span>
        );
    });
}

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
    const groupNames = entries.map(([gname]) => gname);
    const colorOf = (name) => {
        const idx = groupNames.indexOf(name);
        return idx === -1 ? null : GROUP_PALETTE[idx % GROUP_PALETTE.length];
    };

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
                    {entries.map(([gname, members]) => {
                        const color = colorOf(gname);
                        return (
                            <div key={gname} style={groupBox}>
                                <span style={groupNamePill(color)}>{gname}</span>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                    {members.map((m, i) => (
                                        <span key={m.ce_id ?? `${m.name}-${i}`} style={memberChip(color)}>{m.name}</span>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div style={{ fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic' }}>{emptyHint}</div>
            )}
            {cond && (
                <code style={conditionStyle} title="Firing condition over the groups above">
                    {renderCondition(cond, colorOf)}
                </code>
            )}
        </div>
    );
}
