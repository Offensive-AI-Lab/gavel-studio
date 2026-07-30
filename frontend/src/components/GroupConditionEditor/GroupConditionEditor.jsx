// GroupConditionEditor — the minimal shared editor for a rule's firing logic
// under the groups + condition model. Renders:
//   * a group list — add / rename / delete groups, add / remove CEs per group
//   * a monospace condition input written over the group names
//
// Controlled component. Value shape (ordered, editing-friendly):
//   groupList = [{ name: 'required', ceIds: [3, 7] }]
//   condition = 'all of required'
// Callers convert to the wire shape { gname: [ce_id] } on save; the backend
// validates the condition authoritatively (utils/rule_condition) and the
// caller surfaces its 400/409 error text. `pool` is the CEs available to add:
// [{ ce_id, name }].

import { FiPlus, FiTrash2, FiX } from 'react-icons/fi';
import { isValidGroupName } from '../../utils/ruleLogic';

const box = {
    border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: 12,
    background: 'rgba(15, 23, 42, 0.5)', padding: 12,
    display: 'flex', flexDirection: 'column', gap: 8,
};
const nameInput = (valid) => ({
    fontFamily: 'monospace', fontSize: '0.85rem', fontWeight: 700,
    padding: '6px 10px', borderRadius: 8, outline: 'none',
    background: 'rgba(2, 6, 23, 0.55)', color: '#e2e8f0',
    border: `1px solid ${valid ? 'rgba(148, 163, 184, 0.25)' : 'rgba(248, 113, 113, 0.65)'}`,
    minWidth: 0, flex: 1,
});
const chip = {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    padding: '3px 8px', borderRadius: 999, fontSize: '0.78rem', fontWeight: 700,
    color: '#f8fafc', background: 'rgba(167, 139, 250, 0.15)',
    border: '1px solid #a78bfa', whiteSpace: 'nowrap',
};
const iconBtn = {
    display: 'inline-flex', alignItems: 'center', background: 'none',
    border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 2,
};
const selectStyle = {
    padding: '5px 8px', borderRadius: 8, fontSize: '0.78rem',
    background: 'rgba(2, 6, 23, 0.55)', color: '#cbd5e1',
    border: '1px solid rgba(148, 163, 184, 0.25)', maxWidth: 220,
};
const ghostBtn = {
    display: 'inline-flex', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
    background: 'rgba(148, 163, 184, 0.12)', color: '#cbd5e1',
    border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: 8,
    padding: '6px 12px', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer',
};

// Pick a fresh default name for a new group: option_1, option_2, …
function nextGroupName(groupList) {
    const taken = new Set(groupList.map((g) => g.name));
    if (!taken.has('required')) return 'required';
    for (let i = 1; ; i += 1) {
        const candidate = `option_${i}`;
        if (!taken.has(candidate)) return candidate;
    }
}

export default function GroupConditionEditor({ pool, groupList, condition, onChange }) {
    const ceName = (id) => (pool || []).find((c) => c.ce_id === id)?.name || `CE_${id}`;

    const patch = (nextGroups, nextCondition = condition) =>
        onChange({ groupList: nextGroups, condition: nextCondition });

    const updateGroup = (idx, updater) => {
        const next = groupList.map((g, i) => (i === idx ? updater(g) : g));
        patch(next);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {groupList.map((g, idx) => {
                const addable = (pool || []).filter((c) => !g.ceIds.includes(c.ce_id));
                return (
                    <div key={idx} style={box}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>Group</span>
                            <input
                                value={g.name}
                                aria-label={`Group ${idx + 1} name`}
                                onChange={(e) => updateGroup(idx, (grp) => ({ ...grp, name: e.target.value }))}
                                style={nameInput(isValidGroupName(g.name.trim()))}
                                placeholder="group_name"
                            />
                            <button
                                type="button"
                                title="Delete this group"
                                aria-label={`Delete group ${g.name || idx + 1}`}
                                onClick={() => patch(groupList.filter((_, i) => i !== idx))}
                                style={{ ...iconBtn, color: '#f87171' }}
                            >
                                <FiTrash2 size={15} />
                            </button>
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                            {g.ceIds.length === 0 && (
                                <span style={{ fontSize: '0.78rem', color: '#64748b', fontStyle: 'italic' }}>Empty — add CEs below.</span>
                            )}
                            {g.ceIds.map((id) => (
                                <span key={id} style={chip}>
                                    {ceName(id)}
                                    <button
                                        type="button"
                                        title={`Remove ${ceName(id)} from ${g.name}`}
                                        aria-label={`Remove ${ceName(id)} from ${g.name}`}
                                        onClick={() => updateGroup(idx, (grp) => ({ ...grp, ceIds: grp.ceIds.filter((x) => x !== id) }))}
                                        style={iconBtn}
                                    >
                                        <FiX size={12} />
                                    </button>
                                </span>
                            ))}
                            {addable.length > 0 && (
                                <select
                                    value=""
                                    aria-label={`Add a CE to ${g.name}`}
                                    onChange={(e) => {
                                        const id = parseInt(e.target.value, 10);
                                        if (!Number.isNaN(id)) updateGroup(idx, (grp) => ({ ...grp, ceIds: [...grp.ceIds, id] }));
                                    }}
                                    style={selectStyle}
                                >
                                    <option value="" disabled>+ Add CE…</option>
                                    {addable.map((c) => (
                                        <option key={c.ce_id} value={c.ce_id}>{c.name}</option>
                                    ))}
                                </select>
                            )}
                        </div>
                    </div>
                );
            })}

            <button
                type="button"
                style={ghostBtn}
                onClick={() => patch([...groupList, { name: nextGroupName(groupList), ceIds: [] }])}
            >
                <FiPlus size={13} /> Add group
            </button>

            <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>
                    Firing condition
                </label>
                <input
                    value={condition}
                    onChange={(e) => patch(groupList, e.target.value)}
                    placeholder='e.g. all of required and 1 of option_1'
                    aria-label="Firing condition"
                    style={{
                        width: '100%', boxSizing: 'border-box', fontFamily: 'monospace',
                        padding: '10px 12px', borderRadius: 10, outline: 'none',
                        background: 'rgba(2, 6, 23, 0.6)', color: '#e2e8f0', fontSize: '0.85rem',
                        border: '1px solid rgba(148, 163, 184, 0.25)',
                    }}
                />
                <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: '#64748b' }}>
                    Written over the group names: <code>all of g</code>, <code>2 of g</code>, combined with{' '}
                    <code>and</code> / <code>or</code> and parentheses. Groups left out of the condition are
                    supporting-only. Validated when you save.
                </p>
            </div>
        </div>
    );
}
