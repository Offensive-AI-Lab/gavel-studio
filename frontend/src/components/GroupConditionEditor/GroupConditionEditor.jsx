// GroupConditionEditor — the minimal shared editor for a rule's firing logic
// under the groups + condition model. Renders:
//   * a group list — add / rename / delete groups, add / remove CEs per group
//   * a per-group quantifier picker (all / at least k / not used)
//   * an and/or joiner, and the generated condition as READ-ONLY text
//
// The condition used to be free text, which let users name groups that did not
// exist and write logic the validator would reject only on save. It is now
// derived from the pickers: the string stays the stored source of truth (and is
// still validated server-side), but it is written by the UI, not typed.
//
// Conditions outside the flat "one selector per group" subset (parentheses,
// mixed and/or, `not`) cannot be represented by the pickers. Rather than
// rewrite logic we cannot model, those fall back to editing the raw text —
// see utils/conditionBuilder.
//
// Controlled component. Value shape (ordered, editing-friendly):
//   groupList = [{ name: 'required', ceIds: [3, 7] }]
//   condition = 'all of required'
// Callers convert to the wire shape { gname: [ce_id] } on save; the backend
// validates the condition authoritatively (utils/rule_condition) and the
// caller surfaces its 400/409 error text. `pool` is the CEs available to add:
// [{ ce_id, name }].

import { FiPlus, FiTrash2, FiX, FiAlertTriangle } from 'react-icons/fi';
import { isValidGroupName } from '../../utils/ruleLogic';
import {
    ALL, UNUSED, toEditorModel, buildCondition, clampQuant, describeCondition,
} from '../../utils/conditionBuilder';

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
const quantSelectStyle = { ...selectStyle, maxWidth: 170, flexShrink: 0 };
const ghostBtn = {
    display: 'inline-flex', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
    background: 'rgba(148, 163, 184, 0.12)', color: '#cbd5e1',
    border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: 8,
    padding: '6px 12px', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer',
};
const conditionBox = (readOnly) => ({
    width: '100%', boxSizing: 'border-box', fontFamily: 'monospace',
    padding: '10px 12px', borderRadius: 10, outline: 'none',
    background: readOnly ? 'rgba(2, 6, 23, 0.75)' : 'rgba(2, 6, 23, 0.6)',
    color: readOnly ? '#cbd5e1' : '#e2e8f0', fontSize: '0.85rem',
    border: '1px solid rgba(148, 163, 184, 0.25)',
    cursor: readOnly ? 'default' : 'text',
});

// Pick a fresh default name for a new group: option_1, option_2, …
function nextGroupName(groupList) {
    const taken = new Set(groupList.map((g) => g.name));
    if (!taken.has('required')) return 'required';
    for (let i = 1; ; i += 1) {
        const candidate = `option_${i}`;
        if (!taken.has(candidate)) return candidate;
    }
}

// Value used by the per-group <select>. Serialized because option values are
// strings and we need to distinguish "all", "not used" and a member count.
const quantValue = (quant) => {
    if (quant === UNUSED || quant === undefined) return 'unused';
    return quant === ALL ? ALL : String(quant);
};
const quantFromValue = (value) => {
    if (value === 'unused') return UNUSED;
    if (value === ALL) return ALL;
    const n = parseInt(value, 10);
    return Number.isNaN(n) ? ALL : n;
};

export default function GroupConditionEditor({ pool, groupList, condition, onChange }) {
    const ceName = (id) => (pool || []).find((c) => c.ce_id === id)?.name || `CE_${id}`;

    const model = toEditorModel(groupList, condition);
    const { representable, joiner, quants } = model;

    // Every edit rebuilds the condition from the pickers, so the text can never
    // drift from what the controls show. In the fallback (advanced) mode the
    // condition is passed through untouched.
    const patch = (nextGroups, nextQuants = quants, nextJoiner = joiner) => {
        if (!representable) {
            onChange({ groupList: nextGroups, condition });
            return;
        }
        onChange({
            groupList: nextGroups,
            condition: buildCondition(nextGroups, nextJoiner, nextQuants),
        });
    };

    const updateGroup = (idx, updater, quantsForNext) => {
        const next = groupList.map((g, i) => (i === idx ? updater(g) : g));
        patch(next, quantsForNext);
    };

    // Renaming carries the group's quantifier across, so the condition follows
    // the new name instead of quietly dropping the requirement.
    const renameGroup = (idx, nextName) => {
        const oldName = (groupList[idx]?.name || '').trim();
        const trimmed = nextName.trim();
        const nextQuants = { ...quants };
        if (oldName !== trimmed) {
            const carried = nextQuants[oldName];
            delete nextQuants[oldName];
            if (trimmed) nextQuants[trimmed] = carried === undefined ? ALL : carried;
        }
        updateGroup(idx, (grp) => ({ ...grp, name: nextName }), nextQuants);
    };

    const deleteGroup = (idx) => {
        const name = (groupList[idx]?.name || '').trim();
        const nextQuants = { ...quants };
        delete nextQuants[name];
        patch(groupList.filter((_, i) => i !== idx), nextQuants);
    };

    const addGroup = () => {
        const name = nextGroupName(groupList);
        patch([...groupList, { name, ceIds: [] }], { ...quants, [name]: ALL });
    };

    // Adding or removing a CE can invalidate "at least k" — k must stay within
    // the group's size or the rule can never fire.
    const setMembers = (idx, nextCeIds) => {
        const name = (groupList[idx]?.name || '').trim();
        const nextQuants = { ...quants };
        if (name in nextQuants) {
            nextQuants[name] = clampQuant(nextQuants[name], nextCeIds.length);
        }
        updateGroup(idx, (grp) => ({ ...grp, ceIds: nextCeIds }), nextQuants);
    };

    const setQuant = (name, value) => {
        patch(groupList, { ...quants, [name]: quantFromValue(value) });
    };

    const usedCount = (groupList || []).filter((g) => {
        const q = quants[(g?.name || '').trim()];
        return q !== UNUSED && q !== undefined;
    }).length;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {groupList.map((g, idx) => {
                const addable = (pool || []).filter((c) => !g.ceIds.includes(c.ce_id));
                const name = (g.name || '').trim();
                const quant = quants[name];
                const size = g.ceIds.length;
                return (
                    <div key={idx} style={box}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>Group</span>
                            <input
                                value={g.name}
                                aria-label={`Group ${idx + 1} name`}
                                onChange={(e) => renameGroup(idx, e.target.value)}
                                style={nameInput(isValidGroupName(name))}
                                placeholder="group_name"
                            />
                            {representable && (
                                <select
                                    value={quantValue(quant)}
                                    aria-label={`How much of ${name || `group ${idx + 1}`} must match`}
                                    onChange={(e) => setQuant(name, e.target.value)}
                                    style={quantSelectStyle}
                                    disabled={!name}
                                >
                                    <option value={ALL}>Needs all of them</option>
                                    {Array.from({ length: Math.max(size - 1, 0) }, (_, i) => i + 1).map((k) => (
                                        <option key={k} value={String(k)}>
                                            {k === 1 ? 'Needs any 1' : `Needs any ${k}`}
                                        </option>
                                    ))}
                                    <option value="unused">Supporting only</option>
                                </select>
                            )}
                            <button
                                type="button"
                                title="Delete this group"
                                aria-label={`Delete group ${g.name || idx + 1}`}
                                onClick={() => deleteGroup(idx)}
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
                                        onClick={() => setMembers(idx, g.ceIds.filter((x) => x !== id))}
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
                                        if (!Number.isNaN(id)) setMembers(idx, [...g.ceIds, id]);
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

            <button type="button" style={ghostBtn} onClick={addGroup}>
                <FiPlus size={13} /> Add group
            </button>

            <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>
                    Firing condition
                </label>

                {representable && usedCount > 1 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>Fire when</span>
                        <select
                            value={joiner}
                            aria-label="Combine the groups with and / or"
                            onChange={(e) => patch(groupList, quants, e.target.value)}
                            style={selectStyle}
                        >
                            <option value="and">every group above is satisfied</option>
                            <option value="or">any group above is satisfied</option>
                        </select>
                    </div>
                )}

                <input
                    value={condition}
                    readOnly={representable}
                    onChange={representable ? undefined : (e) => onChange({ groupList, condition: e.target.value })}
                    placeholder={representable ? 'Set by the pickers above' : 'e.g. all of required and 1 of option_1'}
                    aria-label="Firing condition"
                    aria-readonly={representable ? 'true' : undefined}
                    tabIndex={representable ? -1 : undefined}
                    style={conditionBox(representable)}
                />

                {representable ? (
                    <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: '#64748b' }}>
                        Written for you from the choices above. {describeCondition(groupList, joiner, quants)}
                        {' '}Groups set to <em>supporting only</em> are not part of it.
                    </p>
                ) : (
                    <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: '#fbbf24', display: 'flex', gap: 6 }}>
                        <FiAlertTriangle size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>
                            This rule uses advanced logic (parentheses, mixed <code>and</code>/<code>or</code>,
                            or a repeated group) that the pickers cannot show, so it stays editable as text.
                            Clear it to switch back to the pickers.
                        </span>
                    </p>
                )}
            </div>
        </div>
    );
}
