// GroupConditionEditor — the shared editor for a rule's firing logic under the
// groups + condition model. Renders:
//   * the group list — add / rename / delete groups, add / remove CEs per group
//   * a BANK of pieces: per group "all of them" / "K of them", plus the
//     operators "and", "or" and brackets (GAVEL rules have no "not")
//   * the firing condition as a row of pieces — click a bank piece to add it at
//     the end, or drag it to where it belongs; drag pieces to reorder; × removes
//   * a live check in plain words: what the rule fires on, or what is wrong
//
// The condition string stays the stored source of truth (and is validated
// again server-side): pieces are parsed from it on every render and every edit
// writes it back — see utils/conditionBuilder. A stored condition the pieces
// cannot show (it uses "not", or a stray word) stays editable as text.
//
// Controlled component. Value shape (ordered, editing-friendly):
//   groupList = [{ name: 'required', ceIds: [3, 7] }]
//   condition = 'all of required'
// Callers convert to the wire shape { gname: [ce_id] } on save and refuse to
// save while ruleLogic.validateEditorState reports a problem. `pool` is the CEs
// available to add: [{ ce_id, name }].

import { useRef, useState } from 'react';
import { FiPlus, FiTrash2, FiX, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';
import { isValidGroupName } from '../../utils/ruleLogic';
import {
    ALL, LOGIC_PIECES, conditionToPieces, piecesToCondition, pieceLabel, quantChoices,
    checkCondition, removeGroupPieces, renamePieces, clampGroupPieces,
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
const ghostBtn = {
    display: 'inline-flex', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
    background: 'rgba(148, 163, 184, 0.12)', color: '#cbd5e1',
    border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: 8,
    padding: '6px 12px', fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer',
};
const sectionLabel = { display: 'block', fontSize: '0.78rem', fontWeight: 700, color: '#94a3b8', marginBottom: 4 };
const hint = { margin: '0 0 8px', fontSize: '0.75rem', color: '#64748b' };
const smallLabel = {
    fontSize: '0.7rem', fontWeight: 700, color: '#94a3b8',
    textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0,
};
const bankBox = {
    border: '1px solid rgba(148, 163, 184, 0.2)', borderRadius: 10,
    background: 'rgba(2, 6, 23, 0.4)', padding: '8px 10px',
    display: 'flex', flexDirection: 'column', gap: 6, minWidth: 130,
};
// Group pieces are violet, operators and brackets amber — the same colours in
// the bank and in the condition row, so a piece is recognisable where it lands.
const pieceColors = (kind) => (kind === 'sel'
    ? { color: '#f8fafc', background: 'rgba(167, 139, 250, 0.18)', border: '1px solid #a78bfa' }
    : { color: '#fde68a', background: 'rgba(251, 191, 36, 0.12)', border: '1px solid rgba(251, 191, 36, 0.7)' });
const bankPiece = (kind) => ({
    ...pieceColors(kind), borderRadius: 8, padding: '4px 10px', fontSize: '0.78rem',
    fontWeight: 700, cursor: 'grab', textAlign: 'left', fontFamily: kind === 'sel' ? 'inherit' : 'monospace',
});
const conditionRow = (highlight) => ({
    display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6,
    minHeight: 54, padding: 10, borderRadius: 10,
    background: 'rgba(2, 6, 23, 0.55)',
    border: `1px dashed ${highlight ? '#a78bfa' : 'rgba(148, 163, 184, 0.35)'}`,
});
const conditionPiece = (kind, dropBefore) => ({
    ...pieceColors(kind), display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: '3px 6px 3px 10px', borderRadius: 8, fontSize: '0.8rem', fontWeight: 700,
    cursor: 'grab', whiteSpace: 'nowrap',
    boxShadow: dropBefore ? '-3px 0 0 0 #a78bfa' : 'none',
});
const textBox = {
    width: '100%', boxSizing: 'border-box', fontFamily: 'monospace',
    padding: '10px 12px', borderRadius: 10, outline: 'none',
    background: 'rgba(2, 6, 23, 0.6)', color: '#e2e8f0', fontSize: '0.85rem',
    border: '1px solid rgba(148, 163, 184, 0.25)',
};

// Pick a fresh default name for a new group: required, then option_1, option_2, …
function nextGroupName(groupList) {
    const taken = new Set(groupList.map((g) => g.name));
    if (!taken.has('required')) return 'required';
    for (let i = 1; ; i += 1) {
        const candidate = `option_${i}`;
        if (!taken.has(candidate)) return candidate;
    }
}

// What a bank piece says inside its group's box.
const bankText = (quant) => (quant === ALL ? 'all of them' : `${quant} of them`);

export default function GroupConditionEditor({ pool, groupList, condition, onChange }) {
    const ceName = (id) => (pool || []).find((c) => c.ce_id === id)?.name || `CE_${id}`;

    const parsed = conditionToPieces(condition);
    const textMode = parsed === null;          // e.g. a stored condition that uses "not"
    const pieces = parsed || [];
    const check = checkCondition(groupList, condition);

    // While a group's typed name is not a valid name (say, the field was
    // emptied), its pieces keep the last valid name; this remembers it so the
    // pieces follow once the name is valid again.
    const heldNames = useRef({});
    const dragged = useRef(null);              // { from: 'bank', piece } | { from: 'row', index }
    const [dropAt, setDropAt] = useState(null);

    const writePieces = (nextGroups, nextPieces) => {
        onChange({ groupList: nextGroups, condition: piecesToCondition(nextPieces) });
    };
    // Group edits carry the condition along (or leave the text untouched in text mode).
    const writeGroups = (nextGroups, transform) => {
        if (textMode) {
            onChange({ groupList: nextGroups, condition });
            return;
        }
        writePieces(nextGroups, transform ? transform(pieces) : pieces);
    };

    // ---- groups -------------------------------------------------------------
    const renameGroup = (idx, nextName) => {
        const next = groupList.map((g, i) => (i === idx ? { ...g, name: nextName } : g));
        const typed = nextName.trim();
        const current = heldNames.current[idx] ?? (groupList[idx]?.name || '').trim();
        if (!isValidGroupName(typed)) {
            if (isValidGroupName(current)) heldNames.current[idx] = current;
            writeGroups(next);
            return;
        }
        delete heldNames.current[idx];
        writeGroups(next, (ps) => (current && current !== typed ? renamePieces(ps, current, typed) : ps));
    };

    const deleteGroup = (idx) => {
        const name = (heldNames.current[idx] ?? (groupList[idx]?.name || '')).trim();
        heldNames.current = {};
        writeGroups(groupList.filter((_, i) => i !== idx), (ps) => removeGroupPieces(ps, name));
    };

    const addGroup = () => {
        writeGroups([...groupList, { name: nextGroupName(groupList), ceIds: [] }]);
    };

    // Adding or removing a CE can make "K of" impossible — clamp it.
    const setMembers = (idx, nextCeIds) => {
        const name = (groupList[idx]?.name || '').trim();
        const next = groupList.map((g, i) => (i === idx ? { ...g, ceIds: nextCeIds } : g));
        writeGroups(next, (ps) => clampGroupPieces(ps, name, nextCeIds.length));
    };

    // ---- pieces -------------------------------------------------------------
    const insertPiece = (piece, at = pieces.length) => {
        const next = [...pieces];
        next.splice(at, 0, piece);
        writePieces(groupList, next);
    };
    const movePiece = (from, to) => {
        if (from === to || from + 1 === to) return;
        const next = [...pieces];
        const [p] = next.splice(from, 1);
        next.splice(from < to ? to - 1 : to, 0, p);
        writePieces(groupList, next);
    };
    const removePiece = (i) => writePieces(groupList, pieces.filter((_, k) => k !== i));
    const clearAll = () => onChange({ groupList, condition: '' });

    // Native drag-and-drop. Dropping on a piece puts the dragged one before it;
    // dropping on the empty part of the row puts it at the end.
    const startDrag = (payload) => (e) => {
        dragged.current = payload;
        try {
            e.dataTransfer.setData('text/plain', payload.from === 'bank' ? pieceLabel(payload.piece) : '');
            e.dataTransfer.effectAllowed = payload.from === 'bank' ? 'copy' : 'move';
        } catch { /* some environments have no dataTransfer */ }
    };
    const over = (at) => (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropAt !== at) setDropAt(at);
    };
    const drop = (at) => (e) => {
        e.preventDefault();
        e.stopPropagation();
        const d = dragged.current;
        dragged.current = null;
        setDropAt(null);
        if (!d) return;
        if (d.from === 'bank') insertPiece(d.piece, at);
        else movePiece(d.index, at);
    };
    const endDrag = () => {
        dragged.current = null;
        setDropAt(null);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* ---- groups ---- */}
            {groupList.map((g, idx) => {
                const addable = (pool || []).filter((c) => !g.ceIds.includes(c.ce_id));
                const name = (g.name || '').trim();
                return (
                    <div key={idx} style={box}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            <span style={smallLabel}>Group</span>
                            <input
                                value={g.name}
                                aria-label={`Group ${idx + 1} name`}
                                onChange={(e) => renameGroup(idx, e.target.value)}
                                style={nameInput(isValidGroupName(name))}
                                placeholder="group_name"
                            />
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

            {textMode ? (
                /* ---- a stored condition the pieces cannot show ---- */
                <div>
                    <label style={sectionLabel} htmlFor="gce-condition-text">Firing condition</label>
                    <input
                        id="gce-condition-text"
                        value={condition}
                        onChange={(e) => onChange({ groupList, condition: e.target.value })}
                        aria-label="Condition text"
                        style={textBox}
                    />
                    <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: '#fbbf24', display: 'flex', gap: 6 }}>
                        <FiAlertTriangle size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>
                            This condition contains something the pieces cannot show — for example
                            <code> not</code>, which GAVEL rules do not allow. Fix the text, or clear it to
                            build the condition from pieces.
                        </span>
                    </p>
                    <button type="button" style={{ ...ghostBtn, marginTop: 8 }} onClick={clearAll}>
                        Clear and use pieces
                    </button>
                </div>
            ) : (
                <>
                    {/* ---- the bank ---- */}
                    <div>
                        <span style={sectionLabel}>Pieces</span>
                        <p style={hint}>
                            Click a piece to add it to the end of the firing condition, or drag it to where it belongs.
                        </p>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                            {groupList.map((g, idx) => {
                                const name = (g.name || '').trim();
                                if (!isValidGroupName(name)) return null;
                                const choices = quantChoices(g.ceIds.length);
                                return (
                                    <div key={idx} style={bankBox} role="group" aria-label={`Pieces for ${name}`}>
                                        <span style={{ ...smallLabel, fontFamily: 'monospace', textTransform: 'none' }}>{name}</span>
                                        {choices.length === 0 ? (
                                            <span style={{ fontSize: '0.75rem', color: '#64748b', fontStyle: 'italic' }}>Add CEs to this group first.</span>
                                        ) : choices.map((quant) => {
                                            const piece = { kind: 'sel', group: name, quant };
                                            return (
                                                <button
                                                    key={String(quant)}
                                                    type="button"
                                                    draggable
                                                    onDragStart={startDrag({ from: 'bank', piece })}
                                                    onDragEnd={endDrag}
                                                    onClick={() => insertPiece(piece)}
                                                    aria-label={`Add "${pieceLabel(piece)}"`}
                                                    style={bankPiece('sel')}
                                                >
                                                    {bankText(quant)}
                                                </button>
                                            );
                                        })}
                                    </div>
                                );
                            })}
                            <div style={bankBox} role="group" aria-label="Operators">
                                <span style={smallLabel}>operators</span>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                    {LOGIC_PIECES.map((kind) => (
                                        <button
                                            key={kind}
                                            type="button"
                                            draggable
                                            onDragStart={startDrag({ from: 'bank', piece: { kind } })}
                                            onDragEnd={endDrag}
                                            onClick={() => insertPiece({ kind })}
                                            aria-label={`Add "${kind}"`}
                                            style={bankPiece(kind)}
                                        >
                                            {kind}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ---- the firing condition ---- */}
                    <div>
                        <span style={sectionLabel}>Firing condition</span>
                        <div
                            role="list"
                            aria-label="Firing condition pieces"
                            style={conditionRow(dropAt === pieces.length)}
                            onDragOver={over(pieces.length)}
                            onDragLeave={() => setDropAt(null)}
                            onDrop={drop(pieces.length)}
                        >
                            {pieces.length === 0 && (
                                <span style={{ fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic' }}>
                                    Click or drag pieces here.
                                </span>
                            )}
                            {pieces.map((p, i) => (
                                <span
                                    key={i}
                                    role="listitem"
                                    draggable
                                    onDragStart={startDrag({ from: 'row', index: i })}
                                    onDragEnd={endDrag}
                                    onDragOver={over(i)}
                                    onDrop={drop(i)}
                                    style={conditionPiece(p.kind, dropAt === i)}
                                >
                                    {pieceLabel(p)}
                                    <button
                                        type="button"
                                        title="Remove this piece"
                                        aria-label={`Remove piece ${i + 1} ("${pieceLabel(p)}")`}
                                        onClick={() => removePiece(i)}
                                        style={{ ...iconBtn, color: 'inherit', opacity: 0.75 }}
                                    >
                                        <FiX size={12} />
                                    </button>
                                </span>
                            ))}
                        </div>
                        {pieces.length > 0 && (
                            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                                <button type="button" style={ghostBtn} onClick={() => removePiece(pieces.length - 1)}>
                                    Remove last
                                </button>
                                <button type="button" style={ghostBtn} onClick={clearAll}>
                                    Clear
                                </button>
                            </div>
                        )}
                    </div>
                </>
            )}

            {/* ---- live check ---- */}
            <div aria-live="polite">
                {check.problems.length === 0 ? (
                    <p style={{ margin: 0, fontSize: '0.82rem', color: '#4ade80', display: 'flex', gap: 6 }}>
                        <FiCheckCircle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>Fires when: {check.reading}</span>
                    </p>
                ) : (
                    <ul aria-label="Condition problems" style={{ margin: 0, paddingLeft: 18, fontSize: '0.78rem', color: '#fbbf24' }}>
                        {check.problems.map((problem) => <li key={problem}>{problem}</li>)}
                    </ul>
                )}
                {!textMode && (
                    <p style={{ margin: '6px 0 0', fontSize: '0.72rem', color: '#64748b' }}>
                        Saved as: <code aria-label="Condition text">{condition || '—'}</code>
                    </p>
                )}
            </div>
        </div>
    );
}
