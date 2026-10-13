// Firing-condition model for the rule editor's piece builder.
//
// The condition grammar (gavel-rules FORMAT.md; backend utils/rule_condition.py):
//
//     condition   ::= conjunction ( "or" conjunction )*
//     conjunction ::= atom ( "and" atom )*
//     atom        ::= selector | "(" condition ")"
//     selector    ::= ( "all" | K ) "of" group
//
// `and` binds tighter than `or`. GAVEL rules do not allow `not` (the library
// validator rejects it until its calibration semantics are defined), so the
// builder never offers it.
//
// The editor shows a condition as a row of PIECES — selectors ("all of
// scam_context", "1 of pressure_tactics"), operators (and / or) and brackets —
// that the user clicks or drags in from a bank. The condition STRING stays the
// single source of truth: pieces are parsed from it on every render and every
// edit writes it back.
//
// checkCondition() applies the same rules as the backend validator
// (utils/rule_condition.validate_condition), so the editor can say what is
// wrong while you build and the pages refuse to save what the backend would
// reject.

export const ALL = 'all';

const KEYWORDS = new Set(['all', 'of', 'and', 'or', 'not']);
// Same as ruleLogic.GROUP_NAME_RE (kept local: ruleLogic imports this module).
const NAME_RE = /^[a-z][a-z0-9_]*$/;
const OPERATORS = new Set(['and', 'or']);

// The operator/bracket pieces offered next to the groups (no `not`).
export const LOGIC_PIECES = ['and', 'or', '(', ')'];

function tokenize(condition) {
    const raw = String(condition || '').replace(/\(/g, ' ( ').replace(/\)/g, ' ) ').trim();
    if (!raw) return [];
    const tokens = raw.split(/\s+/);
    for (const t of tokens) {
        const ok = KEYWORDS.has(t) || t === '(' || t === ')' || /^\d+$/.test(t) || NAME_RE.test(t);
        if (!ok) return null;
    }
    return tokens;
}

/**
 * The condition as pieces:
 *   { kind: 'sel', group, quant }   quant: ALL | positive integer
 *   { kind: 'and' } | { kind: 'or' } | { kind: '(' } | { kind: ')' }
 * Pieces may be in any order (the user is still building); checkCondition
 * says whether the order is valid. Returns null when the text holds something
 * the builder cannot show as pieces — `not`, a stray word.
 */
export function conditionToPieces(condition) {
    const tokens = tokenize(condition);
    if (tokens === null) return null;
    const pieces = [];
    for (let i = 0; i < tokens.length;) {
        const t = tokens[i];
        if (OPERATORS.has(t) || t === '(' || t === ')') {
            pieces.push({ kind: t });
            i += 1;
        } else if (t === ALL || /^\d+$/.test(t)) {
            const quant = t === ALL ? ALL : parseInt(t, 10);
            const group = tokens[i + 1] === 'of' ? tokens[i + 2] : undefined;
            if (!group || KEYWORDS.has(group) || !NAME_RE.test(group) || (quant !== ALL && quant < 1)) return null;
            pieces.push({ kind: 'sel', group, quant });
            i += 3;
        } else {
            return null;
        }
    }
    return pieces;
}

/** Pieces back to condition text: `(all of a and 1 of b) or all of c`. */
export function piecesToCondition(pieces) {
    let out = '';
    (pieces || []).forEach((p) => {
        const text = p.kind === 'sel' ? `${p.quant === ALL ? ALL : p.quant} of ${p.group}` : p.kind;
        if (!out) out = text;
        else if (p.kind === ')' || out.endsWith('(')) out += text;
        else out += ` ${text}`;
    });
    return out;
}

/** What a piece says in the condition row. */
export function pieceLabel(piece) {
    if (piece.kind !== 'sel') return piece.kind;
    return `${piece.quant === ALL ? ALL : piece.quant} of ${piece.group}`;
}

/**
 * The selector choices a group offers: "all of them", then "1 of them" up to
 * one less than its size ("K of them" with K equal to the size must be written
 * "all of them" — the library's lint). A group without CEs offers none.
 */
export function quantChoices(size) {
    if (!(size > 0)) return [];
    return [ALL, ...Array.from({ length: size - 1 }, (_, i) => i + 1)];
}

/**
 * Clamp a quantifier to a group's member count. "3 of" a two-member group can
 * never fire, and "2 of" a two-member group has to be written "all of".
 */
export function clampQuant(quant, memberCount) {
    if (quant === ALL || quant === undefined) return quant;
    const n = Number(quant);
    if (!Number.isFinite(n) || n < 1) return 1;
    if (memberCount > 0 && n >= memberCount) return ALL;
    return n;
}

const isOp = (p) => !!p && OPERATORS.has(p.kind);

// Remove what an edit left dangling: empty brackets, an operator at either end,
// right after "(" or right before ")", and a doubled operator.
function tidy(pieces) {
    const out = [...pieces];
    for (let changed = true; changed;) {
        changed = false;
        for (let i = 0; i < out.length; i += 1) {
            const prev = out[i - 1];
            const next = out[i + 1];
            const p = out[i];
            const drop =
                (p.kind === '(' && next && next.kind === ')') ? 2
                    : (isOp(p) && (i === 0 || i === out.length - 1 || prev.kind === '(' || next.kind === ')' || isOp(next))) ? 1
                        : 0;
            if (drop) {
                out.splice(i, drop);
                changed = true;
                break;
            }
        }
    }
    return out;
}

/** Remove every piece of a deleted group, with the operator that joined it. */
export function removeGroupPieces(pieces, name) {
    const out = [...(pieces || [])];
    for (;;) {
        const i = out.findIndex((p) => p.kind === 'sel' && p.group === name);
        if (i === -1) break;
        if (isOp(out[i - 1])) out.splice(i - 1, 2);
        else if (isOp(out[i + 1])) out.splice(i, 2);
        else out.splice(i, 1);
    }
    return tidy(out);
}

/** A renamed group's pieces follow the new name. */
export function renamePieces(pieces, from, to) {
    return (pieces || []).map((p) => (p.kind === 'sel' && p.group === from ? { ...p, group: to } : p));
}

/** A group that shrank keeps quantifiers it can still satisfy. */
export function clampGroupPieces(pieces, name, size) {
    return (pieces || []).map((p) => (
        p.kind === 'sel' && p.group === name ? { ...p, quant: clampQuant(p.quant, size) } : p));
}

class PieceError extends Error {}

// Parse pieces into a tree, or explain the first thing in the way.
function parsePieces(pieces) {
    let i = 0;
    const quoted = (p) => `"${pieceLabel(p)}"`;
    const fail = (message) => { throw new PieceError(message); };

    function atom() {
        const p = pieces[i];
        const prev = pieces[i - 1];
        if (!p) {
            if (prev && prev.kind === '(') fail('A bracket is opened but never closed — add ")".');
            fail(`The condition ends with ${quoted(prev)} — add a piece after it.`);
        }
        if (p.kind === 'sel') {
            i += 1;
            return { kind: 'sel', group: p.group, quant: p.quant };
        }
        if (p.kind === '(') {
            i += 1;
            if (pieces[i] && pieces[i].kind === ')') fail('Empty brackets — put pieces inside them or remove them.');
            const inner = condition();
            if (!pieces[i] || pieces[i].kind !== ')') fail('A bracket is opened but never closed — add ")".');
            i += 1;
            return inner;
        }
        if (p.kind === ')') fail('A ")" has no matching "(".');
        // an operator where a group piece or "(" belongs
        if (!prev) fail(`The condition cannot start with ${quoted(p)}.`);
        if (isOp(prev)) fail(`${quoted(prev)} and ${quoted(p)} are next to each other — put a group piece between them.`);
        fail(`${quoted(p)} cannot come right after "(".`);
        return null;
    }
    function conjunction() {
        const kids = [atom()];
        while (pieces[i] && pieces[i].kind === 'and') {
            i += 1;
            kids.push(atom());
        }
        return kids.length === 1 ? kids[0] : { kind: 'and', kids };
    }
    function condition() {
        const kids = [conjunction()];
        while (pieces[i] && pieces[i].kind === 'or') {
            i += 1;
            kids.push(conjunction());
        }
        return kids.length === 1 ? kids[0] : { kind: 'or', kids };
    }

    try {
        const tree = condition();
        if (i < pieces.length) {
            const p = pieces[i];
            if (p.kind === ')') fail('A ")" has no matching "(".');
            fail(`Put "and" or "or" between ${quoted(pieces[i - 1])} and ${quoted(p)}.`);
        }
        return { tree };
    } catch (e) {
        if (e instanceof PieceError) return { error: e.message };
        throw e;
    }
}

// The condition in words, with brackets wherever "and" and "or" mix.
function describe(node, parentKind) {
    if (node.kind === 'sel') {
        return node.quant === ALL ? `all of ${node.group}` : `at least ${node.quant} of ${node.group}`;
    }
    const text = node.kids.map((k) => describe(k, node.kind)).join(node.kind === 'and' ? ' AND ' : ' OR ');
    return parentKind && parentKind !== node.kind ? `(${text})` : text;
}

/**
 * Check a (groups, condition) pair against GAVEL's rules — the same ones the
 * backend validator enforces: valid grammar, no `not`, only existing groups,
 * satisfiable quantifiers ("K of" a group of exactly K must be "all of"), no
 * empty group, and every group used. Group NAME rules are checked by
 * ruleLogic.validateEditorState.
 *
 * Returns { problems: [plain-language strings], reading: the condition in
 * words, or null when it does not parse }.
 */
export function checkCondition(groupList, condition) {
    const sizes = new Map();
    (groupList || []).forEach((g) => {
        const name = (g?.name || '').trim();
        if (name) sizes.set(name, (g.ceIds || []).length);
    });

    if (!String(condition || '').trim()) {
        return { problems: ['Nothing would make this rule fire yet — add pieces to the firing condition.'], reading: null };
    }
    const tokens = tokenize(condition);
    if (tokens && tokens.includes('not')) {
        return { problems: ['GAVEL rules cannot use "not" — remove it.'], reading: null };
    }
    const pieces = conditionToPieces(condition);
    if (!pieces) {
        return { problems: ['The condition contains something that is not a group piece, "and", "or" or a bracket.'], reading: null };
    }

    const problems = [];
    const { tree, error } = parsePieces(pieces);
    if (error) problems.push(error);

    const used = new Set();
    pieces.filter((p) => p.kind === 'sel').forEach((p) => {
        used.add(p.group);
        if (!sizes.has(p.group)) {
            problems.push(`"${p.group}" is not one of the groups.`);
            return;
        }
        const size = sizes.get(p.group);
        if (p.quant === ALL || size === 0) return;
        if (p.quant > size) {
            problems.push(`"${pieceLabel(p)}" can never happen — ${p.group} has only ${size} CE${size === 1 ? '' : 's'}.`);
        } else if (p.quant === size && p.quant > 1) {
            problems.push(`"${pieceLabel(p)}" means all of them — use "all of ${p.group}".`);
        }
    });
    sizes.forEach((size, name) => {
        if (size === 0) problems.push(`Group "${name}" has no CEs — add some or delete the group.`);
        if (!used.has(name)) problems.push(`Group "${name}" is not used by the condition — add one of its pieces or delete the group.`);
    });

    return { problems: [...new Set(problems)], reading: tree && !error ? describe(tree) : null };
}
