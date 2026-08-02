// Structured view of a rule's firing condition, so the UI can offer controls
// instead of free text.
//
// The v2 grammar (backend utils/rule_condition.py) allows nesting, parentheses
// and `or`/`and` mixing. The UI covers the FLAT subset — a list of one
// selector per group joined by a single operator:
//
//     all of lure_creation and 1 of info_solicitation
//
// That is every condition in the public library today, and it is what a
// picker can represent without lying about what the user chose. Anything
// outside it (parentheses, mixed and/or, a group selected twice, `not`)
// returns null from parseCondition, and the caller falls back to editing the
// raw text rather than silently rewriting logic it cannot model.
//
// The condition STRING stays the single source of truth: quantifiers are
// derived from it on every render and every edit rebuilds it. Nothing has to
// be kept in sync.

import { GROUP_NAME_RE } from './ruleLogic';

const KEYWORDS = new Set(['all', 'of', 'and', 'or', 'not']);

// A group is either absent from the condition (supporting-only), required in
// full, or required to match at least k of its members.
export const ALL = 'all';
export const UNUSED = null;

function tokenize(condition) {
    // Parentheses are not part of the flat subset, but they must still tokenize
    // so we can detect and reject them rather than mis-parsing.
    const raw = String(condition || '').replace(/\(/g, ' ( ').replace(/\)/g, ' ) ').trim();
    if (!raw) return [];
    const tokens = raw.split(/\s+/);
    for (const t of tokens) {
        const ok = KEYWORDS.has(t) || t === '(' || t === ')'
            || /^\d+$/.test(t) || GROUP_NAME_RE.test(t);
        if (!ok) return null;
    }
    return tokens;
}

/**
 * Decompose a condition into { joiner, quants } — or null when it falls
 * outside the flat subset the UI can represent.
 *
 * quants maps group name -> ALL | positive integer. Groups absent from the
 * condition are simply absent from the map.
 */
export function parseCondition(condition) {
    const tokens = tokenize(condition);
    if (tokens === null || tokens.length === 0) return null;
    if (tokens.includes('(') || tokens.includes(')') || tokens.includes('not')) return null;

    const quants = {};
    let joiner = null;
    let i = 0;

    for (;;) {
        // selector ::= ( "all" | INTEGER ) "of" GROUP_NAME
        const quantTok = tokens[i];
        if (quantTok === undefined) return null;
        let quant;
        if (quantTok === ALL) {
            quant = ALL;
        } else if (/^\d+$/.test(quantTok)) {
            quant = parseInt(quantTok, 10);
            if (quant < 1) return null;
        } else {
            return null;
        }
        if (tokens[i + 1] !== 'of') return null;
        const group = tokens[i + 2];
        if (!group || KEYWORDS.has(group) || !GROUP_NAME_RE.test(group)) return null;
        // The picker offers one control per group, so it cannot express a group
        // appearing twice.
        if (Object.prototype.hasOwnProperty.call(quants, group)) return null;
        quants[group] = quant;
        i += 3;

        if (i === tokens.length) break;
        const op = tokens[i];
        if (op !== 'and' && op !== 'or') return null;
        // A single joiner throughout: `a and b or c` means (a and b) or c,
        // which the flat model would misrepresent.
        if (joiner !== null && joiner !== op) return null;
        joiner = op;
        i += 1;
    }

    return { joiner: joiner || 'and', quants };
}

/**
 * Render the condition string for a set of quantifiers, in groupList order so
 * the text tracks what the user sees on screen.
 */
export function buildCondition(groupList, joiner, quants) {
    const parts = [];
    (groupList || []).forEach((g) => {
        const name = (g?.name || '').trim();
        if (!name) return;
        const quant = quants?.[name];
        if (quant === UNUSED || quant === undefined) return;
        parts.push(`${quant === ALL ? ALL : quant} of ${name}`);
    });
    return parts.join(` ${joiner === 'or' ? 'or' : 'and'} `);
}

/**
 * The editor's working model for a (groupList, condition) pair.
 *
 * `representable` is false when the condition uses grammar the picker cannot
 * express — the caller keeps the raw text in that case instead of rewriting it.
 *
 * An empty condition means "no group is required yet", NOT "everything is
 * required": setting the last group to supporting-only empties the string, and
 * defaulting that back to all-required would snap the pickers to the opposite
 * of what the user just chose. Callers that want a seeded rule pass a seeded
 * condition (the build wizard seeds `all of required`).
 */
export function toEditorModel(groupList, condition) {
    const groups = (groupList || []).map((g) => (g?.name || '').trim()).filter(Boolean);
    const hasText = String(condition || '').trim().length > 0;

    if (!hasText) return { representable: true, joiner: 'and', quants: {} };

    const parsed = parseCondition(condition);
    if (!parsed) return { representable: false, joiner: 'and', quants: {} };

    // A condition naming a group that no longer exists is stale rather than
    // advanced — drop it so the picker still opens, and let the rebuilt string
    // reflect the groups that are actually there.
    const quants = {};
    Object.entries(parsed.quants).forEach(([name, quant]) => {
        if (groups.includes(name)) quants[name] = quant;
    });
    return { representable: true, joiner: parsed.joiner, quants };
}

/**
 * Clamp a group's quantifier to its member count. "at least 3" in a two-member
 * group can never fire, and the library's validator rejects it outright.
 */
export function clampQuant(quant, memberCount) {
    if (quant === ALL || quant === UNUSED || quant === undefined) return quant;
    const n = Number(quant);
    if (!Number.isFinite(n) || n < 1) return 1;
    if (memberCount > 0 && n > memberCount) return ALL;
    return n;
}

/**
 * Human-readable summary of what the condition means, for the picker's header.
 */
export function describeCondition(groupList, joiner, quants) {
    const used = (groupList || []).filter((g) => {
        const q = quants?.[(g?.name || '').trim()];
        return q !== UNUSED && q !== undefined;
    });
    if (used.length === 0) return 'This rule never fires — no group is required yet.';
    if (used.length === 1) return 'The rule fires when this group is satisfied.';
    return joiner === 'or'
        ? 'The rule fires when ANY of these groups is satisfied.'
        : 'The rule fires when ALL of these groups are satisfied.';
}
