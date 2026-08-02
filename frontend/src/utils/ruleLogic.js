// Rule-logic helpers for the groups + condition model — the gavel-rules
// condition grammar (see FORMAT.md in the library repo).
//
// A rule's firing logic is:
//   * `groups`    — named CE groups: { group_name: [members] } where a member
//                   is either a CE name (string) or { ce_id, name }.
//   * `condition` — a condition-grammar expression over group names, e.g.
//                   "all of required and 1 of option_1". Rendered verbatim.
//   * `predicate` — a backend-derived display string (render_predicate output);
//                   kept only as a fallback for legacy rows without groups.
//
// The old role vocabulary (necessary / fallback / sufficient) is gone.

// Group names: [a-z][a-z0-9_]*, not a grammar keyword.
export const CONDITION_KEYWORDS = ['all', 'of', 'and', 'or', 'not'];
export const GROUP_NAME_RE = /^[a-z][a-z0-9_]*$/;
// Positional / non-descriptive group names the library's CI lint rejects.
export const LAZY_GROUP_NAME_RE = /^(g|grp|group|sel|selection)_?\d*$/;

export function isValidGroupName(name) {
    return typeof name === 'string'
        && GROUP_NAME_RE.test(name)
        && !CONDITION_KEYWORDS.includes(name);
}

// Normalize a groups object (members as strings OR {ce_id, name} objects)
// to { group_name: [{ ce_id, name }] }. Unknown ce_ids stay null.
export function normalizeGroups(groups) {
    const out = {};
    if (!groups || typeof groups !== 'object' || Array.isArray(groups)) return out;
    Object.entries(groups).forEach(([gname, members]) => {
        out[gname] = (Array.isArray(members) ? members : []).map((m) => {
            if (m && typeof m === 'object') {
                return { ce_id: m.ce_id ?? null, name: m.name ?? String(m.ce_id ?? '') };
            }
            return { ce_id: null, name: String(m) };
        });
    });
    return out;
}

// Pull { groups, condition, predicate } off any rule-shaped row. Backend
// responses nest the logic under a `logic` key ({groups: {g: [{ce_id,
// name}]}, condition, predicate} — get_classifier_rules rows, GET
// /rules/{id}/detail, public library / rule-set member rows); flatter
// shapes carry groups / ce_groups + condition at the top level. Members
// may be names or {ce_id, name}. Returns groups: null when the row
// predates the groups model (caller falls back to the predicate string).
export function extractLogic(rule) {
    const nested = rule && typeof rule.logic === 'object' && !Array.isArray(rule.logic)
        ? rule.logic
        : null;
    const raw = nested?.groups ?? rule?.groups ?? rule?.ce_groups ?? null;
    const groups = raw && typeof raw === 'object' && !Array.isArray(raw) && Object.keys(raw).length > 0
        ? normalizeGroups(raw)
        : null;
    return {
        groups,
        condition: nested?.condition || rule?.condition || '',
        predicate: nested?.predicate || rule?.predicate || '',
    };
}

// { gname: [{ce_id, name}] } -> { gname: [ce_id] } for request bodies.
// Members without a ce_id are dropped (the editor only handles known CEs).
export function groupsToIdPayload(groups) {
    const out = {};
    Object.entries(normalizeGroups(groups)).forEach(([gname, members]) => {
        out[gname] = members.map((m) => m.ce_id).filter((id) => id != null);
    });
    return out;
}

// Pretty-print an underscored identifier for display: "directive_to_user"
// -> "directive to user". Used for CE role badges and group headers.
export function prettyName(value) {
    return String(value || '').replaceAll('_', ' ');
}

// "2/3 of solicited_action" — how close a group got to satisfying its
// quantifier. `info` is a realtime group report: { members, hits, quantifier,
// satisfied }. quantifier is 'all', an integer, or null (unreferenced group).
export function groupProgressLabel(gname, info) {
    const members = info?.members || [];
    const hits = info?.hits || [];
    const quant = info?.quantifier;
    if (quant == null) return `${gname} (not in condition)`;
    const need = quant === 'all' ? members.length : Number(quant);
    return `${hits.length}/${need} of ${gname}`;
}

// Client-side sanity check for the group/condition editor. Server-side
// validation (utils/rule_condition.validate_condition) is authoritative —
// this only catches the obvious mistakes before a round-trip. Mirrors the
// gavel-rules CI invariants: descriptive group names, no negation, and no
// dead groups (every defined group must be referenced by the condition).
// `groupList` = [{ name, ceIds }], condition = string.
// Returns an error message or null.
export function validateEditorState(groupList, condition) {
    if (!Array.isArray(groupList) || groupList.length === 0) {
        return 'Add at least one group.';
    }
    const seen = new Set();
    for (const g of groupList) {
        const name = (g.name || '').trim();
        if (!isValidGroupName(name)) {
            return `"${name || '(empty)'}" is not a valid group name — use lowercase letters, digits and underscores, starting with a letter (and not a keyword like "all" or "of").`;
        }
        if (LAZY_GROUP_NAME_RE.test(name)) {
            return `"${name}" is not a descriptive group name — name the role the group plays in the detection (e.g. "aggression", "rapport_tactic"), not its position.`;
        }
        if (seen.has(name)) return `Duplicate group name "${name}".`;
        seen.add(name);
        if (!Array.isArray(g.ceIds) || g.ceIds.length === 0) {
            return `Group "${name}" has no cognitive elements — add some or delete the group.`;
        }
    }
    const cond = String(condition || '').trim();
    if (!cond) {
        return 'Write a firing condition, e.g. "all of required".';
    }
    const tokens = new Set(cond.split(/[()\s]+/).filter(Boolean));
    if (tokens.has('not')) {
        return 'Negation ("not") is not supported in rule conditions.';
    }
    for (const name of seen) {
        if (!tokens.has(name)) {
            return `Group "${name}" is not referenced by the condition — use it or delete the group.`;
        }
    }
    return null;
}
