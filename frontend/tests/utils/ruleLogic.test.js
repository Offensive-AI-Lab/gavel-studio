// Tests for utils/ruleLogic — helpers for the groups + condition model.
import { describe, it, expect } from 'vitest';
import {
    isValidGroupName,
    normalizeGroups,
    extractLogic,
    groupsToIdPayload,
    prettyName,
    groupProgressLabel,
    validateEditorState,
} from '../../src/utils/ruleLogic';

describe('isValidGroupName', () => {
    it('accepts [a-z][a-z0-9_]* names', () => {
        expect(isValidGroupName('required')).toBe(true);
        expect(isValidGroupName('option_1')).toBe(true);
        expect(isValidGroupName('g2x')).toBe(true);
    });
    it('rejects keywords, capitals, leading digits/underscores, empties', () => {
        ['all', 'of', 'and', 'or', 'not', 'Required', '1group', '_g', '', 'a b'].forEach((n) => {
            expect(isValidGroupName(n)).toBe(false);
        });
    });
});

describe('normalizeGroups', () => {
    it('normalizes string members to {ce_id: null, name}', () => {
        expect(normalizeGroups({ g: ['a'] })).toEqual({ g: [{ ce_id: null, name: 'a' }] });
    });
    it('passes through {ce_id, name} members', () => {
        expect(normalizeGroups({ g: [{ ce_id: 3, name: 'a' }] })).toEqual({ g: [{ ce_id: 3, name: 'a' }] });
    });
    it('returns {} for null/array/non-object input', () => {
        expect(normalizeGroups(null)).toEqual({});
        expect(normalizeGroups([])).toEqual({});
        expect(normalizeGroups('x')).toEqual({});
    });
});

describe('extractLogic', () => {
    it('reads groups/condition/predicate off a rule row', () => {
        const logic = extractLogic({
            groups: { required: ['a'] },
            condition: 'all of required',
            predicate: 'a',
        });
        expect(logic.groups).toEqual({ required: [{ ce_id: null, name: 'a' }] });
        expect(logic.condition).toBe('all of required');
        expect(logic.predicate).toBe('a');
    });
    it('accepts the ce_groups field name too', () => {
        const logic = extractLogic({ ce_groups: { g: ['a'] }, condition: 'all of g' });
        expect(logic.groups).toEqual({ g: [{ ce_id: null, name: 'a' }] });
    });
    it('returns null groups for legacy rows (predicate fallback)', () => {
        const logic = extractLogic({ predicate: 'a AND b' });
        expect(logic.groups).toBeNull();
        expect(logic.predicate).toBe('a AND b');
    });
    it('prefers a nested logic object over top-level fields', () => {
        // Backend rows (get_classifier_rules, GET /rules/{id}/detail) nest the
        // v2 logic under `logic`; stale top-level fields must not shadow it.
        const logic = extractLogic({
            logic: {
                groups: { required: [{ ce_id: 1, name: 'A' }] },
                condition: 'all of required',
                predicate: 'A',
            },
            groups: { stale: ['z'] },
            condition: 'all of stale',
            predicate: 'Z',
        });
        expect(logic.groups).toEqual({ required: [{ ce_id: 1, name: 'A' }] });
        expect(logic.condition).toBe('all of required');
        expect(logic.predicate).toBe('A');
    });
    it('treats a nested logic with empty groups as a legacy row', () => {
        // Rows without ce_groups come back as logic.groups = {} — the caller
        // must fall back to the derived predicate string.
        const logic = extractLogic({
            logic: { groups: {}, condition: '', predicate: 'A AND B' },
        });
        expect(logic.groups).toBeNull();
        expect(logic.predicate).toBe('A AND B');
    });
});

describe('groupsToIdPayload', () => {
    it('keeps only members with a ce_id', () => {
        expect(groupsToIdPayload({ g: [{ ce_id: 1, name: 'a' }, 'nameonly'] })).toEqual({ g: [1] });
    });
});

describe('prettyName', () => {
    it('replaces underscores with spaces', () => {
        expect(prettyName('directive_to_user')).toBe('directive to user');
    });
});

describe('groupProgressLabel', () => {
    it('formats "hits/needed of group" for all-quantified groups', () => {
        const info = { members: ['a', 'b', 'c'], hits: ['a', 'b'], quantifier: 'all', satisfied: false };
        expect(groupProgressLabel('solicited_action', info)).toBe('2/3 of solicited_action');
    });
    it('uses k for k-of quantifiers', () => {
        const info = { members: ['a', 'b', 'c'], hits: ['a'], quantifier: 2, satisfied: false };
        expect(groupProgressLabel('opts', info)).toBe('1/2 of opts');
    });
    it('marks unreferenced (supporting) groups', () => {
        const info = { members: ['a'], hits: [], quantifier: null, satisfied: true };
        expect(groupProgressLabel('supporting', info)).toBe('supporting (not in condition)');
    });
});

describe('validateEditorState', () => {
    const g = (name, ceIds = [1]) => ({ name, ceIds });
    it('passes a valid state', () => {
        expect(validateEditorState([g('required')], 'all of required')).toBeNull();
    });
    it('rejects empty group list', () => {
        expect(validateEditorState([], 'all of g')).toMatch(/at least one group/);
    });
    it('rejects invalid group names', () => {
        expect(validateEditorState([g('All')], 'x')).toMatch(/not a valid group name/);
    });
    it('rejects duplicate group names', () => {
        expect(validateEditorState([g('a'), g('a')], 'all of a')).toMatch(/Duplicate/);
    });
    it('rejects empty groups', () => {
        expect(validateEditorState([g('a', [])], 'all of a')).toMatch(/no cognitive elements/);
    });
    it('rejects a blank condition', () => {
        expect(validateEditorState([g('a')], '   ')).toMatch(/firing condition/);
    });
});
