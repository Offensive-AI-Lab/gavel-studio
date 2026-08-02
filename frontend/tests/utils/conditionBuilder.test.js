// The condition string is what gets stored and what the library's validator
// judges, so these tests pin the round trip: every shape the pickers can build
// must parse back to the same choices, and every shape they CANNOT model must
// be reported as such rather than silently rewritten.

import { describe, it, expect } from 'vitest';
import {
    ALL, UNUSED, parseCondition, buildCondition, toEditorModel, clampQuant,
    describeCondition,
} from '../../src/utils/conditionBuilder';

const groups = (...names) => names.map((name) => ({ name, ceIds: [1, 2] }));

describe('parseCondition — the flat subset', () => {
    it('reads a single all-of selector', () => {
        expect(parseCondition('all of required'))
            .toEqual({ joiner: 'and', quants: { required: ALL } });
    });

    it('reads a numeric quantifier', () => {
        expect(parseCondition('2 of aggression'))
            .toEqual({ joiner: 'and', quants: { aggression: 2 } });
    });

    it('reads the library’s dominant shape', () => {
        // Every rule in gavel-rules today looks like this.
        expect(parseCondition('all of lure_creation and 1 of info_solicitation'))
            .toEqual({ joiner: 'and', quants: { lure_creation: ALL, info_solicitation: 1 } });
    });

    it('reads three and-joined selectors', () => {
        expect(parseCondition('all of a and 1 of b and 1 of c')).toEqual({
            joiner: 'and', quants: { a: ALL, b: 1, c: 1 },
        });
    });

    it('reads an or-joined condition', () => {
        expect(parseCondition('1 of a or 1 of b'))
            .toEqual({ joiner: 'or', quants: { a: 1, b: 1 } });
    });
});

describe('parseCondition — what the pickers cannot represent', () => {
    it.each([
        ['parentheses', '(all of a or 1 of b) and 1 of c'],
        ['mixed and/or', 'all of a and 1 of b or 1 of c'],
        ['negation', 'not all of a'],
        ['a group used twice', '1 of a and 2 of a'],
        ['a missing quantifier', 'of a'],
        ['a missing "of"', 'all a'],
        ['a dangling joiner', 'all of a and'],
        ['a bare group name', 'required'],
        ['zero as a quantifier', '0 of a'],
        ['an uppercase group name', 'all of Required'],
        ['empty text', '   '],
    ])('rejects %s', (_label, condition) => {
        expect(parseCondition(condition)).toBeNull();
    });
});

describe('buildCondition', () => {
    it('emits nothing when no group is required', () => {
        expect(buildCondition(groups('a', 'b'), 'and', {})).toBe('');
    });

    it('emits a single selector without a joiner', () => {
        expect(buildCondition(groups('a'), 'and', { a: ALL })).toBe('all of a');
    });

    it('joins selectors in the order the groups appear on screen', () => {
        expect(buildCondition(groups('first', 'second'), 'and',
            { second: 1, first: ALL })).toBe('all of first and 1 of second');
    });

    it('honours the or joiner', () => {
        expect(buildCondition(groups('a', 'b'), 'or', { a: 1, b: 2 }))
            .toBe('1 of a or 2 of b');
    });

    it('omits supporting-only groups', () => {
        expect(buildCondition(groups('a', 'b'), 'and', { a: ALL, b: UNUSED }))
            .toBe('all of a');
    });

    it('skips unnamed groups rather than emitting broken syntax', () => {
        expect(buildCondition([{ name: '  ', ceIds: [] }, { name: 'b', ceIds: [1] }],
            'and', { b: ALL })).toBe('all of b');
    });

    it('round-trips everything parseCondition accepts', () => {
        for (const text of [
            'all of required',
            '2 of aggression',
            'all of lure_creation and 1 of info_solicitation',
            '1 of a or 1 of b',
        ]) {
            const parsed = parseCondition(text);
            const names = Object.keys(parsed.quants).map((name) => ({ name, ceIds: [1, 2] }));
            expect(buildCondition(names, parsed.joiner, parsed.quants)).toBe(text);
        }
    });
});

describe('toEditorModel', () => {
    it('treats an empty condition as "nothing required yet"', () => {
        // Not "everything required" — clearing the last requirement must not
        // snap every picker back to all-of.
        expect(toEditorModel(groups('a', 'b'), '')).toEqual({
            representable: true, joiner: 'and', quants: {},
        });
    });

    it('hydrates the pickers from an existing condition', () => {
        expect(toEditorModel(groups('a', 'b'), 'all of a and 1 of b')).toEqual({
            representable: true, joiner: 'and', quants: { a: ALL, b: 1 },
        });
    });

    it('treats a group absent from the condition as supporting-only', () => {
        const model = toEditorModel(groups('a', 'b'), 'all of a');
        expect(model.quants.b).toBeUndefined();
        expect(model.representable).toBe(true);
    });

    it('flags advanced logic instead of rewriting it', () => {
        const model = toEditorModel(groups('a', 'b', 'c'), '(all of a or 1 of b) and 1 of c');
        expect(model.representable).toBe(false);
    });

    it('drops a reference to a group that no longer exists', () => {
        // The old free-text field allowed naming groups that were never created.
        const model = toEditorModel(groups('required'), 'all of required and 1 of ghost');
        expect(model.representable).toBe(true);
        expect(model.quants).toEqual({ required: ALL });
        expect(buildCondition(groups('required'), model.joiner, model.quants))
            .toBe('all of required');
    });

    it('survives a null group list', () => {
        expect(toEditorModel(null, '').quants).toEqual({});
    });
});

describe('clampQuant', () => {
    it('leaves all-of and supporting-only alone', () => {
        expect(clampQuant(ALL, 3)).toBe(ALL);
        expect(clampQuant(UNUSED, 3)).toBe(UNUSED);
    });

    it('keeps a quantifier that fits', () => {
        expect(clampQuant(2, 3)).toBe(2);
    });

    it('falls back to all-of when the group shrank below k', () => {
        // "at least 3 of a 2-member group" can never fire; the validator rejects it.
        expect(clampQuant(3, 2)).toBe(ALL);
    });

    it('floors at 1', () => {
        expect(clampQuant(0, 3)).toBe(1);
        expect(clampQuant(-2, 3)).toBe(1);
    });
});

describe('describeCondition', () => {
    it('warns when nothing would ever fire', () => {
        expect(describeCondition(groups('a'), 'and', {})).toMatch(/never fires/i);
    });

    it('describes and / or in words', () => {
        expect(describeCondition(groups('a', 'b'), 'and', { a: ALL, b: 1 })).toMatch(/ALL/);
        expect(describeCondition(groups('a', 'b'), 'or', { a: ALL, b: 1 })).toMatch(/ANY/);
    });

    it('does not talk about combining when only one group is used', () => {
        expect(describeCondition(groups('a', 'b'), 'and', { a: ALL }))
            .toMatch(/this group/i);
    });
});
