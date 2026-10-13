// The condition string is what gets stored and what the library's validator
// judges. These tests pin the piece model the editor builds it from, and the
// live check: it must accept and reject exactly what the backend validator
// (utils/rule_condition.validate_condition) does, and say why in plain words.

import { describe, it, expect } from 'vitest';
import {
    ALL, LOGIC_PIECES, conditionToPieces, piecesToCondition, pieceLabel, quantChoices, clampQuant,
    checkCondition, removeGroupPieces, renamePieces, clampGroupPieces,
} from '../../src/utils/conditionBuilder';

const sel = (group, quant = ALL) => ({ kind: 'sel', group, quant });
const op = (kind) => ({ kind });
const groups = (spec) => Object.entries(spec).map(([name, size]) => ({
    name, ceIds: Array.from({ length: size }, (_, i) => i + 1),
}));

describe('conditionToPieces', () => {
    it('reads selectors and operators', () => {
        expect(conditionToPieces('all of a and 1 of b')).toEqual([sel('a'), op('and'), sel('b', 1)]);
    });

    it('reads brackets, with or without spaces around them', () => {
        const expected = [op('('), sel('a'), op('and'), sel('b', 1), op(')'), op('or'), sel('c')];
        expect(conditionToPieces('(all of a and 1 of b) or all of c')).toEqual(expected);
        expect(conditionToPieces('( all of a and 1 of b ) or all of c')).toEqual(expected);
    });

    it('keeps pieces that are still in the wrong order (the user is building)', () => {
        expect(conditionToPieces('and all of a or')).toEqual([op('and'), sel('a'), op('or')]);
    });

    it('reads nothing as no pieces', () => {
        expect(conditionToPieces('')).toEqual([]);
        expect(conditionToPieces(null)).toEqual([]);
    });

    it.each([
        ['"not" (GAVEL rules have none)', 'not all of a'],
        ['a stray word', 'all of a and banana'],
        ['a capitalised group name', 'all of Scam'],
        ['a selector without "of"', 'all a'],
        ['"0 of"', '0 of a'],
    ])('cannot show %s', (_, text) => {
        expect(conditionToPieces(text)).toBeNull();
    });
});

describe('piecesToCondition', () => {
    it('writes brackets tight against what they enclose', () => {
        expect(piecesToCondition([op('('), sel('a'), op('and'), sel('b', 1), op(')'), op('or'), sel('c')]))
            .toBe('(all of a and 1 of b) or all of c');
    });

    it('writes nothing for no pieces', () => {
        expect(piecesToCondition([])).toBe('');
    });

    it.each([
        'all of a',
        'all of a and 1 of b or all of c',
        '(all of a or 1 of b) and 2 of c',
        '((all of a and all of b) or all of c) and 1 of d',
    ])('round-trips %s', (text) => {
        expect(piecesToCondition(conditionToPieces(text))).toBe(text);
    });
});

describe('pieces offered by the bank', () => {
    it('labels pieces the way the condition reads', () => {
        expect(pieceLabel(sel('scam_context'))).toBe('all of scam_context');
        expect(pieceLabel(sel('pressure_tactics', 2))).toBe('2 of pressure_tactics');
        expect(pieceLabel(op('or'))).toBe('or');
    });

    it('offers "all of them" and every count below the group size', () => {
        expect(quantChoices(3)).toEqual([ALL, 1, 2]);
        expect(quantChoices(1)).toEqual([ALL]);
        expect(quantChoices(0)).toEqual([]);
    });

    it('offers and, or and brackets — never "not"', () => {
        expect(LOGIC_PIECES).toEqual(['and', 'or', '(', ')']);
        expect(LOGIC_PIECES).not.toContain('not');
    });
});

describe('clampQuant', () => {
    it('leaves all-of alone', () => {
        expect(clampQuant(ALL, 1)).toBe(ALL);
    });
    it('keeps a quantifier that fits', () => {
        expect(clampQuant(2, 5)).toBe(2);
    });
    it('becomes all-of when the group shrank to k or fewer', () => {
        expect(clampQuant(3, 2)).toBe(ALL);
        expect(clampQuant(2, 2)).toBe(ALL);
    });
    it('floors at 1', () => {
        expect(clampQuant(0, 5)).toBe(1);
    });
});

describe('checkCondition — valid conditions', () => {
    it('accepts a valid condition and reads it in words', () => {
        const res = checkCondition(groups({ a: 2, b: 3 }), 'all of a and 1 of b');
        expect(res.problems).toEqual([]);
        expect(res.reading).toBe('all of a AND at least 1 of b');
    });

    it('shows the grouping when and/or mix (and binds tighter than or)', () => {
        expect(checkCondition(groups({ a: 2, b: 2, c: 1 }), 'all of a and 1 of b or all of c').reading)
            .toBe('(all of a AND at least 1 of b) OR all of c');
        expect(checkCondition(groups({ a: 2, b: 2, c: 1 }), 'all of a and (1 of b or all of c)').reading)
            .toBe('all of a AND (at least 1 of b OR all of c)');
    });

    it('accepts a group used more than once', () => {
        expect(checkCondition(groups({ a: 2, b: 2, c: 2 }), '(all of a and all of b) or (all of a and 1 of c)').problems)
            .toEqual([]);
    });

    it('accepts "1 of" a one-CE group, as the backend does', () => {
        expect(checkCondition(groups({ a: 1 }), '1 of a').problems).toEqual([]);
    });
});

describe('checkCondition — what is wrong, in plain words', () => {
    const g = groups({ a: 2, b: 3, c: 1 });
    const firstProblem = (text, gl = g) => checkCondition(gl, text).problems[0];

    it.each([
        ['an empty condition', '', /Nothing would make this rule fire/],
        ['"not"', 'all of a and not all of b and all of c', /cannot use "not"/],
        ['a stray word', 'all of a and banana', /not a group piece/],
        ['a leading operator', 'and all of a and all of b and all of c', /cannot start with "and"/],
        ['a trailing operator', 'all of a and all of b and all of c or', /ends with "or"/],
        ['two operators in a row', 'all of a and or all of b and all of c', /"and" and "or" are next to each other/],
        ['two pieces with nothing between', 'all of a all of b and all of c', /Put "and" or "or" between "all of a" and "all of b"/],
        ['a bracket with no operator before it', 'all of a (all of b) and all of c', /Put "and" or "or" between "all of a" and "\("/],
        ['a closing bracket without an opening one', 'all of a) and all of b and all of c', /"\)" has no matching "\("/],
        ['an unclosed bracket', '(all of a and all of b and all of c', /opened but never closed/],
        ['empty brackets', 'all of a and () and all of b and all of c', /Empty brackets/],
        ['an operator right after "("', '(and all of a) and all of b and all of c', /"and" cannot come right after "\("/],
        ['a group that does not exist', 'all of a and all of b and all of c and all of zed', /"zed" is not one of the groups/],
        ['"K of" more than the group has', 'all of a and 4 of b and all of c', /"4 of b" can never happen — b has only 3 CEs/],
        ['"K of" exactly the group size', '2 of a and all of b and all of c', /"2 of a" means all of them — use "all of a"/],
        ['an unused group', 'all of a and all of b', /Group "c" is not used by the condition/],
    ])('explains %s', (_, text, message) => {
        expect(firstProblem(text)).toMatch(message);
    });

    it('explains an empty group', () => {
        expect(checkCondition(groups({ a: 2, b: 0 }), 'all of a and all of b').problems)
            .toContain('Group "b" has no CEs — add some or delete the group.');
    });

    it('lists every problem, not just the first', () => {
        const { problems, reading } = checkCondition(g, 'all of a all of b');
        expect(problems).toHaveLength(2);           // missing operator + unused group c
        expect(reading).toBeNull();
    });
});

describe('editing helpers', () => {
    const P = (text) => conditionToPieces(text);
    const T = (pieces) => piecesToCondition(pieces);

    it('removing a group takes the operator that joined it', () => {
        expect(T(removeGroupPieces(P('all of a and all of b and all of c'), 'b'))).toBe('all of a and all of c');
        expect(T(removeGroupPieces(P('all of a and all of b'), 'a'))).toBe('all of b');
    });

    it('removing a group cleans up the brackets it leaves', () => {
        expect(T(removeGroupPieces(P('(all of a) and all of b'), 'a'))).toBe('all of b');
        expect(T(removeGroupPieces(P('(all of a and all of b) or all of c'), 'b'))).toBe('(all of a) or all of c');
    });

    it('removing a group used twice removes both pieces', () => {
        expect(T(removeGroupPieces(P('(all of a and all of b) or (all of a and 1 of c)'), 'a')))
            .toBe('(all of b) or (1 of c)');
    });

    it('renaming follows every piece of the group', () => {
        expect(T(renamePieces(P('all of a or (1 of a and all of b)'), 'a', 'scam'))).toBe('all of scam or (1 of scam and all of b)');
    });

    it('shrinking a group clamps its pieces', () => {
        expect(T(clampGroupPieces(P('2 of a and 2 of b'), 'a', 2))).toBe('all of a and 2 of b');
    });
});
