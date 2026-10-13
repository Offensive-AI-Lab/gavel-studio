// Behavior tests for the groups + condition editor.
//
// The firing condition is built from PIECES: a bank offers "all of them" /
// "K of them" per group plus "and", "or" and brackets (GAVEL rules have no
// "not"); clicking a piece adds it to the end of the condition, dragging puts
// it where it is dropped. These tests pin that the pieces write the right
// condition text, that the live check explains what is wrong in plain words,
// that group edits carry the condition along, and that a condition the pieces
// cannot show stays editable as text.

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';

import GroupConditionEditor from '../../../src/components/GroupConditionEditor/GroupConditionEditor';

const POOL = [
    { ce_id: 1, name: 'being_conspiratorial' },
    { ce_id: 2, name: 'being_sycophantic' },
    { ce_id: 3, name: 'trust_seeding' },
    { ce_id: 4, name: 'urgency_pressure' },
];

// The component is controlled; this harness holds the state the real pages hold
// so multi-step interactions behave the way they do in the app.
function Harness({ groupList: initialGroups, condition: initialCondition, onChange }) {
    const [value, setValue] = React.useState({
        groupList: initialGroups, condition: initialCondition,
    });
    return (
        <GroupConditionEditor
            pool={POOL}
            groupList={value.groupList}
            condition={value.condition}
            onChange={(next) => { setValue(next); onChange?.(next); }}
        />
    );
}

const THREE = [
    { name: 'scam_context', ceIds: [1, 2] },
    { name: 'pressure_tactics', ceIds: [2, 3, 4] },
    { name: 'payment_request', ceIds: [3] },
];

const renderEditor = ({ groupList = THREE, condition = '', onChange } = {}) => render(
    <Harness groupList={groupList} condition={condition} onChange={onChange} />,
);

const conditionText = () => screen.getByLabelText('Condition text').textContent;
const rowPieces = () => within(screen.getByRole('list', { name: 'Firing condition pieces' }))
    .queryAllByRole('listitem').map((li) => li.textContent);
const add = (label) => fireEvent.click(screen.getByRole('button', { name: `Add "${label}"` }));
const problems = () => within(screen.getByRole('list', { name: 'Condition problems' }))
    .getAllByRole('listitem').map((li) => li.textContent);

describe('GroupConditionEditor — the bank of pieces', () => {
    it('offers each group "all of them" and every count below its size', () => {
        renderEditor();
        const pressure = within(screen.getByRole('group', { name: 'Pieces for pressure_tactics' }));
        expect(pressure.getAllByRole('button').map((b) => b.textContent)).toEqual(['all of them', '1 of them', '2 of them']);
        const payment = within(screen.getByRole('group', { name: 'Pieces for payment_request' }));
        expect(payment.getAllByRole('button').map((b) => b.textContent)).toEqual(['all of them']);
    });

    it('offers and, or and brackets — but no "not"', () => {
        renderEditor();
        const ops = within(screen.getByRole('group', { name: 'Operators' }));
        expect(ops.getAllByRole('button').map((b) => b.textContent)).toEqual(['and', 'or', '(', ')']);
        expect(screen.queryByRole('button', { name: 'Add "not"' })).toBeNull();
    });

    it('asks for CEs before a group offers pieces', () => {
        renderEditor({ groupList: [...THREE, { name: 'extra_context', ceIds: [] }] });
        const extra = within(screen.getByRole('group', { name: 'Pieces for extra_context' }));
        expect(extra.getByText(/Add CEs to this group first/)).toBeInTheDocument();
        expect(extra.queryAllByRole('button')).toHaveLength(0);
    });

    it('leaves a group with an invalid name out of the bank', () => {
        renderEditor({ groupList: [{ name: 'Bad Name', ceIds: [1] }] });
        expect(screen.queryByRole('group', { name: /Pieces for/ })).toBeNull();
    });
});

describe('GroupConditionEditor — building the condition', () => {
    it('clicking pieces adds them to the end of the condition', () => {
        renderEditor();
        add('all of scam_context');
        add('and');
        add('1 of pressure_tactics');
        add('and');
        add('all of payment_request');
        expect(rowPieces()).toEqual(['all of scam_context', 'and', '1 of pressure_tactics', 'and', 'all of payment_request']);
        expect(conditionText()).toBe('all of scam_context and 1 of pressure_tactics and all of payment_request');
        expect(screen.getByText(/Fires when: all of scam_context AND at least 1 of pressure_tactics AND all of payment_request/))
            .toBeInTheDocument();
    });

    it('builds mixed and/or with brackets and shows the grouping in words', () => {
        renderEditor();
        ['(', 'all of scam_context', 'and', '1 of pressure_tactics', ')', 'or', 'all of payment_request'].forEach(add);
        expect(conditionText()).toBe('(all of scam_context and 1 of pressure_tactics) or all of payment_request');
        expect(screen.getByText(/Fires when: \(all of scam_context AND at least 1 of pressure_tactics\) OR all of payment_request/))
            .toBeInTheDocument();
    });

    it('explains what is wrong while building', () => {
        renderEditor();
        add('all of scam_context');
        add('all of pressure_tactics');
        expect(problems()).toContain('Put "and" or "or" between "all of scam_context" and "all of pressure_tactics".');
        expect(problems()).toContain('Group "payment_request" is not used by the condition — add one of its pieces or delete the group.');
    });

    it('says nothing fires yet when the condition is empty', () => {
        renderEditor();
        expect(problems()[0]).toMatch(/Nothing would make this rule fire yet/);
        expect(screen.getByText('Click or drag pieces here.')).toBeInTheDocument();
    });

    it('removes a single piece with its ×', () => {
        renderEditor({ condition: 'all of scam_context and all of pressure_tactics and all of payment_request' });
        fireEvent.click(screen.getByRole('button', { name: 'Remove piece 2 ("and")' }));
        expect(conditionText()).toBe('all of scam_context all of pressure_tactics and all of payment_request');
    });

    it('"Remove last" and "Clear"', () => {
        renderEditor({ condition: 'all of scam_context and all of pressure_tactics' });
        fireEvent.click(screen.getByRole('button', { name: 'Remove last' }));
        expect(conditionText()).toBe('all of scam_context and');
        fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
        expect(rowPieces()).toEqual([]);
        expect(conditionText()).toBe('—');
    });

    it('the condition text is shown, not typed', () => {
        renderEditor({ condition: 'all of scam_context' });
        expect(screen.getByLabelText('Condition text').tagName).toBe('CODE');
    });
});

describe('GroupConditionEditor — drag and drop', () => {
    const piece = (i) => within(screen.getByRole('list', { name: 'Firing condition pieces' })).getAllByRole('listitem')[i];

    it('a bank piece dropped on a piece goes in front of it', () => {
        renderEditor({ condition: 'all of scam_context or all of payment_request' });
        fireEvent.dragStart(screen.getByRole('button', { name: 'Add "("' }));
        fireEvent.dragOver(piece(0));
        fireEvent.drop(piece(0));
        expect(conditionText()).toBe('(all of scam_context or all of payment_request');
    });

    it('a bank piece dropped on the empty part of the row goes at the end', () => {
        renderEditor({ condition: 'all of scam_context' });
        fireEvent.dragStart(screen.getByRole('button', { name: 'Add "and"' }));
        const row = screen.getByRole('list', { name: 'Firing condition pieces' });
        fireEvent.dragOver(row);
        fireEvent.drop(row);
        expect(conditionText()).toBe('all of scam_context and');
    });

    it('pieces are reordered by dragging them', () => {
        renderEditor({ condition: 'all of scam_context and 1 of pressure_tactics' });
        fireEvent.dragStart(piece(2));
        fireEvent.dragOver(piece(0));
        fireEvent.drop(piece(0));
        expect(conditionText()).toBe('1 of pressure_tactics all of scam_context and');
    });

    it('a drop with nothing being dragged changes nothing', () => {
        const onChange = vi.fn();
        renderEditor({ condition: 'all of scam_context', onChange });
        fireEvent.drop(piece(0));
        expect(onChange).not.toHaveBeenCalled();
    });
});

describe('GroupConditionEditor — group edits carry the condition along', () => {
    it('renaming a group renames its pieces', () => {
        renderEditor({ condition: '(all of scam_context and 1 of pressure_tactics) or all of payment_request' });
        fireEvent.change(screen.getByLabelText('Group 3 name'), { target: { value: 'payment_demand' } });
        expect(conditionText()).toBe('(all of scam_context and 1 of pressure_tactics) or all of payment_demand');
    });

    it('a name emptied while typing keeps the pieces until the new name is valid', () => {
        renderEditor({ condition: 'all of scam_context and all of pressure_tactics and all of payment_request' });
        const field = screen.getByLabelText('Group 1 name');
        fireEvent.change(field, { target: { value: '' } });
        expect(conditionText()).toBe('all of scam_context and all of pressure_tactics and all of payment_request');
        fireEvent.change(field, { target: { value: 'fraud_context' } });
        expect(conditionText()).toBe('all of fraud_context and all of pressure_tactics and all of payment_request');
    });

    it('deleting a group removes its pieces and the operator that joined them', () => {
        renderEditor({ condition: 'all of scam_context and all of pressure_tactics and all of payment_request' });
        fireEvent.click(screen.getByLabelText('Delete group pressure_tactics'));
        expect(conditionText()).toBe('all of scam_context and all of payment_request');
    });

    it('a new group is not added to the condition by itself', () => {
        renderEditor({ condition: 'all of scam_context and all of pressure_tactics and all of payment_request' });
        fireEvent.click(screen.getByText(/Add group/));
        expect(conditionText()).toBe('all of scam_context and all of pressure_tactics and all of payment_request');
        expect(problems()).toContain('Group "required" has no CEs — add some or delete the group.');
        expect(problems()).toContain('Group "required" is not used by the condition — add one of its pieces or delete the group.');
    });

    it('shrinking a group turns an impossible "K of" into "all of"', () => {
        renderEditor({ condition: 'all of scam_context and 2 of pressure_tactics and all of payment_request' });
        fireEvent.click(screen.getByLabelText('Remove trust_seeding from pressure_tactics'));
        expect(conditionText()).toBe('all of scam_context and all of pressure_tactics and all of payment_request');
    });

    it('adds a CE to a group', () => {
        const onChange = vi.fn();
        renderEditor({ groupList: [{ name: 'required', ceIds: [1] }], condition: 'all of required', onChange });
        fireEvent.change(screen.getByLabelText('Add a CE to required'), { target: { value: '2' } });
        expect(onChange).toHaveBeenLastCalledWith({
            groupList: [{ name: 'required', ceIds: [1, 2] }],
            condition: 'all of required',
        });
    });

    it('marks an invalid group name', () => {
        renderEditor({ groupList: [{ name: 'Bad Name', ceIds: [1] }], condition: '' });
        expect(screen.getByLabelText('Group 1 name').style.border).toMatch(/248, 113, 113/);
    });
});

describe('GroupConditionEditor — stored conditions', () => {
    it('opens a mixed condition as pieces', () => {
        renderEditor({ condition: '(all of scam_context and 1 of pressure_tactics) or all of payment_request' });
        expect(rowPieces()).toEqual(['(', 'all of scam_context', 'and', '1 of pressure_tactics', ')', 'or', 'all of payment_request']);
        expect(screen.queryByRole('list', { name: 'Condition problems' })).toBeNull();
    });

    describe('a condition the pieces cannot show', () => {
        const withNot = 'all of scam_context and not all of pressure_tactics and all of payment_request';

        it('stays editable as text and says why', () => {
            renderEditor({ condition: withNot });
            const field = screen.getByLabelText('Condition text');
            expect(field).toHaveValue(withNot);
            expect(field).not.toHaveAttribute('readonly');
            expect(screen.getByText(/which GAVEL rules do not allow/)).toBeInTheDocument();
            expect(problems()[0]).toMatch(/cannot use "not"/);
            expect(screen.queryByRole('group', { name: 'Operators' })).toBeNull();
        });

        it('is not rewritten by an unrelated group edit', () => {
            const onChange = vi.fn();
            renderEditor({ condition: withNot, onChange });
            fireEvent.change(screen.getByLabelText('Add a CE to payment_request'), { target: { value: '1' } });
            expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ condition: withNot }));
        });

        it('"Clear and use pieces" brings the builder back', () => {
            renderEditor({ condition: withNot });
            fireEvent.click(screen.getByRole('button', { name: 'Clear and use pieces' }));
            expect(screen.getByRole('group', { name: 'Operators' })).toBeInTheDocument();
            expect(rowPieces()).toEqual([]);
        });
    });
});
