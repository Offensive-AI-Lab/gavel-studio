// Behavior tests for the groups + condition editor.
//
// The point of the rework: the firing condition is no longer typed. It is
// generated from per-group pickers and shown read-only, so the user cannot
// write logic referencing groups that do not exist. These tests pin that the
// pickers produce the right condition text, that the text field is locked, and
// that conditions the pickers cannot model stay editable instead of being
// silently rewritten.

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';

import GroupConditionEditor from '../../../src/components/GroupConditionEditor/GroupConditionEditor';

const POOL = [
    { ce_id: 1, name: 'being_conspiratorial' },
    { ce_id: 2, name: 'being_sycophantic' },
    { ce_id: 3, name: 'trust_seeding' },
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

const renderEditor = ({ groupList, condition = '', onChange } = {}) => render(
    <Harness
        groupList={groupList ?? [{ name: 'required', ceIds: [1, 2] }]}
        condition={condition}
        onChange={onChange}
    />,
);

const conditionField = () => screen.getByLabelText('Firing condition');
const quantPicker = (groupName) =>
    screen.getByLabelText(`How much of ${groupName} must match`);

describe('GroupConditionEditor — the condition is generated, not typed', () => {
    it('locks the condition field', () => {
        renderEditor({ condition: 'all of required' });
        expect(conditionField()).toHaveAttribute('readonly');
    });

    it('explains that the text is written from the pickers', () => {
        renderEditor({ condition: 'all of required' });
        expect(screen.getByText(/Written for you from the choices above/i)).toBeInTheDocument();
    });

    it('rewrites the condition when a quantifier changes', () => {
        renderEditor({ condition: 'all of required' });
        fireEvent.change(quantPicker('required'), { target: { value: '1' } });
        expect(conditionField()).toHaveValue('1 of required');
    });

    it('offers a quantifier per member, capped at the group size', () => {
        renderEditor({ groupList: [{ name: 'required', ceIds: [1, 2] }], condition: 'all of required' });
        const options = within(quantPicker('required')).getAllByRole('option')
            .map((o) => o.textContent);
        // 2 members -> "all", "any 1", and supporting-only. "any 2" would just
        // mean "all", and the library's validator rejects k == group size.
        expect(options).toEqual(['Needs all of them', 'Needs any 1', 'Supporting only']);
    });

    it('drops a group from the condition when set to supporting only', () => {
        renderEditor({
            groupList: [{ name: 'required', ceIds: [1] }, { name: 'extra', ceIds: [2] }],
            condition: 'all of required and all of extra',
        });
        fireEvent.change(quantPicker('extra'), { target: { value: 'unused' } });
        expect(conditionField()).toHaveValue('all of required');
    });
});

describe('GroupConditionEditor — editing groups keeps the condition true', () => {
    it('adds a new group to the condition', () => {
        renderEditor({ condition: 'all of required' });
        fireEvent.click(screen.getByRole('button', { name: /Add group/i }));
        expect(conditionField()).toHaveValue('all of required and all of option_1');
    });

    it('removes a deleted group from the condition', () => {
        renderEditor({
            groupList: [{ name: 'required', ceIds: [1] }, { name: 'extra', ceIds: [2] }],
            condition: 'all of required and 1 of extra',
        });
        fireEvent.click(screen.getByRole('button', { name: /Delete group extra/i }));
        expect(conditionField()).toHaveValue('all of required');
    });

    it('carries a quantifier across a rename', () => {
        renderEditor({
            groupList: [{ name: 'required', ceIds: [1, 2] }],
            condition: '1 of required',
        });
        fireEvent.change(screen.getByLabelText('Group 1 name'), { target: { value: 'aggression' } });
        // The requirement follows the new name instead of vanishing.
        expect(conditionField()).toHaveValue('1 of aggression');
    });

    it('clamps "at least k" when the group shrinks below k', () => {
        renderEditor({
            groupList: [{ name: 'required', ceIds: [1, 2, 3] }],
            condition: '2 of required',
        });
        fireEvent.click(screen.getByRole('button', { name: /Remove trust_seeding from required/i }));
        fireEvent.click(screen.getByRole('button', { name: /Remove being_sycophantic from required/i }));
        // A one-member group cannot need "at least 2".
        expect(conditionField()).toHaveValue('all of required');
    });

    it('keeps the condition in the order the groups are shown', () => {
        renderEditor({
            groupList: [{ name: 'first', ceIds: [1] }, { name: 'second', ceIds: [2] }],
            condition: 'all of second and all of first',
        });
        fireEvent.change(quantPicker('first'), { target: { value: 'all' } });
        expect(conditionField()).toHaveValue('all of first and all of second');
    });
});

describe('GroupConditionEditor — combining groups', () => {
    it('hides the and/or control until a second group is required', () => {
        renderEditor({ condition: 'all of required' });
        expect(screen.queryByLabelText(/Combine the groups/i)).toBeNull();
    });

    it('shows the and/or control once two groups are required', () => {
        renderEditor({
            groupList: [{ name: 'a', ceIds: [1] }, { name: 'b', ceIds: [2] }],
            condition: 'all of a and all of b',
        });
        expect(screen.getByLabelText(/Combine the groups/i)).toBeInTheDocument();
    });

    it('switches the whole condition to or', () => {
        renderEditor({
            groupList: [{ name: 'a', ceIds: [1] }, { name: 'b', ceIds: [2] }],
            condition: 'all of a and 1 of b',
        });
        fireEvent.change(screen.getByLabelText(/Combine the groups/i), { target: { value: 'or' } });
        expect(conditionField()).toHaveValue('all of a or 1 of b');
    });

    it('describes what the condition means in words', () => {
        renderEditor({
            groupList: [{ name: 'a', ceIds: [1] }, { name: 'b', ceIds: [2] }],
            condition: 'all of a and 1 of b',
        });
        expect(screen.getByText(/fires when ALL of these groups/i)).toBeInTheDocument();
    });
});

describe('GroupConditionEditor — advanced conditions are not rewritten', () => {
    const advanced = {
        groupList: [{ name: 'a', ceIds: [1] }, { name: 'b', ceIds: [2] }, { name: 'c', ceIds: [3] }],
        condition: '(all of a or 1 of b) and 1 of c',
    };

    it('leaves the field editable', () => {
        renderEditor(advanced);
        expect(conditionField()).not.toHaveAttribute('readonly');
    });

    it('says why the pickers are unavailable', () => {
        renderEditor(advanced);
        expect(screen.getByText(/advanced logic/i)).toBeInTheDocument();
    });

    it('hides the quantifier pickers rather than showing wrong values', () => {
        renderEditor(advanced);
        expect(screen.queryByLabelText(/How much of a must match/i)).toBeNull();
    });

    it('preserves the original text when an unrelated edit happens', () => {
        const onChange = vi.fn();
        renderEditor({ ...advanced, onChange });
        fireEvent.click(screen.getByRole('button', { name: /Add group/i }));
        expect(onChange).toHaveBeenCalledWith(
            expect.objectContaining({ condition: '(all of a or 1 of b) and 1 of c' }));
    });

    it('returns to the pickers once the text is cleared', () => {
        renderEditor(advanced);
        fireEvent.change(conditionField(), { target: { value: '' } });
        expect(conditionField()).toHaveAttribute('readonly');
    });
});

describe('GroupConditionEditor — group editing still works', () => {
    it('adds a CE to a group', () => {
        const onChange = vi.fn();
        renderEditor({ condition: 'all of required', onChange });
        fireEvent.change(screen.getByLabelText('Add a CE to required'), { target: { value: '3' } });
        expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
            groupList: [expect.objectContaining({ ceIds: [1, 2, 3] })],
        }));
    });

    it('marks an invalid group name', () => {
        renderEditor({ groupList: [{ name: 'Bad Name', ceIds: [1] }], condition: '' });
        expect(screen.getByLabelText('Group 1 name')).toHaveStyle({
            border: '1px solid rgba(248, 113, 113, 0.65)',
        });
    });

    it('warns when nothing would fire', () => {
        renderEditor({ groupList: [{ name: 'a', ceIds: [1] }], condition: 'all of a' });
        fireEvent.change(quantPicker('a'), { target: { value: 'unused' } });
        expect(conditionField()).toHaveValue('');
        expect(screen.getByText(/never fires/i)).toBeInTheDocument();
    });
});
