// RuleLogicPreview — groups + condition rendering.
//
// The component shows one chip-cluster per named CE group (color-coded
// group-name pill + member pills) and the condition expression in monospace
// underneath, with group names inside the condition tinted to their group's
// color (so each name appears twice: on the pill and in the condition).

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RuleLogicPreview from '../../../src/components/RuleLogicPreview/RuleLogicPreview';

describe('RuleLogicPreview', () => {
    it('renders a chip cluster per group with member pills', () => {
        render(
            <RuleLogicPreview
                groups={{ required: ['hatespeech'], target: ['ethnoracial', 'LGBTQ'] }}
                condition="all of required and 1 of target"
            />,
        );
        // Group names appear twice — the group pill and tinted inside the condition.
        expect(screen.getAllByText('required')).toHaveLength(2);
        expect(screen.getAllByText('target')).toHaveLength(2);
        expect(screen.getByText('hatespeech')).toBeInTheDocument();
        expect(screen.getByText('ethnoracial')).toBeInTheDocument();
        expect(screen.getByText('LGBTQ')).toBeInTheDocument();
    });

    it('renders the condition with group names tinted to their group color', () => {
        render(
            <RuleLogicPreview
                groups={{ required: ['a', 'b'] }}
                condition="all of required"
            />,
        );
        const code = document.querySelector('code');
        expect(code).toHaveTextContent('all of required');
        const [pill, inCondition] = screen.getAllByText('required');
        // Same group color on the pill and on the name inside the condition.
        expect(inCondition.style.color).toBe(pill.style.color);
    });

    it('accepts {ce_id, name} member objects', () => {
        render(
            <RuleLogicPreview
                groups={{ g1: [{ ce_id: 4, name: 'click_or_enter' }] }}
                condition="all of g1"
            />,
        );
        expect(screen.getByText('click_or_enter')).toBeInTheDocument();
    });

    it('shows the empty hint when there are no groups', () => {
        render(<RuleLogicPreview groups={{}} condition="" />);
        expect(screen.getByText(/Add cognitive elements to a group/)).toBeInTheDocument();
    });

    it('renders a custom title', () => {
        render(<RuleLogicPreview title="My Title" groups={{ g: ['x'] }} condition="all of g" />);
        expect(screen.getByText('My Title')).toBeInTheDocument();
    });
});
