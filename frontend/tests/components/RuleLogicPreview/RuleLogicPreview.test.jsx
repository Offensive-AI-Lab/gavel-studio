// RuleLogicPreview — groups + condition rendering.
//
// The component shows one chip-cluster per named CE group (group-name header
// + member pills) and the condition expression verbatim in monospace.

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
        expect(screen.getByText('required')).toBeInTheDocument();
        expect(screen.getByText('target')).toBeInTheDocument();
        expect(screen.getByText('hatespeech')).toBeInTheDocument();
        expect(screen.getByText('ethnoracial')).toBeInTheDocument();
        expect(screen.getByText('LGBTQ')).toBeInTheDocument();
    });

    it('renders the condition verbatim', () => {
        render(
            <RuleLogicPreview
                groups={{ required: ['a', 'b'] }}
                condition="all of required"
            />,
        );
        expect(screen.getByText('all of required')).toBeInTheDocument();
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
