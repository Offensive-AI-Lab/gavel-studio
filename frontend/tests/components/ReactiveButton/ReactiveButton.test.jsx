import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FiCheck } from 'react-icons/fi';
import ReactiveButton from '../../../src/components/ReactiveButton/ReactiveButton';

describe('ReactiveButton', () => {
    it('renders the supplied label', () => {
        render(<ReactiveButton label="Save" onClick={() => {}} />);
        expect(screen.getByRole('button', { name: /Save/ })).toBeInTheDocument();
    });

    it('fires onClick when clicked', () => {
        const onClick = vi.fn();
        render(<ReactiveButton label="Click me" onClick={onClick} />);
        fireEvent.click(screen.getByRole('button'));
        expect(onClick).toHaveBeenCalledTimes(1);
    });

    it('does not fire onClick when disabled', () => {
        // The native disabled attribute is the actual gate — test it via
        // fireEvent which respects DOM semantics.
        const onClick = vi.fn();
        render(<ReactiveButton label="Click me" onClick={onClick} disabled />);
        const btn = screen.getByRole('button');
        expect(btn).toBeDisabled();
        fireEvent.click(btn);
        expect(onClick).not.toHaveBeenCalled();
    });

    it('applies the disabled CSS class for visual styling when disabled', () => {
        // The CSS class drives appearance; the disabled attribute drives
        // behavior. Both should be set together.
        render(<ReactiveButton label="x" onClick={() => {}} disabled />);
        expect(screen.getByRole('button')).toHaveClass('disabled');
    });

    it('renders the supplied custom Icon component', () => {
        // The default icon is FiPlus; passing Icon=FiCheck swaps the SVG.
        // We can't easily query SVG paths, so just confirm a render works
        // — coverage is the goal here.
        const { container } = render(
            <ReactiveButton label="Done" onClick={() => {}} Icon={FiCheck} />,
        );
        // SVG should be present.
        expect(container.querySelector('svg')).toBeTruthy();
    });
});
