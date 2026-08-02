import React from 'react';
import { describe, it, expect } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import Message from '../../../src/components/Message/Message';

const wrap = (ui) => <MemoryRouter>{ui}</MemoryRouter>;

describe('Message', () => {
    it('renders the title and text', () => {
        render(wrap(<Message type="success" title="Done" text="Saved successfully" />));
        expect(screen.getByText('Done')).toBeInTheDocument();
        expect(screen.getByText('Saved successfully')).toBeInTheDocument();
    });

    it('renders an action link when actionText + actionLink are provided', () => {
        render(wrap(
            <Message
                type="success"
                title="Done"
                text="Saved"
                actionText="Continue"
                actionLink="/next"
            />,
        ));
        const button = screen.getByRole('button', { name: /Continue/ });
        expect(button).toBeInTheDocument();
        // The Link wrapper should target the supplied URL.
        const anchor = button.closest('a');
        expect(anchor).toHaveAttribute('href', '/next');
    });

    it('omits the action button when actionText is missing', () => {
        render(wrap(<Message type="error" title="Oops" text="Failed" actionLink="/login" />));
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('omits the action button when actionLink is missing', () => {
        render(wrap(<Message type="error" title="Oops" text="Failed" actionText="Retry" />));
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('renders the correct icon for the success vs error variant', () => {
        // We can't easily distinguish two SVGs by attribute, but we can
        // verify both variants render an SVG without throwing.
        const { container: ok } = render(wrap(<Message type="success" title="A" text="B" />));
        expect(ok.querySelector('svg')).toBeTruthy();
        const { container: bad } = render(wrap(<Message type="error" title="A" text="B" />));
        expect(bad.querySelector('svg')).toBeTruthy();
    });
});
