// Unit tests for the Explainer guidance panel.
//
// Explainer is a pure presentational component:
//   <Explainer title="...">{children}</Explainer>
// It renders an outer panel <div>, an <h3> title (default "About this page"),
// and a body <div> wrapping whatever children are passed. No API, no hooks,
// no routing — so these tests focus on rendering, the title default/override
// branch, children pass-through, and inline-style application.

import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Explainer from '../../../src/components/Explainer/Explainer';

describe('Explainer', () => {
    it('renders the default title when none is provided', () => {
        render(<Explainer />);
        const heading = screen.getByRole('heading', { level: 3 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent('About this page');
    });

    it('renders a custom title when provided', () => {
        render(<Explainer title="About this step" />);
        const heading = screen.getByRole('heading', { level: 3 });
        expect(heading).toHaveTextContent('About this step');
        // Default text must NOT be present when a title is given.
        expect(screen.queryByText('About this page')).not.toBeInTheDocument();
    });

    it('renders an empty-string title (explicit override of the default)', () => {
        render(<Explainer title="" />);
        const heading = screen.getByRole('heading', { level: 3 });
        // Empty string is a real value, so the default should not kick in.
        expect(heading).toHaveTextContent('');
        expect(heading.textContent).toBe('');
    });

    it('renders provided children inside the body', () => {
        render(
            <Explainer title="Guidance">
                <p>Plain-language paragraph explaining the page.</p>
            </Explainer>,
        );
        expect(
            screen.getByText('Plain-language paragraph explaining the page.'),
        ).toBeInTheDocument();
    });

    it('renders multiple/complex children (paragraph + list)', () => {
        render(
            <Explainer title="Steps">
                <p>Intro paragraph.</p>
                <ul>
                    <li>First item</li>
                    <li>Second item</li>
                </ul>
            </Explainer>,
        );
        expect(screen.getByText('Intro paragraph.')).toBeInTheDocument();
        expect(screen.getByText('First item')).toBeInTheDocument();
        expect(screen.getByText('Second item')).toBeInTheDocument();
        const items = screen.getAllByRole('listitem');
        expect(items).toHaveLength(2);
    });

    it('renders cleanly with no children', () => {
        const { container } = render(<Explainer title="Empty" />);
        // Outer panel div -> h3 + body div. Body div should exist but be empty.
        const panel = container.firstChild;
        expect(panel.tagName).toBe('DIV');
        // Title h3 plus a body div = exactly two element children.
        expect(panel.children).toHaveLength(2);
        expect(panel.children[0].tagName).toBe('H3');
        expect(panel.children[1].tagName).toBe('DIV');
        expect(panel.children[1].childNodes).toHaveLength(0);
    });

    it('renders a string child as the body text', () => {
        render(<Explainer title="Note">just some text</Explainer>);
        expect(screen.getByText('just some text')).toBeInTheDocument();
    });

    it('applies the indigo-tinted panel inline styles to the outer container', () => {
        const { container } = render(<Explainer title="Styled" />);
        const panel = container.firstChild;
        expect(panel).toHaveStyle({ borderRadius: '12px' });
        expect(panel).toHaveStyle({ marginBottom: '16px' });
        // The component drives all visuals through inline style objects.
        expect(panel.getAttribute('style')).toBeTruthy();
    });

    it('applies uppercase styling to the title heading', () => {
        render(<Explainer title="lowercase title" />);
        const heading = screen.getByRole('heading', { level: 3 });
        expect(heading).toHaveStyle({ textTransform: 'uppercase' });
        expect(heading).toHaveStyle({ fontWeight: '700' });
    });

    it('wraps children in a dedicated body div (second child of panel)', () => {
        const { container } = render(
            <Explainer title="Body check">
                <span data-testid="inner">x</span>
            </Explainer>,
        );
        const panel = container.firstChild;
        const body = panel.children[1];
        expect(body.tagName).toBe('DIV');
        expect(body.querySelector('[data-testid="inner"]')).not.toBeNull();
    });

    it('updates the title when the prop changes on re-render', () => {
        const { rerender } = render(<Explainer title="First" />);
        expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('First');
        rerender(<Explainer title="Second" />);
        expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('Second');
    });

    it('falls back to the default title when title is explicitly undefined', () => {
        render(<Explainer title={undefined}>body</Explainer>);
        expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent(
            'About this page',
        );
    });

    // --- collapsible (opt-in) -------------------------------------------------

    it('non-collapsible is always expanded with no toggle affordance', () => {
        render(<Explainer title="X"><p>always shown</p></Explainer>);
        expect(screen.getByText('always shown')).toBeInTheDocument();
        expect(screen.queryByText(/show ▼|hide ▲/)).not.toBeInTheDocument();
    });

    it('collapsible + defaultOpen=false hides the body until the header is clicked', () => {
        render(
            <Explainer title="Realtime" collapsible defaultOpen={false}>
                <p>hidden body</p>
            </Explainer>,
        );
        // Body collapsed initially; a "show" affordance is present.
        expect(screen.queryByText('hidden body')).not.toBeInTheDocument();
        expect(screen.getByText(/show ▼/)).toBeInTheDocument();
        // Clicking the header expands it.
        fireEvent.click(screen.getByRole('heading', { level: 3 }));
        expect(screen.getByText('hidden body')).toBeInTheDocument();
        expect(screen.getByText(/hide ▲/)).toBeInTheDocument();
    });

    it('collapsible defaults to open and toggles closed on click', () => {
        render(<Explainer title="X" collapsible><p>visible body</p></Explainer>);
        expect(screen.getByText('visible body')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('heading', { level: 3 }));
        expect(screen.queryByText('visible body')).not.toBeInTheDocument();
    });
});
