// Behavior tests for CognitiveElementCard.
//
// The card is a mostly-presentational component, but it has a fair amount of
// conditional branching: draft/public badges, bookmark/publish/delete
// affordances, author link, examples rendering (string vs object samples,
// YES/NO verdicts).
//
// the api module to keep everything off the network. We don't assert on
// it renders at all, driven by ce.public_id.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, within } from '@testing-library/react';

import CognitiveElementCard from '../../../src/components/CognitiveElementCard/CognitiveElementCard';

const renderCard = (props) =>
    render(
        <MemoryRouter>
            <CognitiveElementCard {...props} />
        </MemoryRouter>,
    );

const baseCe = (overrides = {}) => ({
    ce_id: 7,
    name: 'Sarcasm',
    definition: 'Detects sarcastic intent',
    ...overrides,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
});

describe('CognitiveElementCard — header / collapsed state', () => {
    it('renders the CE name and definition', () => {
        renderCard({ ce: baseCe(), isOpen: false, onToggle: vi.fn() });
        expect(screen.getByText('Sarcasm')).toBeInTheDocument();
        expect(screen.getByText('Detects sarcastic intent')).toBeInTheDocument();
    });

    it('shows a fallback when no definition is provided', () => {
        renderCard({
            ce: baseCe({ definition: '' }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('No definition provided')).toBeInTheDocument();
    });

    it('is collapsed (no ce-content) when isOpen is false', () => {
        const { container } = renderCard({
            ce: baseCe(),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(container.querySelector('.ce-content')).toBeNull();
        expect(container.querySelector('.ce-card.expanded')).toBeNull();
    });

    it('applies the expanded class and renders content when isOpen is true', () => {
        const { container } = renderCard({
            ce: baseCe(),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(container.querySelector('.ce-card.expanded')).not.toBeNull();
        expect(container.querySelector('.ce-content')).not.toBeNull();
    });

    it('calls onToggle with the ce_id when the header is clicked', () => {
        const onToggle = vi.fn();
        const { container } = renderCard({
            ce: baseCe(),
            isOpen: false,
            onToggle,
        });
        fireEvent.click(container.querySelector('.ce-header'));
        expect(onToggle).toHaveBeenCalledWith(7);
    });

    it('shows the full definition in the expanded body under a "What this CE means" label', () => {
        renderCard({ ce: baseCe(), isOpen: true, onToggle: vi.fn() });
        expect(screen.getByText(/What this CE means/i)).toBeInTheDocument();
        // Appears in the header preview AND the expanded body (full, readable).
        expect(screen.getAllByText('Detects sarcastic intent').length).toBeGreaterThanOrEqual(2);
    });

    it('does not render the definition block when collapsed', () => {
        renderCard({ ce: baseCe(), isOpen: false, onToggle: vi.fn() });
        expect(screen.queryByText(/What this CE means/i)).toBeNull();
    });
});

describe('CognitiveElementCard — draft / public badge', () => {
    it('renders a Draft badge when is_local_draft is true', () => {
        renderCard({
            ce: baseCe({ is_local_draft: true }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('Draft')).toBeInTheDocument();
    });

    it('renders a Public badge when is_local_draft is false', () => {
        renderCard({
            ce: baseCe({ is_local_draft: false }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('Public')).toBeInTheDocument();
    });

    it('renders no badge when is_local_draft is undefined (not a boolean)', () => {
        renderCard({ ce: baseCe(), isOpen: false, onToggle: vi.fn() });
        expect(screen.queryByText('Draft')).toBeNull();
        expect(screen.queryByText('Public')).toBeNull();
    });
});

describe('CognitiveElementCard — categories', () => {
    it('renders category pills in the collapsed header', () => {
        renderCard({
            ce: baseCe({ categories: ['Safety', 'Tone'] }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('Safety')).toBeInTheDocument();
        expect(screen.getByText('Tone')).toBeInTheDocument();
    });

    it('renders categories both in header and content when expanded', () => {
        renderCard({
            ce: baseCe({ categories: ['Safety'] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        // One in the header info block, one inside the expanded content block.
        expect(screen.getAllByText('Safety')).toHaveLength(2);
    });

    it('renders no category pills when categories is empty', () => {
        const { container } = renderCard({
            ce: baseCe({ categories: [] }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(container.querySelector('.pill')).toBeNull();
    });
});

describe('CognitiveElementCard — author link', () => {
    it('renders an author link to the profile when created_by_username is set', () => {
        renderCard({
            ce: baseCe({ created_by_username: 'alice' }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        const link = screen.getByRole('link', { name: /by @alice/i });
        expect(link).toHaveAttribute('href', '/profile/alice');
    });

    it('does not render an author link when created_by_username is absent', () => {
        renderCard({ ce: baseCe(), isOpen: false, onToggle: vi.fn() });
        expect(screen.queryByRole('link')).toBeNull();
    });

    it('stops propagation so clicking the author link does not toggle the card', () => {
        const onToggle = vi.fn();
        renderCard({
            ce: baseCe({ created_by_username: 'alice' }),
            isOpen: false,
            onToggle,
        });
        fireEvent.click(screen.getByRole('link', { name: /by @alice/i }));
        expect(onToggle).not.toHaveBeenCalled();
    });
});

describe('CognitiveElementCard — bookmark affordance', () => {
    const bookmarkable = (overrides = {}) =>
        baseCe({ is_local_draft: false, public_id: 'pub-1', ...overrides });

    it('renders a Save button for a bookmarkable, not-yet-bookmarked CE', () => {
        renderCard({
            ce: bookmarkable(),
            isOpen: false,
            onToggle: vi.fn(),
            onBookmark: vi.fn(),
            isBookmarked: false,
        });
        expect(screen.getByRole('button', { name: /bookmark ce/i })).toHaveTextContent('Save');
    });

    it('renders a Remove label when already bookmarked', () => {
        renderCard({
            ce: bookmarkable(),
            isOpen: false,
            onToggle: vi.fn(),
            onBookmark: vi.fn(),
            isBookmarked: true,
        });
        expect(screen.getByRole('button', { name: /bookmark ce/i })).toHaveTextContent('Remove');
    });

    it('honors a custom bookmarkLabel', () => {
        renderCard({
            ce: bookmarkable(),
            isOpen: false,
            onToggle: vi.fn(),
            onBookmark: vi.fn(),
            bookmarkLabel: 'Add to set',
        });
        expect(screen.getByRole('button', { name: /bookmark ce/i })).toHaveTextContent('Add to set');
    });

    it('calls onBookmark with the ce when clicked', () => {
        const onBookmark = vi.fn();
        const ce = bookmarkable();
        renderCard({ ce, isOpen: false, onToggle: vi.fn(), onBookmark });
        fireEvent.click(screen.getByRole('button', { name: /bookmark ce/i }));
        expect(onBookmark).toHaveBeenCalledWith(ce);
    });

    it('does not render the bookmark button for a draft', () => {
        renderCard({
            ce: bookmarkable({ is_local_draft: true }),
            isOpen: false,
            onToggle: vi.fn(),
            onBookmark: vi.fn(),
        });
        expect(screen.queryByRole('button', { name: /bookmark ce/i })).toBeNull();
    });

    it('does not render the bookmark button without a public_id', () => {
        renderCard({
            ce: bookmarkable({ public_id: undefined }),
            isOpen: false,
            onToggle: vi.fn(),
            onBookmark: vi.fn(),
        });
        expect(screen.queryByRole('button', { name: /bookmark ce/i })).toBeNull();
    });

    it('does not render the bookmark button when onBookmark is not supplied', () => {
        renderCard({ ce: bookmarkable(), isOpen: false, onToggle: vi.fn() });
        expect(screen.queryByRole('button', { name: /bookmark ce/i })).toBeNull();
    });
});

describe('CognitiveElementCard — publish affordance', () => {
    it('renders a Publish button only for local drafts with onPublish wired', () => {
        renderCard({
            ce: baseCe({ is_local_draft: true }),
            isOpen: false,
            onToggle: vi.fn(),
            onPublish: vi.fn(),
        });
        expect(screen.getByRole('button', { name: /publish ce to library/i })).toBeInTheDocument();
    });

    it('calls onPublish with the ce when clicked', () => {
        const onPublish = vi.fn();
        const ce = baseCe({ is_local_draft: true });
        renderCard({ ce, isOpen: false, onToggle: vi.fn(), onPublish });
        fireEvent.click(screen.getByRole('button', { name: /publish ce to library/i }));
        expect(onPublish).toHaveBeenCalledWith(ce);
    });

    it('does not render Publish for a non-draft CE', () => {
        renderCard({
            ce: baseCe({ is_local_draft: false }),
            isOpen: false,
            onToggle: vi.fn(),
            onPublish: vi.fn(),
        });
        expect(screen.queryByRole('button', { name: /publish ce to library/i })).toBeNull();
    });

    it('does not render Publish when onPublish is missing', () => {
        renderCard({
            ce: baseCe({ is_local_draft: true }),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(screen.queryByRole('button', { name: /publish ce to library/i })).toBeNull();
    });
});

describe('CognitiveElementCard — delete affordance', () => {
    it('renders a delete icon when onDelete is provided', () => {
        const { container } = renderCard({
            ce: baseCe(),
            isOpen: false,
            onToggle: vi.fn(),
            onDelete: vi.fn(),
        });
        expect(container.querySelector('.delete-icon')).not.toBeNull();
    });

    it('does not render a delete icon when onDelete is absent', () => {
        const { container } = renderCard({
            ce: baseCe(),
            isOpen: false,
            onToggle: vi.fn(),
        });
        expect(container.querySelector('.delete-icon')).toBeNull();
    });

    it('calls onDelete with the ce and does not toggle when clicked', () => {
        const onDelete = vi.fn();
        const onToggle = vi.fn();
        const ce = baseCe();
        const { container } = renderCard({ ce, isOpen: false, onToggle, onDelete });
        fireEvent.click(container.querySelector('.delete-icon'));
        expect(onDelete).toHaveBeenCalledWith(ce);
        expect(onToggle).not.toHaveBeenCalled();
    });
});

describe('CognitiveElementCard — chevron indicator', () => {
    it('shows the up chevron when open and down chevron when closed', () => {
        const { container: closed } = renderCard({
            ce: baseCe(),
            isOpen: false,
            onToggle: vi.fn(),
        });
        const closedSvgs = closed.querySelectorAll('.ce-actions svg');
        expect(closedSvgs.length).toBeGreaterThan(0);

        const { container: open } = renderCard({
            ce: baseCe(),
            isOpen: true,
            onToggle: vi.fn(),
        });
        // Both render a single chevron svg in the actions area (no other
        // action buttons in this minimal config).
        expect(open.querySelectorAll('.ce-actions svg').length).toBe(1);
        expect(closedSvgs.length).toBe(1);
    });
});

describe('CognitiveElementCard — examples (expanded content)', () => {
    it('shows the empty state when there are no examples', () => {
        renderCard({
            ce: baseCe({ examples: [] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('No examples available.')).toBeInTheDocument();
    });

    it('shows the empty state when examples is not an array', () => {
        renderCard({
            ce: baseCe({ examples: 'not-an-array' }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('No examples available.')).toBeInTheDocument();
    });

    it('renders string examples with a default YES verdict', () => {
        renderCard({
            ce: baseCe({ examples: ['this is sarcastic'] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('this is sarcastic')).toBeInTheDocument();
        expect(screen.getByText('YES')).toBeInTheDocument();
    });

    it('renders object examples honoring input and output fields', () => {
        renderCard({
            ce: baseCe({
                examples: [
                    { input: 'great job', output: 'NO' },
                    { input: 'oh wonderful', output: 'YES' },
                ],
            }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('great job')).toBeInTheDocument();
        expect(screen.getByText('oh wonderful')).toBeInTheDocument();
        expect(screen.getByText('NO')).toBeInTheDocument();
        expect(screen.getByText('YES')).toBeInTheDocument();
    });

    it('treats a non-YES output verdict as the negative variant', () => {
        renderCard({
            ce: baseCe({ examples: [{ input: 'x', output: 'maybe' }] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        // Lowercase 'maybe' is rendered verbatim and is not 'YES'.
        expect(screen.getByText('maybe')).toBeInTheDocument();
    });

    it('defaults output to YES for object examples missing an output field', () => {
        renderCard({
            ce: baseCe({ examples: [{ input: 'no verdict' }] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        expect(screen.getByText('no verdict')).toBeInTheDocument();
        expect(screen.getByText('YES')).toBeInTheDocument();
    });

    it('renders an empty input for an object example missing input', () => {
        // Should not throw; the input span renders empty string.
        const { container } = renderCard({
            ce: baseCe({ examples: [{ output: 'YES' }] }),
            isOpen: true,
            onToggle: vi.fn(),
        });
        // The verdict pill still shows.
        expect(within(container).getByText('YES')).toBeInTheDocument();
    });
});


describe('CognitiveElementCard — combined affordances', () => {
    it('can show publish and delete together for a draft', () => {
        const { container } = renderCard({
            ce: baseCe({ is_local_draft: true }),
            isOpen: true,
            onToggle: vi.fn(),
            onPublish: vi.fn(),
            onDelete: vi.fn(),
        });
        expect(screen.getByRole('button', { name: /publish ce to library/i })).toBeInTheDocument();
        expect(container.querySelector('.delete-icon')).not.toBeNull();
    });
});
