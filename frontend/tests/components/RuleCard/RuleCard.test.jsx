// Behavior tests for RuleCard.
//
// RuleCard is a presentational card with a lot of conditional branches:
//   * header (status pill draft/public, category pills, author link,
//     bookmark / disabled-publish / rule-page / delete buttons, chevron)
//   * expanded body (groups + condition firing logic via RuleLogicPreview,
//     legacy predicate fallback, cognitive-elements list).
//
// Every interactive handler stops propagation so a click on a button does
// NOT bubble up to the header's onToggle — we assert that explicitly.
//
// We mock ../../api (RuleCard reads CE bookmarks) and sweetalert2 (used by
// alert dialogs) so nothing hits the network or pops a real modal.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, within } from '@testing-library/react';

// react-router's useNavigate — spy so we can assert navigation targets.
const navigateSpy = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => navigateSpy };
});

// RuleCard imports the CE-bookmark helpers; keep them benign so mounting the
// card never reaches the network.
vi.mock('../../../src/api', () => ({
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    addCEBookmark: vi.fn(() => Promise.resolve({ data: {} })),
    removeCEBookmark: vi.fn(() => Promise.resolve({ data: {} })),
}));

// showAlertDialog -> Swal.fire. Mock so the role-help button doesn't pop a modal.
const swalFire = vi.fn(() => Promise.resolve({ isConfirmed: true }));
vi.mock('sweetalert2', () => ({ default: { fire: (...a) => swalFire(...a), close: vi.fn() } }));

import RuleCard from '../../../src/components/RuleCard/RuleCard';

// A baseline LEGACY rule (no groups/condition — predicate fallback path).
const baseRule = () => ({
    setup_id: 7,
    rule_id: 99,
    custom_name: 'My Rule',
    predicate: 'A AND B',
    active_ces: [
        { ce_id: 1, name: 'CE One' },
        { ce_id: 2, name: 'CE Two' },
    ],
});

// A v2 rule with named groups + a condition.
const groupedRule = () => ({
    ...baseRule(),
    groups: {
        required: [{ ce_id: 1, name: 'CE One' }],
        option_1: [{ ce_id: 2, name: 'CE Two' }, { ce_id: 3, name: 'CE Three' }],
    },
    condition: 'all of required and 1 of option_1',
});

const renderCard = (props = {}) => render(
    <MemoryRouter>
        <RuleCard rule={baseRule()} isExpanded={false} onToggle={() => {}} {...props} />
    </MemoryRouter>,
);

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
});

describe('RuleCard — header / collapsed view', () => {
    it('renders the rule name and CE count summary', () => {
        renderCard();
        expect(screen.getByText('My Rule')).toBeInTheDocument();
        // 2 active CEs, private rule (readOnly false by default)
        expect(screen.getByText(/2 Cognitive Elements • Private Rule/)).toBeInTheDocument();
    });

    it('labels a readOnly card as a Public Rule', () => {
        renderCard({ readOnly: true });
        expect(screen.getByText(/Public Rule/)).toBeInTheDocument();
    });

    it('shows a chevron-down when collapsed and nothing of the expanded body', () => {
        renderCard({ isExpanded: false });
        // Boolean Logic only renders in the expanded body.
        expect(screen.queryByText('Boolean Logic')).not.toBeInTheDocument();
    });

    it('calls onToggle when the header is clicked', () => {
        const onToggle = vi.fn();
        const { container } = renderCard({ onToggle });
        fireEvent.click(container.querySelector('.rule-header'));
        expect(onToggle).toHaveBeenCalledTimes(1);
    });

    it('applies the expanded class to the card when isExpanded', () => {
        const { container } = renderCard({ isExpanded: true });
        expect(container.querySelector('.rule-card')).toHaveClass('expanded');
    });
});

describe('RuleCard — status pill', () => {
    it('shows a Draft pill when is_local_draft is true', () => {
        const rule = { ...baseRule(), is_local_draft: true };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.getByText('Draft')).toBeInTheDocument();
    });

    it('shows a Public pill when is_local_draft is false', () => {
        const rule = { ...baseRule(), is_local_draft: false };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.getByText('Public')).toBeInTheDocument();
    });

    it('shows no pill when is_local_draft is undefined (not a boolean)', () => {
        renderCard();
        expect(screen.queryByText('Draft')).not.toBeInTheDocument();
        expect(screen.queryByText('Public')).not.toBeInTheDocument();
    });
});

describe('RuleCard — categories & author link', () => {
    it('renders category pills when categories are present', () => {
        const rule = { ...baseRule(), categories: ['Safety', 'Bias'] };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.getByText('Safety')).toBeInTheDocument();
        expect(screen.getByText('Bias')).toBeInTheDocument();
    });

    it('renders the author link inside the categories row when both exist', () => {
        const rule = { ...baseRule(), categories: ['Safety'], created_by_username: 'alice' };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        const link = screen.getByRole('link', { name: /@alice/ });
        expect(link).toHaveAttribute('href', '/profile/alice');
    });

    it('renders the author link on its own row when there are no categories', () => {
        const rule = { ...baseRule(), created_by_username: 'bob' };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.getByRole('link', { name: /@bob/ })).toHaveAttribute('href', '/profile/bob');
    });

    it('does not bubble a click on the author link to onToggle', () => {
        const onToggle = vi.fn();
        const rule = { ...baseRule(), created_by_username: 'bob' };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={onToggle} /></MemoryRouter>);
        fireEvent.click(screen.getByRole('link', { name: /@bob/ }));
        expect(onToggle).not.toHaveBeenCalled();
    });
});

describe('RuleCard — bookmark button', () => {
    const bookmarkable = () => ({ ...baseRule(), is_local_draft: false, public_id: 'pub-1' });

    it('renders the bookmark button (with default label) only on a non-draft rule with a public_id', () => {
        const onBookmark = vi.fn();
        render(<MemoryRouter><RuleCard rule={bookmarkable()} isExpanded={false} onToggle={() => {}} onBookmark={onBookmark} /></MemoryRouter>);
        expect(screen.getByRole('button', { name: 'Bookmark rule' })).toHaveTextContent('Save');
    });

    it('uses a custom bookmarkLabel', () => {
        render(<MemoryRouter><RuleCard rule={bookmarkable()} isExpanded={false} onToggle={() => {}} onBookmark={() => {}} bookmarkLabel="Add" /></MemoryRouter>);
        expect(screen.getByRole('button', { name: 'Bookmark rule' })).toHaveTextContent('Add');
    });

    it('shows "Remove" when already bookmarked', () => {
        render(<MemoryRouter><RuleCard rule={bookmarkable()} isExpanded={false} onToggle={() => {}} onBookmark={() => {}} isBookmarked /></MemoryRouter>);
        expect(screen.getByRole('button', { name: 'Bookmark rule' })).toHaveTextContent('Remove');
    });

    it('calls onBookmark with the rule and does not toggle the card', () => {
        const onBookmark = vi.fn();
        const onToggle = vi.fn();
        const rule = bookmarkable();
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={onToggle} onBookmark={onBookmark} /></MemoryRouter>);
        fireEvent.click(screen.getByRole('button', { name: 'Bookmark rule' }));
        expect(onBookmark).toHaveBeenCalledWith(rule);
        expect(onToggle).not.toHaveBeenCalled();
    });

    it('hides the bookmark button on a draft even with a public_id', () => {
        const rule = { ...baseRule(), is_local_draft: true, public_id: 'pub-1' };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} onBookmark={() => {}} /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: 'Bookmark rule' })).not.toBeInTheDocument();
    });

    it('hides the bookmark button when there is no public_id', () => {
        const rule = { ...baseRule(), is_local_draft: false };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} onBookmark={() => {}} /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: 'Bookmark rule' })).not.toBeInTheDocument();
    });

    it('hides the bookmark button when onBookmark is not provided', () => {
        render(<MemoryRouter><RuleCard rule={bookmarkable()} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: 'Bookmark rule' })).not.toBeInTheDocument();
    });
});

describe('RuleCard — publish button (disabled: PR-based contributions)', () => {
    it('shows a DISABLED publish button on a draft with the PR tooltip', () => {
        const rule = { ...baseRule(), is_local_draft: true };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        const btn = screen.getByRole('button', { name: 'Publish rule to library (coming soon)' });
        expect(btn).toBeDisabled();
        expect(btn).toHaveAttribute('title', expect.stringContaining('gavel-rules pull requests'));
    });

    it('hides the publish button when the rule is not a draft', () => {
        const rule = { ...baseRule(), is_local_draft: false };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: /Publish rule to library/ })).not.toBeInTheDocument();
    });
});

describe('RuleCard — rule page navigation', () => {
    it('navigates using source_rule_id when present', () => {
        const rule = { ...baseRule(), source_rule_id: 555 };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        fireEvent.click(screen.getByRole('button', { name: "Open this rule's page" }));
        expect(navigateSpy).toHaveBeenCalledWith('/rules/555');
    });

    it('falls back to rule_id when there is no source_rule_id', () => {
        renderCard();
        fireEvent.click(screen.getByRole('button', { name: "Open this rule's page" }));
        expect(navigateSpy).toHaveBeenCalledWith('/rules/99');
    });

    it('does not bubble the navigation click to onToggle', () => {
        const onToggle = vi.fn();
        renderCard({ onToggle });
        fireEvent.click(screen.getByRole('button', { name: "Open this rule's page" }));
        expect(onToggle).not.toHaveBeenCalled();
    });

    it('hides the rule-page button when there is no nav id', () => {
        const rule = { ...baseRule(), rule_id: undefined };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={() => {}} /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: "Open this rule's page" })).not.toBeInTheDocument();
    });
});

describe('RuleCard — delete affordance', () => {
    it('shows the delete icon and calls onDelete with setup_id when not readOnly', () => {
        const onDelete = vi.fn();
        const onToggle = vi.fn();
        const { container } = renderCard({ onDelete, onToggle });
        const del = container.querySelector('.delete-icon');
        expect(del).toBeInTheDocument();
        fireEvent.click(del);
        expect(onDelete).toHaveBeenCalledWith(7);
        expect(onToggle).not.toHaveBeenCalled();
    });

    it('hides the delete icon in readOnly mode', () => {
        const { container } = renderCard({ readOnly: true });
        expect(container.querySelector('.delete-icon')).not.toBeInTheDocument();
    });
});

describe('RuleCard — expanded body, read view', () => {
    it('falls back to the predicate string when the rule has no groups', () => {
        renderCard({ isExpanded: true });
        expect(screen.getByText('Boolean Logic')).toBeInTheDocument();
        expect(screen.getByText('A AND B')).toBeInTheDocument();
    });

    it('renders group chips titled by group name plus the condition verbatim', () => {
        render(<MemoryRouter><RuleCard rule={groupedRule()} isExpanded onToggle={() => {}} /></MemoryRouter>);
        // Group name headers
        expect(screen.getByText('required')).toBeInTheDocument();
        expect(screen.getByText('option_1')).toBeInTheDocument();
        // Member pills (CE One appears in both the group chip and the CE list)
        expect(screen.getAllByText('CE One').length).toBeGreaterThan(0);
        expect(screen.getByText('CE Three')).toBeInTheDocument();
        // Condition rendered verbatim
        expect(screen.getByText('all of required and 1 of option_1')).toBeInTheDocument();
        // The raw predicate string is NOT shown when groups exist
        expect(screen.queryByText('A AND B')).not.toBeInTheDocument();
    });

    it('lists every CE name (membership, no role badges)', () => {
        renderCard({ isExpanded: true });
        expect(screen.getByText('CE One')).toBeInTheDocument();
        expect(screen.getByText('CE Two')).toBeInTheDocument();
        expect(screen.queryByText('Necessary')).not.toBeInTheDocument();
        expect(screen.queryByText('Supporting')).not.toBeInTheDocument();
    });

    it('renders string CEs (ce is a bare string)', () => {
        const rule = { ...baseRule(), active_ces: ['Just A String'] };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded onToggle={() => {}} /></MemoryRouter>);
        expect(screen.getByText('Just A String')).toBeInTheDocument();
    });

    it('handles a rule with no active_ces array', () => {
        const rule = { ...baseRule(), active_ces: undefined };
        render(<MemoryRouter><RuleCard rule={rule} isExpanded onToggle={() => {}} /></MemoryRouter>);
        // Summary reads 0 elements, body still renders the elements header.
        expect(screen.getByText('Cognitive Elements')).toBeInTheDocument();
    });

    it('does not show remove-CE or add-CE in the default read view', () => {
        const { container } = renderCard({ isExpanded: true });
        expect(container.querySelector('.remove-ce-btn')).not.toBeInTheDocument();
        expect(screen.queryByText('+ Add CE')).not.toBeInTheDocument();
    });


});

describe('RuleCard — edit logic affordance', () => {
    it('shows Edit logic only when onEditLogic is wired and not readOnly', () => {
        const onEditLogic = vi.fn();
        const onToggle = vi.fn();
        // Needs a signed-in user for nothing here — Edit logic is gated only
        // on the prop + readOnly.
        const rule = groupedRule();
        render(<MemoryRouter><RuleCard rule={rule} isExpanded={false} onToggle={onToggle} onEditLogic={onEditLogic} /></MemoryRouter>);
        const btn = screen.getByRole('button', { name: "Edit this rule's groups and condition in place" });
        fireEvent.click(btn);
        expect(onEditLogic).toHaveBeenCalledWith(rule);
        expect(onToggle).not.toHaveBeenCalled();
    });

    it('hides Edit logic in readOnly mode', () => {
        render(<MemoryRouter><RuleCard rule={groupedRule()} isExpanded={false} onToggle={() => {}} onEditLogic={() => {}} readOnly /></MemoryRouter>);
        expect(screen.queryByRole('button', { name: "Edit this rule's groups and condition in place" })).not.toBeInTheDocument();
    });
});
