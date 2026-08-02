// Behavior tests for the BookmarksRules page.
//
// BookmarksRules fetches the user's bookmarked rules (getRuleBookmarks +
// getPublicRules), renders them as RuleCards inside a SearchPanel, supports
// local + server-side search (searchBookmarks), pagination, expand/collapse,
// and removing bookmarks (removeRuleBookmark). These tests drive each of
// those paths deterministically by controlling the mocked '../api' returns.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// --- Mock the navigate hook so we can assert navigations without a real router move.
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- Create-entry-point modals are stubbed (same as Browse.test.jsx): the
// real ones call useTaskTray() on render, which needs a TaskTrayProvider this
// harness doesn't mount. We only care that they render when open.
vi.mock('../../src/pages/RuleGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="rule-modal-open" /> : null),
}));
vi.mock('../../src/pages/BuildRuleFromCEsModal', () => ({
    default: ({ open }) => (open ? <div data-testid="build-from-ces-modal-open" /> : null),
}));

// --- API mock. Cover every export BookmarksRules + Layout/Sidebar + RuleCard
// might call so nothing hits the network. Per-test overrides use
// mockResolvedValueOnce / mockImplementation on these spies.
vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return {
        default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()), delete: vi.fn(() => empty()), put: vi.fn(() => empty()) },
        getCEBookmarks: vi.fn(() => empty({ bookmarks: [] })),
        addCEBookmark: vi.fn(() => empty()),
        removeCEBookmark: vi.fn(() => empty()),
        getRuleBookmarks: vi.fn(() => empty({ bookmarks: [] })),
        removeRuleBookmark: vi.fn(() => empty()),
        getPublicRules: vi.fn(() => empty({ rules: [] })),
        searchBookmarks: vi.fn(() => empty({ results: [], total_results: 0 })),
        // Sidebar / Layout might touch these on mount (Sidebar is stubbed, but
        // keep them benign just in case).
        getUserModels: vi.fn(() => empty({ models: [] })),
        getClassifiers: vi.fn(() => empty({ classifiers: [] })),
    };
});

// Stub Sidebar — it has its own fetches/routing irrelevant to this page.
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

// Stub the alert dialog so handleRemoveBookmark doesn't open a real Swal.
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: vi.fn(() => Promise.resolve()),
    showConfirmDialog: vi.fn(() => Promise.resolve(true)),
    showLoadingDialog: vi.fn(() => () => {}),
    escapeHtml: (s) => String(s ?? ''),
}));

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));

import * as api from '../../src/api';
import { showAlertDialog } from '../../src/components/ConfirmDialog/confirmDialog';
import BookmarksRules from '../../src/pages/BookmarksRules';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderPage = () => render(
    <MemoryRouter initialEntries={['/bookmarks/rules']}>
        <Routes>
            <Route path="/bookmarks/rules" element={<BookmarksRules />} />
        </Routes>
    </MemoryRouter>,
);

// Build a public rule row as returned by getPublicRules.
const makeRule = (id, overrides = {}) => ({
    rule_id: id,
    name: `Rule ${id}`,
    predicate: `predicate-${id}`,
    public_id: `pub-${id}`,
    is_local_draft: false,
    categories: [`Cat${id}`],
    active_ces: [{ name: `CE${id}` }],
    ...overrides,
});

beforeEach(() => {
    vi.clearAllMocks();
    setUser();
    // Re-apply benign defaults after clearAllMocks wiped implementations.
    api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.getPublicRules.mockResolvedValue({ data: { rules: [] } });
    api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });
    api.removeRuleBookmark.mockResolvedValue({ data: {} });
});

describe('BookmarksRules — mount', () => {
    it('fetches bookmarks for the logged-in user on mount', async () => {
        renderPage();
        await waitFor(() => expect(api.getRuleBookmarks).toHaveBeenCalledWith(7));
    });

    it('renders the page chrome (headings + tab buttons)', async () => {
        renderPage();
        expect(await screen.findByText('My Bookmarked Rules')).toBeInTheDocument();
        expect(screen.getByText('My Rules')).toBeInTheDocument();
        expect(screen.getByText('My CEs')).toBeInTheDocument();
        expect(screen.getByText(/Back to Community/i)).toBeInTheDocument();
    });
});

describe('BookmarksRules — empty state', () => {
    it('shows the empty-bookmarks message and a Browse button when none exist', async () => {
        renderPage();
        expect(await screen.findByText('No Rules Found')).toBeInTheDocument();
        expect(screen.getByText("You haven't bookmarked or drafted any rules yet.")).toBeInTheDocument();
        expect(screen.getByText('Browse Public Rules')).toBeInTheDocument();
    });

    it('Browse Public Rules button navigates to /browse', async () => {
        renderPage();
        const btn = await screen.findByText('Browse Public Rules');
        fireEvent.click(btn);
        expect(mockNavigate).toHaveBeenCalledWith('/browse');
    });

    it('skips getPublicRules when there are no bookmark ids', async () => {
        renderPage();
        await screen.findByText('No Rules Found');
        expect(api.getPublicRules).not.toHaveBeenCalled();
    });
});

describe('BookmarksRules — loaded bookmarks', () => {
    beforeEach(() => {
        api.getRuleBookmarks.mockResolvedValue({
            data: { bookmarks: [{ rule_id: 1 }, { rule_id: 2 }] },
        });
        api.getPublicRules.mockResolvedValue({
            data: { rules: [makeRule(1), makeRule(2), makeRule(3)] },
        });
    });

    it('renders only the bookmarked rules (filtered by id)', async () => {
        renderPage();
        expect(await screen.findByText('Rule 1')).toBeInTheDocument();
        expect(screen.getByText('Rule 2')).toBeInTheDocument();
        // Rule 3 was public but not bookmarked → excluded.
        expect(screen.queryByText('Rule 3')).not.toBeInTheDocument();
    });

    it('expands a rule card when its header is clicked', async () => {
        renderPage();
        const title = await screen.findByText('Rule 1');
        // Collapsed: predicate code-box is hidden.
        expect(screen.queryByText('predicate-1')).not.toBeInTheDocument();
        fireEvent.click(title);
        expect(await screen.findByText('predicate-1')).toBeInTheDocument();
    });

    it('collapses an expanded card on a second click', async () => {
        renderPage();
        const title = await screen.findByText('Rule 1');
        fireEvent.click(title);
        await screen.findByText('predicate-1');
        fireEvent.click(title);
        await waitFor(() => expect(screen.queryByText('predicate-1')).not.toBeInTheDocument());
    });

    it('derives available categories from the bookmarked rules', async () => {
        renderPage();
        await screen.findByText('Rule 1');
        // SearchPanel renders category option buttons from availableCategories.
        // (Categories also appear as pills inside each RuleCard, so there may be
        // more than one match — assert presence via getAllByText.)
        expect(screen.getAllByText('Cat1').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Cat2').length).toBeGreaterThan(0);
        // The selectable category option lives inside the search panel.
        const panel = document.querySelector('.search-panel');
        expect(within(panel).getByText('Cat1')).toBeInTheDocument();
    });
});

describe('BookmarksRules — remove bookmark', () => {
    beforeEach(() => {
        api.getRuleBookmarks.mockResolvedValue({
            data: { bookmarks: [{ rule_id: 1 }, { rule_id: 2 }] },
        });
        api.getPublicRules.mockResolvedValue({
            data: { rules: [makeRule(1), makeRule(2)] },
        });
    });

    it('removes a bookmark and shows a success alert', async () => {
        renderPage();
        await screen.findByText('Rule 1');
        const removeButtons = screen.getAllByRole('button', { name: /Bookmark rule/i });
        fireEvent.click(removeButtons[0]);

        await waitFor(() => expect(api.removeRuleBookmark).toHaveBeenCalledWith(7, expect.any(Number)));
        await waitFor(() =>
            expect(showAlertDialog).toHaveBeenCalledWith(
                expect.objectContaining({ variant: 'success' }),
            ),
        );
        // One card should be gone.
        await waitFor(() => expect(screen.queryByText('Rule 2')).not.toBeInTheDocument()
            || screen.queryByText('Rule 1') === null);
    });

    it('shows an error alert when removal fails', async () => {
        api.removeRuleBookmark.mockRejectedValueOnce(new Error('boom'));
        renderPage();
        await screen.findByText('Rule 1');
        const removeButtons = screen.getAllByRole('button', { name: /Bookmark rule/i });
        fireEvent.click(removeButtons[0]);
        await waitFor(() =>
            expect(showAlertDialog).toHaveBeenCalledWith(
                expect.objectContaining({ variant: 'error' }),
            ),
        );
    });
});

describe('BookmarksRules — server-side search', () => {
    beforeEach(() => {
        api.getRuleBookmarks.mockResolvedValue({
            data: { bookmarks: [{ rule_id: 1 }, { rule_id: 2 }] },
        });
        api.getPublicRules.mockResolvedValue({
            data: { rules: [makeRule(1), makeRule(2)] },
        });
    });

    it('runs searchBookmarks when a query is typed and shows the results header', async () => {
        api.searchBookmarks.mockResolvedValue({
            data: {
                results: [{ id: 99, name: 'Found Rule', content: 'c', ces: ['X'], categories: ['Z'], public_id: 'pub-99', is_local_draft: false }],
                total_results: 1,
            },
        });
        renderPage();
        await screen.findByText('Rule 1');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'found' } });

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        const call = api.searchBookmarks.mock.calls.at(-1)[0];
        expect(call).toMatchObject({ user_id: 7, q: 'found', asset_types: 'rule' });

        expect(await screen.findByText(/Search Results \(1 found\)/)).toBeInTheDocument();
        expect(screen.getByText('Found Rule')).toBeInTheDocument();
    });

    it('falls back to local filtering when searchBookmarks rejects', async () => {
        api.searchBookmarks.mockRejectedValue(new Error('server down'));
        renderPage();
        await screen.findByText('Rule 1');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        // Query matches only Rule 1's name.
        fireEvent.change(input, { target: { value: 'Rule 1' } });

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        // Local applyFilters keeps only "Rule 1".
        await waitFor(() => expect(screen.queryByText('Rule 2')).not.toBeInTheDocument());
        expect(screen.getByText('Rule 1')).toBeInTheDocument();
    });

    it('clearing the query resets to showing all bookmarks (no search header)', async () => {
        api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderPage();
        await screen.findByText('Rule 1');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'zzz' } });
        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());

        fireEvent.change(input, { target: { value: '' } });
        await waitFor(() => expect(screen.queryByText(/Search Results/)).not.toBeInTheDocument());
        // Both bookmarks visible again.
        expect(screen.getByText('Rule 1')).toBeInTheDocument();
        expect(screen.getByText('Rule 2')).toBeInTheDocument();
    });
});

describe('BookmarksRules — category filtering triggers search', () => {
    it('selecting a category triggers a server search with that category', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 1 }] } });
        api.getPublicRules.mockResolvedValue({ data: { rules: [makeRule(1)] } });
        api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });

        renderPage();
        await screen.findByText('Rule 1');

        // Cat1 is rendered as a selectable category option in SearchPanel.
        const panel = document.querySelector('.search-panel');
        fireEvent.click(within(panel).getByText('Cat1'));

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        const call = api.searchBookmarks.mock.calls.at(-1)[0];
        expect(call.categories).toContain('Cat1');
    });
});

describe('BookmarksRules — pagination', () => {
    it('renders pagination and changing page re-runs server search when searched', async () => {
        // 15 bookmarks so that with page_size 10 there is a second page.
        const bookmarks = Array.from({ length: 15 }, (_, i) => ({ rule_id: i + 1 }));
        const rules = Array.from({ length: 15 }, (_, i) => makeRule(i + 1));
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks } });
        api.getPublicRules.mockResolvedValue({ data: { rules } });
        api.searchBookmarks.mockResolvedValue({
            data: { results: rules.slice(0, 10).map(r => ({ id: r.rule_id, name: r.name, content: r.predicate, ces: [], categories: [], public_id: r.public_id, is_local_draft: false })), total_results: 15 },
        });

        renderPage();
        // Rules are sorted newest-first, so page 1 of 15 shows Rule 15 first.
        await screen.findByText('Rule 15');

        // Trigger a search so hasSearched is true and pagination drives searchBookmarks.
        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'Rule' } });
        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());

        // Pagination shows because total_results (15) > page_size (10).
        const page2 = await screen.findByRole('button', { name: '2' });
        api.searchBookmarks.mockClear();
        fireEvent.click(page2);

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        const call = api.searchBookmarks.mock.calls.at(-1)[0];
        expect(call.page).toBe(2);
    });

    it('does not render pagination for a single page of bookmarks', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 1 }] } });
        api.getPublicRules.mockResolvedValue({ data: { rules: [makeRule(1)] } });
        renderPage();
        await screen.findByText('Rule 1');
        // Pagination returns null when totalItems <= pageSize.
        expect(screen.queryByRole('button', { name: '2' })).not.toBeInTheDocument();
    });
});

describe('BookmarksRules — fetch error + normalization edge cases', () => {
    it('falls back to empty state when getRuleBookmarks rejects', async () => {
        api.getRuleBookmarks.mockRejectedValue(new Error('network'));
        renderPage();
        expect(await screen.findByText('No Rules Found')).toBeInTheDocument();
    });

    it('handles bookmark rows using id instead of rule_id and rules using id', async () => {
        // Bookmark identifies via `id`; public rule identifies via `id`.
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ id: 5 }] } });
        api.getPublicRules.mockResolvedValue({
            data: { rules: [{ id: 5, name: 'IdRule', logic: 'log', public_id: 'pub-5', is_local_draft: false, required_ces: ['ReqCE'] }] },
        });
        renderPage();
        expect(await screen.findByText('IdRule')).toBeInTheDocument();
    });

    it('reads rules from a bare array payload (publicRes.data is an array)', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 8 }] } });
        api.getPublicRules.mockResolvedValue({ data: [makeRule(8)] });
        renderPage();
        expect(await screen.findByText('Rule 8')).toBeInTheDocument();
    });
});

describe('BookmarksRules — SearchPanel reset', () => {
    it('Reset All restores the full bookmark list and clears the search header', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 1 }, { rule_id: 2 }] } });
        api.getPublicRules.mockResolvedValue({ data: { rules: [makeRule(1), makeRule(2)] } });
        api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderPage();
        await screen.findByText('Rule 1');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'xyz' } });
        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());

        fireEvent.click(screen.getByTitle('Reset all filters'));
        await waitFor(() => expect(screen.queryByText(/Search Results/)).not.toBeInTheDocument());
        expect(screen.getByText('Rule 1')).toBeInTheDocument();
        expect(screen.getByText('Rule 2')).toBeInTheDocument();
    });
});

describe('BookmarksRules — tab navigation', () => {
    it('My CEs button navigates to /bookmarks/ces', async () => {
        renderPage();
        await screen.findByText('My Bookmarked Rules');
        fireEvent.click(screen.getByText('My CEs'));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/ces');
    });

    it('Back to Community navigates to /community', async () => {
        renderPage();
        await screen.findByText('My Bookmarked Rules');
        fireEvent.click(screen.getByText(/Back to Community/i));
        expect(mockNavigate).toHaveBeenCalledWith('/community');
    });
});
