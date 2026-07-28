// Behavior tests for the Browse (public rules) page.
//
// Browse lists both the public rule library AND the requester's own local
// drafts (merged, drafts first), drives a live debounced search through
// useLibrarySearch (which calls searchLibrary from ../api), supports
// bookmarking public rules, publishing drafts (delegated to RuleService),
// category filtering, pagination, and the two dedicated header CTAs
// ("Create Rule with AI" + "Build Rule from CEs").
//
// We mock ../api (every export the page + its children touch), the publish
// service, the confirm dialog helpers, and stub Sidebar (rendered by Layout)
// so nothing hits the network or pops a real modal. Router useNavigate is a
// spy; useSearchParams stays real via MemoryRouter so the ?author= chip works.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// ---- navigate spy; keep the rest of react-router-dom real ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// ---- API mock: benign defaults for everything Browse + children call ----
const empty = (extra = {}) => Promise.resolve({ data: extra });
vi.mock('../../src/api', () => ({
    default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()) },
    getCEBookmarks: vi.fn(() => empty({ bookmarks: [] })),
    addCEBookmark: vi.fn(() => empty()),
    removeCEBookmark: vi.fn(() => empty()),
    getPublicRules: vi.fn(() => empty({ rules: [] })),
    listLocalDrafts: vi.fn(() => empty({ rules: [] })),
    getAllCategories: vi.fn(() => empty([])),
    getRuleBookmarks: vi.fn(() => empty({ bookmarks: [] })),
    addRuleBookmark: vi.fn(() => empty()),
    removeRuleBookmark: vi.fn(() => empty()),
    // useLibrarySearch reaches for this; default = no results.
    searchLibrary: vi.fn(() => empty({ results: [], total_results: 0 })),
}));

// ---- Publish service: assert it's called, never run the real pipeline ----
vi.mock('../../src/services/RuleService', () => ({
    publishDraftRule: vi.fn(),
}));

// ---- confirm dialog helpers ----
const mockShowAlertDialog = vi.fn(() => Promise.resolve());
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: (...a) => mockShowAlertDialog(...a),
    showConfirmDialog: vi.fn(() => Promise.resolve(true)),
}));

// ---- Sidebar stub (Layout renders it) ----
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

// The AI-rule modal pulls in the task-tray context + wizard machinery; it has
// its own coverage. Here it's a lightweight stub that just reflects its `open`
// prop so we can assert the button opens it (instead of the old navigation).
vi.mock('../../src/pages/RuleGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="rule-modal-open" /> : null),
}));
vi.mock('../../src/pages/BuildRuleFromCEsModal', () => ({
    default: ({ open }) => (open ? <div data-testid="build-from-ces-modal-open" /> : null),
}));

// ---- sweetalert2 stub ----
vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })), close: vi.fn() },
}));

import Browse from '../../src/pages/Browse';
import * as api from '../../src/api';
import * as RuleService from '../../src/services/RuleService';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderBrowse = (entry = '/browse') =>
    render(
        <TutorialProvider>
            <MemoryRouter initialEntries={[entry]}>
                <Routes>
                    <Route path="/browse" element={<Browse />} />
                </Routes>
            </MemoryRouter>
        </TutorialProvider>,
    );

// A public library rule (has public_id => bookmarkable, is_local_draft false).
const publicRule = (over = {}) => ({
    rule_id: 100,
    id: 100,
    setup_id: 100,
    name: 'Public Safety Rule',
    public_id: 'pub-100',
    predicate: 'A AND B',
    active_ces: [{ ce_id: 1, name: 'CE One', role: 'necessary' }],
    is_local_draft: false,
    categories: ['Safety & Harm Prevention'],
    ...over,
});

// A local draft rule (publishable, not bookmarkable).
const draftRule = (over = {}) => ({
    rule_id: 200,
    id: 200,
    setup_id: 200,
    name: 'My Draft Rule',
    predicate: 'C OR D',
    active_ces: [{ ce_id: 2, name: 'CE Two', role: 'necessary' }],
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    api.getPublicRules.mockResolvedValue({ data: { rules: [] } });
    api.listLocalDrafts.mockResolvedValue({ data: { rules: [] } });
    api.getAllCategories.mockResolvedValue({ data: [] });
    api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.addRuleBookmark.mockResolvedValue({ data: {} });
    api.removeRuleBookmark.mockResolvedValue({ data: {} });
    api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
    mockShowAlertDialog.mockResolvedValue(undefined);
});

describe('Browse — mount', () => {
    it('fetches rules, categories and bookmarks on mount when authenticated', async () => {
        renderBrowse();
        await waitFor(() => expect(api.getPublicRules).toHaveBeenCalled());
        // Browse is the PUBLIC space — it no longer pulls the user's drafts.
        expect(api.listLocalDrafts).not.toHaveBeenCalled();
        expect(api.getAllCategories).toHaveBeenCalled();
        await waitFor(() => expect(api.getRuleBookmarks).toHaveBeenCalledWith(7));
    });

    it('renders the header, intro copy and the two browse tabs', async () => {
        renderBrowse();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Public Rules' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Rules' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'CEs' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Contributors' })).toBeInTheDocument();
        await waitFor(() => expect(api.getPublicRules).toHaveBeenCalled());
    });
});

describe('Browse — header CTAs & navigation', () => {
    // Create CTAs moved out of the public Community pages into the sidebar
    // Create menu, so the old in-page create-button tests were removed.

    it('navigates back to the hub and to the CE browser', async () => {
        renderBrowse();
        fireEvent.click(screen.getByText('Hub'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
        fireEvent.click(screen.getByRole('button', { name: 'CEs' }));
        expect(mockNavigate).toHaveBeenCalledWith('/community/ces');
    });

    it('navigates to the bookmarks page via "View all"', async () => {
        renderBrowse();
        await waitFor(() => expect(api.getPublicRules).toHaveBeenCalled());
        fireEvent.click(screen.getByRole('button', { name: 'View all' }));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/rules');
    });
});

describe('Browse — empty state', () => {
    it('shows the "No Public Rules" empty state when nothing is returned', async () => {
        renderBrowse();
        expect(await screen.findByText('No Public Rules')).toBeInTheDocument();
        expect(screen.getByText(/Start by searching for rules/i)).toBeInTheDocument();
    });

    it('shows "No bookmarks yet" when the user has none', async () => {
        renderBrowse();
        expect(await screen.findByText('No bookmarks yet.')).toBeInTheDocument();
    });
});

describe('Browse — error handling', () => {
    it('falls back to an empty list when fetching rules rejects', async () => {
        api.getPublicRules.mockRejectedValue(new Error('boom'));
        renderBrowse();
        // listLocalDrafts also rejects -> caught path -> empty.
        expect(await screen.findByText('No Public Rules')).toBeInTheDocument();
    });

    it('still renders public rules when the drafts fetch rejects', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        api.listLocalDrafts.mockRejectedValue(new Error('drafts down'));
        renderBrowse();
        // Promise.all + .catch on drafts means public rules survive.
        expect(await screen.findByText('Public Safety Rule')).toBeInTheDocument();
    });

    it('clears bookmarks to empty when the bookmark fetch rejects', async () => {
        api.getRuleBookmarks.mockRejectedValue(new Error('nope'));
        renderBrowse();
        expect(await screen.findByText('No bookmarks yet.')).toBeInTheDocument();
    });
});

describe('Browse — public & draft rule listing', () => {
    it('renders public library rules under "All Public Rules"', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        renderBrowse();
        expect(await screen.findByText('Public Safety Rule')).toBeInTheDocument();
        expect(screen.getByText('All Public Rules')).toBeInTheDocument();
    });

    it('does NOT show local drafts — Browse is the public space', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        api.listLocalDrafts.mockResolvedValue({ data: { rules: [draftRule()] } });
        renderBrowse();
        expect(await screen.findByText('Public Safety Rule')).toBeInTheDocument();
        // The user's unpublished draft must not appear in the public Browse.
        expect(screen.queryByText('My Draft Rule')).not.toBeInTheDocument();
    });

    it('tolerates a payload where rules is a bare array (no .rules key)', async () => {
        api.getPublicRules.mockResolvedValue({ data: [publicRule({ name: 'Bare Array Rule' })] });
        renderBrowse();
        expect(await screen.findByText('Bare Array Rule')).toBeInTheDocument();
    });

    it('falls back to a default name when a rule has no name/title', async () => {
        api.getPublicRules.mockResolvedValue({
            data: { rules: [publicRule({ name: undefined, title: undefined, custom_name: undefined })] },
        });
        renderBrowse();
        // normalizeRule defaults custom_name to 'Rule'.
        expect(await screen.findByText('Rule')).toBeInTheDocument();
    });

    it('expands a rule card to reveal its predicate', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        renderBrowse();
        const title = await screen.findByText('Public Safety Rule');
        fireEvent.click(title);
        expect(await screen.findByText('A AND B')).toBeInTheDocument();
    });
});

describe('Browse — bookmark on a public rule', () => {
    it('adds a bookmark and shows a Saved alert', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        renderBrowse();
        await screen.findByText('Public Safety Rule');
        fireEvent.click(screen.getByRole('button', { name: 'Bookmark rule' }));
        await waitFor(() => expect(api.addRuleBookmark).toHaveBeenCalledWith(7, 100));
        await waitFor(() => expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Saved' }),
        ));
        // The new bookmark appears in the sidebar list.
        expect(await screen.findByText('Public Safety Rule', { selector: 'span' })).toBeInTheDocument();
    });

    it('removes an already-bookmarked rule and shows a Removed alert', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 100, name: 'Sidebar Saved' }] } });
        renderBrowse();
        await screen.findByText('Public Safety Rule');
        // The card button reads "Remove" once already bookmarked.
        const cardBtn = screen.getByRole('button', { name: 'Bookmark rule' });
        expect(cardBtn).toHaveTextContent('Remove');
        fireEvent.click(cardBtn);
        await waitFor(() => expect(api.removeRuleBookmark).toHaveBeenCalledWith(7, 100));
        await waitFor(() => expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Removed' }),
        ));
    });

    it('shows an error alert when the bookmark call fails', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        api.addRuleBookmark.mockRejectedValue(new Error('fail'));
        renderBrowse();
        await screen.findByText('Public Safety Rule');
        fireEvent.click(screen.getByRole('button', { name: 'Bookmark rule' }));
        await waitFor(() => expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Error' }),
        ));
    });

    it('removes a bookmark from the sidebar list via its Remove button', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 100, name: 'Saved Rule' }] } });
        renderBrowse();
        expect(await screen.findByText('Saved Rule')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Remove bookmark' }));
        await waitFor(() => expect(api.removeRuleBookmark).toHaveBeenCalledWith(7, 100));
    });
});

describe('Browse — publish a draft rule', () => {
    // Drafts no longer appear in Browse (they live in Your Library), so the only
    // remaining expectation here is that public rules never show a Publish button.
    it('does not show a Publish button on a public (non-draft) rule', async () => {
        api.getPublicRules.mockResolvedValue({ data: { rules: [publicRule()] } });
        renderBrowse();
        await screen.findByText('Public Safety Rule');
        expect(screen.queryByRole('button', { name: 'Publish rule to library' })).not.toBeInTheDocument();
    });
});

describe('Browse — live search', () => {
    it('shows search results and a count when searchLibrary returns matches', async () => {
        api.searchLibrary.mockResolvedValue({
            data: {
                results: [{ id: 300, asset_type: 'rule', name: 'Found Rule', logic: 'X', public_id: 'pub-300' }],
                total_results: 1,
            },
        });
        renderBrowse();
        const input = screen.getByPlaceholderText('Search public rules...');
        fireEvent.change(input, { target: { value: 'safety' } });
        expect(await screen.findByText('Found Rule')).toBeInTheDocument();
        expect(screen.getByText(/Search Results \(1 found\)/)).toBeInTheDocument();
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
    });

    it('filters out non-rule asset types from the search results', async () => {
        api.searchLibrary.mockResolvedValue({
            data: {
                results: [
                    { id: 301, asset_type: 'rule', name: 'Rule Hit', logic: 'X' },
                    { id: 302, asset_type: 'ce', name: 'CE Hit', logic: 'Y' },
                ],
                total_results: 2,
            },
        });
        renderBrowse();
        fireEvent.change(screen.getByPlaceholderText('Search public rules...'), { target: { value: 'mixed' } });
        expect(await screen.findByText('Rule Hit')).toBeInTheDocument();
        expect(screen.queryByText('CE Hit')).not.toBeInTheDocument();
    });

    it('shows a "No results" message when a search returns nothing', async () => {
        api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderBrowse();
        fireEvent.change(screen.getByPlaceholderText('Search public rules...'), { target: { value: 'zzz' } });
        expect(await screen.findByText('No results found for "zzz".')).toBeInTheDocument();
    });

    it('shows an error banner when the search request fails', async () => {
        api.searchLibrary.mockRejectedValue(new Error('search down'));
        renderBrowse();
        fireEvent.change(screen.getByPlaceholderText('Search public rules...'), { target: { value: 'boom' } });
        expect(await screen.findByText('Search failed. Please try again.')).toBeInTheDocument();
    });

    it('clears the search query via the Reset button', async () => {
        renderBrowse();
        const input = screen.getByPlaceholderText('Search public rules...');
        fireEvent.change(input, { target: { value: 'temp' } });
        expect(input).toHaveValue('temp');
        fireEvent.click(screen.getByRole('button', { name: /Reset All/i }));
        expect(input).toHaveValue('');
    });
});

describe('Browse — category filtering', () => {
    it('renders available categories in the panel and filters the list when one is selected', async () => {
        api.getAllCategories.mockResolvedValue({ data: ['Safety & Harm Prevention', 'Bias & Fairness'] });
        api.getPublicRules.mockResolvedValue({
            data: {
                rules: [
                    publicRule({ rule_id: 100, name: 'Safety Rule', categories: ['Safety & Harm Prevention'] }),
                    publicRule({ rule_id: 101, public_id: 'pub-101', name: 'Bias Rule', categories: ['Bias & Fairness'] }),
                ],
            },
        });
        renderBrowse();
        await screen.findByText('Safety Rule');
        // Select the Bias category from the panel.
        fireEvent.click(screen.getByRole('button', { name: 'Bias & Fairness' }));
        await waitFor(() => expect(screen.queryByText('Safety Rule')).not.toBeInTheDocument());
        expect(screen.getByText('Bias Rule')).toBeInTheDocument();
    });

    it('shows a "no rules match" message when the selected category matches nothing', async () => {
        api.getAllCategories.mockResolvedValue({ data: ['Privacy'] });
        api.getPublicRules.mockResolvedValue({
            data: { rules: [publicRule({ name: 'Safety Rule', categories: ['Safety & Harm Prevention'] })] },
        });
        renderBrowse();
        await screen.findByText('Safety Rule');
        fireEvent.click(screen.getByRole('button', { name: 'Privacy' }));
        expect(await screen.findByText('No rules match the selected categories.')).toBeInTheDocument();
    });
});

describe('Browse — pagination', () => {
    it('renders pagination for a long list and changes the visible page', async () => {
        // 15 rules => 2 pages at pageSize 10. Page 1 shows the first 10.
        const many = Array.from({ length: 15 }, (_, i) =>
            publicRule({ rule_id: 500 + i, public_id: `pub-${500 + i}`, setup_id: 500 + i, name: `Rule ${i}`, categories: [] }),
        );
        api.getPublicRules.mockResolvedValue({ data: { rules: many } });
        renderBrowse();
        await screen.findByText('Rule 0');
        // Rule 10 is on page 2, not visible yet.
        expect(screen.queryByText('Rule 10')).not.toBeInTheDocument();
        // Go to page 2.
        fireEvent.click(screen.getByRole('button', { name: '2' }));
        expect(await screen.findByText('Rule 10')).toBeInTheDocument();
        expect(screen.queryByText('Rule 0')).not.toBeInTheDocument();
    });
});

describe('Browse — author filter chip', () => {
    it('shows the author chip when ?author= is in the URL and clears it', async () => {
        renderBrowse('/browse?author=Alice');
        expect(await screen.findByText('@alice')).toBeInTheDocument();
        expect(screen.getByText('Filtered to author:')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Clear author filter' }));
        await waitFor(() => expect(screen.queryByText('@alice')).not.toBeInTheDocument());
    });
});

describe('Browse — library refresh event', () => {
    it('refetches rules + bookmarks when gavel:libraryChanged fires', async () => {
        renderBrowse();
        await waitFor(() => expect(api.getPublicRules).toHaveBeenCalledTimes(1));
        window.dispatchEvent(new Event('gavel:libraryChanged'));
        await waitFor(() => expect(api.getPublicRules).toHaveBeenCalledTimes(2));
        expect(api.getRuleBookmarks).toHaveBeenCalledTimes(2);
    });
});
