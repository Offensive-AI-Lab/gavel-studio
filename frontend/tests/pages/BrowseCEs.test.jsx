// Behavior tests for the Browse · Cognitive Elements page (BrowseCEs.jsx).
//
// BrowseCEs lists public CEs (from getCognitiveElements), runs a live,
// debounced library search (via useLibrarySearch -> searchLibrary), lets the
// user bookmark/unbookmark public CEs, paginates both the default list and the
// search results, and routes to /ce-wizard / /login / sidebar links.
//
// We mock the network (../api) with benign defaults for every export the page
// and its hooks touch, the publish service, the confirm-dialog helpers (so no
// real Swal pops), sweetalert2, and spy on useNavigate while keeping the rest
// of react-router (MemoryRouter, useSearchParams) real. The Sidebar (rendered
// by Layout) has its own fetches/routing, so we stub it.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// ---- navigate spy; keep MemoryRouter + useSearchParams real ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// ---- API mock: every export the page (+ useLibrarySearch) calls ----
vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return {
        default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()) },
        getCognitiveElements: vi.fn(() => empty([])),
        getCognitiveDataset: vi.fn(() => empty({ training_data_preview: [] })),
        addCEBookmark: vi.fn(() => empty()),
        removeCEBookmark: vi.fn(() => empty()),
        getCEBookmarks: vi.fn(() => empty({ bookmarks: [] })),
        getAllCategories: vi.fn(() => empty([])),
        searchLibrary: vi.fn(() => empty({ results: [], total_results: 0 })),
    };
});


// ---- confirm-dialog helpers — controllable / capturable ----
const mockShowAlertDialog = vi.fn(() => Promise.resolve());
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: (...a) => mockShowAlertDialog(...a),
    showConfirmDialog: vi.fn(() => Promise.resolve(true)),
}));

// The AI-CE modal pulls in the task-tray context + wizard machinery; it has
// its own coverage. Here it's a lightweight stub reflecting its `open` prop so
// we can assert the button opens it (instead of the old navigation).
vi.mock('../../src/pages/CEGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="ce-modal-open" /> : null),
}));

// ---- Sidebar stub (Layout renders it) ----
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

// ---- sweetalert2 stub ----
vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })), close: vi.fn() },
}));

import BrowseCEs from '../../src/pages/BrowseCEs';
import * as api from '../../src/api';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderPage = (initialEntries = ['/browse/ces']) =>
    render(
        <TutorialProvider>
            <MemoryRouter initialEntries={initialEntries}>
                <Routes>
                    <Route path="/browse/ces" element={<BrowseCEs />} />
                </Routes>
            </MemoryRouter>
        </TutorialProvider>,
    );

// A public CE row as returned by getCognitiveElements (the default list).
const publicCe = (over = {}) => ({
    ce_id: 1,
    name: 'Public CE One',
    definition: 'detects something public',
    categories: ['Safety'],
    is_local_draft: false,
    public_id: 'pub-1',
    examples: [],
    ...over,
});

// A raw search row as returned by searchLibrary (mapped via mapSearchCe).
const searchRow = (over = {}) => ({
    id: 100,
    name: 'Searched CE',
    content: 'found by search',
    categories: ['Privacy'],
    is_local_draft: false,
    examples: [],
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    mockShowAlertDialog.mockResolvedValue(undefined);
    api.getCognitiveElements.mockResolvedValue({ data: [] });
    api.getCognitiveDataset.mockResolvedValue({ data: { training_data_preview: [] } });
    api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.getAllCategories.mockResolvedValue({ data: [] });
    api.addCEBookmark.mockResolvedValue({ data: {} });
    api.removeCEBookmark.mockResolvedValue({ data: {} });
    api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
});

afterEach(() => {
    localStorage.clear();
});

describe('BrowseCEs — mount & header', () => {
    it('fetches CEs, categories and bookmarks for the stored user on mount', async () => {
        renderPage();
        await waitFor(() => expect(api.getCognitiveElements).toHaveBeenCalledWith(7));
        expect(api.getAllCategories).toHaveBeenCalled();
        expect(api.getCEBookmarks).toHaveBeenCalledWith(7);
    });

    it('renders the header, intro copy and the tab buttons', async () => {
        renderPage();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Cognitive Elements' })).toBeInTheDocument();
        expect(screen.getByText(/inspect sample excitation data/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Rules' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'CEs' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Contributors' })).toBeInTheDocument();
        await waitFor(() => expect(api.getCognitiveElements).toHaveBeenCalled());
    });
});

describe('BrowseCEs — loading & empty states', () => {
    it('shows the loading indicator before the CE list resolves', async () => {
        let resolveFetch;
        api.getCognitiveElements.mockReturnValue(new Promise((r) => { resolveFetch = r; }));
        renderPage();
        expect(screen.getByText('Loading...')).toBeInTheDocument();
        resolveFetch({ data: [] });
        await waitFor(() => expect(screen.queryByText('Loading...')).not.toBeInTheDocument());
    });

    it('renders the empty state when there are no public CEs', async () => {
        renderPage();
        expect(await screen.findByText('No Cognitive Elements')).toBeInTheDocument();
        expect(screen.getByText(/Public CEs will appear here/i)).toBeInTheDocument();
    });

    it('falls back to the empty state when the CE fetch rejects', async () => {
        api.getCognitiveElements.mockRejectedValue(new Error('boom'));
        renderPage();
        expect(await screen.findByText('No Cognitive Elements')).toBeInTheDocument();
    });

    it('tolerates a non-array data payload', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: { not: 'an array' } });
        renderPage();
        expect(await screen.findByText('No Cognitive Elements')).toBeInTheDocument();
    });
});

describe('BrowseCEs — public CE list rendering', () => {
    it('renders the "All Public CEs" section with a card per CE', async () => {
        api.getCognitiveElements.mockResolvedValue({
            data: [publicCe(), publicCe({ ce_id: 2, name: 'Public CE Two' })],
        });
        renderPage();
        expect(await screen.findByText('All Public CEs')).toBeInTheDocument();
        expect(screen.getByText('Public CE One')).toBeInTheDocument();
        expect(screen.getByText('Public CE Two')).toBeInTheDocument();
    });

    it('shows the Public badge for non-draft CEs and a bookmark button', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        renderPage();
        await screen.findByText('Public CE One');
        expect(screen.getByText('Public')).toBeInTheDocument();
        // Bookmark affordance only for public CEs with a public_id.
        expect(screen.getByRole('button', { name: /Bookmark CE/i })).toBeInTheDocument();
    });

    it('does NOT show local-draft CEs — Browse is the public space', async () => {
        api.getCognitiveElements.mockResolvedValue({
            data: [
                publicCe({ ce_id: 2, name: 'Published CE' }),
                publicCe({ ce_id: 3, name: 'Draft CE', is_local_draft: true, public_id: undefined }),
            ],
        });
        renderPage();
        await screen.findByText('Published CE');
        expect(screen.queryByText('Draft CE')).not.toBeInTheDocument();
    });
});

// The "Create New CE (AI)" entry moved out of this public page into the sidebar
// Create menu, so its in-page tests were removed.

describe('BrowseCEs — bookmark interactions', () => {
    it('adds a bookmark and shows a saved alert', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        renderPage();
        await screen.findByText('Public CE One');
        fireEvent.click(screen.getByRole('button', { name: /Bookmark CE/i }));
        await waitFor(() => expect(api.addCEBookmark).toHaveBeenCalledWith(7, 1));
        expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Saved', variant: 'success' }),
        );
        // The new bookmark shows up in the sidebar list.
        const aside = screen.getByText('My Bookmarked CEs').closest('aside');
        expect(within(aside).getByText('Public CE One')).toBeInTheDocument();
    });

    it('removes an already-bookmarked CE and shows a removed alert', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [{ ce_id: 1, name: 'Public CE One' }] } });
        renderPage();
        await screen.findByText('All Public CEs');
        // The card's bookmark button reads "Remove" when already bookmarked.
        const card = screen.getByText('Public CE One', { selector: 'h3' }).closest('.ce-card');
        fireEvent.click(within(card).getByRole('button', { name: /Bookmark CE/i }));
        await waitFor(() => expect(api.removeCEBookmark).toHaveBeenCalledWith(7, 1));
        expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Removed', variant: 'success' }),
        );
    });

    it('shows an error alert when the bookmark call fails', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        api.addCEBookmark.mockRejectedValue(new Error('nope'));
        renderPage();
        await screen.findByText('Public CE One');
        fireEvent.click(screen.getByRole('button', { name: /Bookmark CE/i }));
        await waitFor(() => expect(mockShowAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Error', variant: 'error' }),
        ));
    });

    it('removes a bookmark from the sidebar via its Remove button', async () => {
        api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [{ ce_id: 5, name: 'Saved CE' }] } });
        renderPage();
        const aside = (await screen.findByText('My Bookmarked CEs')).closest('aside');
        expect(within(aside).getByText('Saved CE')).toBeInTheDocument();
        fireEvent.click(within(aside).getByRole('button', { name: /Remove bookmark/i }));
        await waitFor(() => expect(api.removeCEBookmark).toHaveBeenCalledWith(7, 5));
    });

    it('shows the empty bookmarks message when there are none', async () => {
        renderPage();
        const aside = (await screen.findByText('My Bookmarked CEs')).closest('aside');
        expect(within(aside).getByText('No bookmarks yet.')).toBeInTheDocument();
    });
});

// Publishing a draft CE now lives in Your Library (drafts no longer appear on
// this public page), so the in-Browse publish test was removed.

describe('BrowseCEs — expand & lazy-load samples', () => {
    it('fetches the dataset on first expand and caches it', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        api.getCognitiveDataset.mockResolvedValue({
            data: { training_data_preview: [{ input: 'hi', output: 'YES' }] },
        });
        renderPage();
        const title = await screen.findByText('Public CE One');
        fireEvent.click(title);
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledWith(1));
        // Collapse + re-expand: cache means no second fetch.
        fireEvent.click(title);
        fireEvent.click(title);
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledTimes(1));
    });

    it('swallows a dataset fetch error and still expands', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        api.getCognitiveDataset.mockRejectedValue(new Error('fail'));
        renderPage();
        const title = await screen.findByText('Public CE One');
        fireEvent.click(title);
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalled());
        expect(await screen.findByText('Examples')).toBeInTheDocument();
    });
});

describe('BrowseCEs — live search', () => {
    it('runs a search as the user types and renders mapped results', async () => {
        // v2 CE fields (role/title/tags) must survive the search-row mapping —
        // the card renders role as its primary badge (underscores prettified).
        api.searchLibrary.mockResolvedValue({
            data: { results: [searchRow({ role: 'directive_to_user' })], total_results: 1 },
        });
        renderPage();
        await screen.findByText('No Cognitive Elements');
        const input = screen.getByPlaceholderText('Search cognitive elements...');
        fireEvent.change(input, { target: { value: 'privacy' } });
        // Debounced; wait for the call + the mapped card.
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        expect(api.searchLibrary.mock.calls[0][0]).toEqual(
            expect.objectContaining({ q: 'privacy', asset_types: 'ce' }),
        );
        expect(await screen.findByText('Searched CE')).toBeInTheDocument();
        expect(await screen.findByText(/Search Results \(1 found\)/)).toBeInTheDocument();
        expect(screen.getByText('directive to user')).toBeInTheDocument();
    });

    it('keeps search results bookmarkable and credited to their author', async () => {
        // Regression: the search-row mapping dropped public_id (which gates the
        // card's Save button) and created_by_username (the author link), so
        // searching silently removed both from every result.
        api.searchLibrary.mockResolvedValue({
            data: {
                results: [searchRow({ public_id: 'pub-100', created_by_username: 'ofek' })],
                total_results: 1,
            },
        });
        renderPage();
        await screen.findByText('No Cognitive Elements');
        const input = screen.getByPlaceholderText('Search cognitive elements...');
        fireEvent.change(input, { target: { value: 'privacy' } });
        const card = (await screen.findByText('Searched CE', { selector: 'h3' })).closest('.ce-card');
        expect(within(card).getByText('by @ofek')).toBeInTheDocument();
        fireEvent.click(within(card).getByRole('button', { name: /Bookmark CE/i }));
        await waitFor(() => expect(api.addCEBookmark).toHaveBeenCalledWith(7, 100));
    });

    it('shows a no-results message when the search returns nothing', async () => {
        api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderPage();
        await screen.findByText('No Cognitive Elements');
        const input = screen.getByPlaceholderText('Search cognitive elements...');
        fireEvent.change(input, { target: { value: 'zzzznotfound' } });
        expect(await screen.findByText(/No cognitive elements found for "zzzznotfound"/)).toBeInTheDocument();
    });

    it('surfaces a search error banner when searchLibrary rejects', async () => {
        api.searchLibrary.mockRejectedValue(new Error('search down'));
        renderPage();
        await screen.findByText('No Cognitive Elements');
        const input = screen.getByPlaceholderText('Search cognitive elements...');
        fireEvent.change(input, { target: { value: 'hello' } });
        expect(await screen.findByText(/Search failed\. Please try again\./i)).toBeInTheDocument();
    });
});

describe('BrowseCEs — pagination of the public list', () => {
    it('renders pagination and pages through a long CE list', async () => {
        // 25 CEs → 3 pages at pageSize 10. Page 1 shows CE #1, not CE #11.
        const many = Array.from({ length: 25 }, (_, i) =>
            publicCe({ ce_id: i + 1, name: `CE Number ${i + 1}`, public_id: `pub-${i + 1}` }),
        );
        api.getCognitiveElements.mockResolvedValue({ data: many });
        renderPage();
        expect(await screen.findByText('CE Number 1')).toBeInTheDocument();
        expect(screen.queryByText('CE Number 11')).not.toBeInTheDocument();

        // Jump to page 2 (button label "2").
        fireEvent.click(screen.getByRole('button', { name: '2' }));
        expect(await screen.findByText('CE Number 11')).toBeInTheDocument();
        expect(screen.queryByText('CE Number 1', { selector: 'h3' })).not.toBeInTheDocument();
    });

    it('does not render pagination for a short list', async () => {
        api.getCognitiveElements.mockResolvedValue({ data: [publicCe()] });
        renderPage();
        await screen.findByText('Public CE One');
        // Pagination renders next/prev as < and >.
        expect(screen.queryByRole('button', { name: '<' })).not.toBeInTheDocument();
    });
});

describe('BrowseCEs — author filter chip', () => {
    it('shows the author chip from ?author= and clears it on ×', async () => {
        renderPage(['/browse/ces?author=alice']);
        await waitFor(() => expect(api.getCognitiveElements).toHaveBeenCalled());
        expect(screen.getByText('@alice')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: /Clear author filter/i }));
        await waitFor(() => expect(screen.queryByText('@alice')).not.toBeInTheDocument());
    });
});

describe('BrowseCEs — tab & sidebar navigation', () => {
    it('navigates to Browse Rules and the Hub from the header', async () => {
        renderPage();
        await screen.findByText('No Cognitive Elements');
        fireEvent.click(screen.getByRole('button', { name: 'Rules' }));
        expect(mockNavigate).toHaveBeenCalledWith('/community');
        fireEvent.click(screen.getByText('Hub'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
    });

    it('navigates to the bookmarks page via "View all"', async () => {
        renderPage();
        await screen.findByText('My Bookmarked CEs');
        fireEvent.click(screen.getByText('View all'));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/ces');
    });
});

describe('BrowseCEs — library refresh event', () => {
    it('refetches CEs when gavel:libraryChanged fires', async () => {
        renderPage();
        await waitFor(() => expect(api.getCognitiveElements).toHaveBeenCalledTimes(1));
        window.dispatchEvent(new Event('gavel:libraryChanged'));
        await waitFor(() => expect(api.getCognitiveElements).toHaveBeenCalledTimes(2));
    });
});
