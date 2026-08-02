// Behavior tests for BookmarksCEs.jsx
//
// Follows the established pattern in pages.smoke.test.jsx: mock '../api'
// with benign defaults for every export the page (and its children:
// Layout/Sidebar) might call, stub Sidebar, mock sweetalert2, wrap in
// MemoryRouter, and set localStorage token + user. Here we go further and
// drive real interactions to exercise BookmarksCEs's branches.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// --- Router: spy navigate but keep everything else real. ---
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- Create-entry-point modal is stubbed (same as BrowseCEs.test.jsx): the
// real one calls useTaskTray() on render, which needs a TaskTrayProvider this
// harness doesn't mount. We only care that it renders when open.
vi.mock('../../src/pages/CEGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="ce-modal-open" /> : null),
}));

// --- API mock. Cover every export the page + Layout/Sidebar could call. ---
vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return {
        default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()), delete: vi.fn(() => empty()), put: vi.fn(() => empty()) },
        getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
        removeCEBookmark: vi.fn(() => Promise.resolve({ data: {} })),
        getCognitiveDataset: vi.fn(() => Promise.resolve({ data: { training_data_preview: [] } })),
        getCognitiveElements: vi.fn(() => Promise.resolve({ data: [] })),
        searchBookmarks: vi.fn(() => Promise.resolve({ data: { results: [], total_results: 0 } })),
        // Misc exports Sidebar/Layout may touch.
        getBackendHealth: vi.fn(() => empty({ ready: true })),
        getUserModels: vi.fn(() => empty({ models: [] })),
    };
});

// Stub the Sidebar — it has its own fetches/routing irrelevant here.
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

// Suppress Swal popups.
vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: true })) },
}));

// Spy on the alert dialog helper so we can assert success/error toasts.
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: vi.fn(() => Promise.resolve()),
    showConfirmDialog: vi.fn(() => Promise.resolve(true)),
}));

import BookmarksCEs from '../../src/pages/BookmarksCEs';
import * as api from '../../src/api';
import { showAlertDialog } from '../../src/components/ConfirmDialog/confirmDialog';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderPage = () => render(
    <MemoryRouter initialEntries={['/bookmarks/ces']}>
        <Routes>
            <Route path="/bookmarks/ces" element={<BookmarksCEs />} />
        </Routes>
    </MemoryRouter>,
);

// Helper to build a bookmark list response + matching CE catalog.
const ceCatalog = [
    {
        ce_id: 101,
        name: 'Toxicity Detector',
        definition: 'Flags toxic language',
        categories: ['Safety & Harm Prevention'],
        public_id: 'pub-101',
        is_local_draft: false,
        examples: ['you are awful'],
    },
    {
        ce_id: 102,
        name: 'PII Detector',
        definition: 'Flags personal info',
        categories: ['Privacy'],
        public_id: 'pub-102',
        is_local_draft: false,
        examples: [],
    },
];

const seedBookmarks = (ids = [101, 102], catalog = ceCatalog) => {
    api.getCEBookmarks.mockResolvedValueOnce({
        data: { bookmarks: ids.map((id) => ({ ce_id: id })) },
    });
    api.getCognitiveElements.mockResolvedValueOnce({ data: catalog });
};

describe('BookmarksCEs', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        setUser();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('renders the header and pill navigation', async () => {
        renderPage();
        expect(await screen.findByText('My Bookmarked CEs')).toBeInTheDocument();
        expect(screen.getByText('Back to Community')).toBeInTheDocument();
        expect(screen.getByText('My Rules')).toBeInTheDocument();
        expect(screen.getByText('My CEs')).toBeInTheDocument();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('shows the empty state when there are no bookmarks', async () => {
        renderPage();
        expect(await screen.findByText('No CEs Found')).toBeInTheDocument();
        expect(screen.getByText("You haven't bookmarked or drafted any CEs yet.")).toBeInTheDocument();
        // The "Browse Public CEs" CTA only appears when bookmarks.length === 0.
        expect(screen.getByText('Browse Public CEs')).toBeInTheDocument();
    });

    it('navigates to /browse/ces from the empty-state CTA', async () => {
        renderPage();
        const cta = await screen.findByText('Browse Public CEs');
        fireEvent.click(cta);
        expect(mockNavigate).toHaveBeenCalledWith('/browse/ces');
    });

    it('renders cards for fetched bookmarks', async () => {
        seedBookmarks();
        renderPage();
        expect(await screen.findByText('Toxicity Detector')).toBeInTheDocument();
        expect(screen.getByText('PII Detector')).toBeInTheDocument();
        expect(api.getCEBookmarks).toHaveBeenCalledWith(7);
        expect(api.getCognitiveElements).toHaveBeenCalledWith(7);
    });

    it('handles getCognitiveElements returning a results wrapper object', async () => {
        api.getCEBookmarks.mockResolvedValueOnce({ data: { bookmarks: [{ id: 101 }] } });
        api.getCognitiveElements.mockResolvedValueOnce({ data: { results: ceCatalog } });
        renderPage();
        expect(await screen.findByText('Toxicity Detector')).toBeInTheDocument();
    });

    it('falls back to empty state when fetchBookmarks throws', async () => {
        api.getCEBookmarks.mockRejectedValueOnce(new Error('boom'));
        renderPage();
        expect(await screen.findByText('No CEs Found')).toBeInTheDocument();
    });

    it('navigates via the pill buttons', async () => {
        renderPage();
        fireEvent.click(await screen.findByText('Back to Community'));
        expect(mockNavigate).toHaveBeenCalledWith('/community/ces');
        fireEvent.click(screen.getByText('My Rules'));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/rules');
        fireEvent.click(screen.getByText('My CEs'));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/ces');
    });

    it('expands a card and fetches its dataset preview', async () => {
        seedBookmarks();
        api.getCognitiveDataset.mockResolvedValueOnce({
            data: { training_data_preview: ['hello there'] },
        });
        renderPage();
        const header = (await screen.findByText('Toxicity Detector'));
        fireEvent.click(header);
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledWith(101));
    });

    it('caches dataset previews so re-expanding the same card does not refetch', async () => {
        seedBookmarks();
        renderPage();
        const header = await screen.findByText('Toxicity Detector');
        fireEvent.click(header); // expand -> fetch
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledTimes(1));
        fireEvent.click(header); // collapse, no fetch
        fireEvent.click(header); // re-expand, cache hit, no fetch
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledTimes(1));
    });

    it('tolerates a dataset preview fetch error', async () => {
        seedBookmarks();
        api.getCognitiveDataset.mockRejectedValueOnce(new Error('no data'));
        renderPage();
        const header = await screen.findByText('Toxicity Detector');
        fireEvent.click(header);
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalled());
        // Still rendered, no crash.
        expect(screen.getByText('Toxicity Detector')).toBeInTheDocument();
    });

    it('removes a bookmark and shows a success toast', async () => {
        seedBookmarks();
        const { container } = renderPage();
        await screen.findByText('Toxicity Detector');
        // Find the card whose heading is "Toxicity Detector" and click its
        // own bookmark/remove button (ordering is LIFO so we can't index 0).
        const toxCard = Array.from(container.querySelectorAll('.ce-card')).find(
            (c) => within(c).queryByText('Toxicity Detector'),
        );
        fireEvent.click(within(toxCard).getByLabelText('Bookmark CE'));
        await waitFor(() => expect(api.removeCEBookmark).toHaveBeenCalledWith(7, 101));
        await waitFor(() =>
            expect(showAlertDialog).toHaveBeenCalledWith(
                expect.objectContaining({ variant: 'success' }),
            ),
        );
        // The removed card disappears from the list.
        await waitFor(() =>
            expect(screen.queryByText('Toxicity Detector')).not.toBeInTheDocument(),
        );
        expect(screen.getByText('PII Detector')).toBeInTheDocument();
    });

    it('shows an error toast when removing a bookmark fails', async () => {
        seedBookmarks();
        api.removeCEBookmark.mockRejectedValueOnce(new Error('fail'));
        renderPage();
        await screen.findByText('Toxicity Detector');
        const removeButtons = screen.getAllByLabelText('Bookmark CE');
        fireEvent.click(removeButtons[0]);
        await waitFor(() =>
            expect(showAlertDialog).toHaveBeenCalledWith(
                expect.objectContaining({ variant: 'error' }),
            ),
        );
    });

    it('runs a server search when a query is typed', async () => {
        seedBookmarks();
        api.searchBookmarks.mockResolvedValue({
            data: {
                results: [
                    { id: 201, name: 'Searched CE', content: 'def', categories: ['Privacy'], public_id: 'p201', is_local_draft: false },
                ],
                total_results: 1,
            },
        });
        renderPage();
        await screen.findByText('Toxicity Detector');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'tox' } });

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        await waitFor(() => expect(screen.getByText(/Search Results/)).toBeInTheDocument());
        expect(await screen.findByText('Searched CE')).toBeInTheDocument();
        // searchBookmarks called with the typed query + ce asset type.
        expect(api.searchBookmarks).toHaveBeenCalledWith(
            expect.objectContaining({ user_id: 7, q: 'tox', asset_types: 'ce' }),
        );
    });

    it('falls back to local filtering when server search throws', async () => {
        seedBookmarks();
        api.searchBookmarks.mockRejectedValue(new Error('server down'));
        renderPage();
        await screen.findByText('Toxicity Detector');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'toxicity' } });

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        // Local applyFilters keeps the matching card, drops the non-matching one.
        await waitFor(() => expect(screen.getByText('Toxicity Detector')).toBeInTheDocument());
        await waitFor(() => expect(screen.queryByText('PII Detector')).not.toBeInTheDocument());
    });

    it('shows "No CEs match your search. when a server search returns nothing', async () => {
        seedBookmarks();
        api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderPage();
        await screen.findByText('Toxicity Detector');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'zzz-nomatch' } });

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
        expect(await screen.findByText('No CEs Found')).toBeInTheDocument();
        expect(screen.getByText('No CEs match your search.')).toBeInTheDocument();
        // No "Browse Public CEs" CTA since bookmarks.length > 0.
        expect(screen.queryByText('Browse Public CEs')).not.toBeInTheDocument();
    });

    it('resets search state and restores all bookmarks via Reset All', async () => {
        seedBookmarks();
        api.searchBookmarks.mockResolvedValue({ data: { results: [], total_results: 0 } });
        renderPage();
        await screen.findByText('Toxicity Detector');

        const input = screen.getByPlaceholderText('Search in your bookmarks...');
        fireEvent.change(input, { target: { value: 'zzz' } });
        await screen.findByText('No CEs Found');

        fireEvent.click(screen.getByText('Reset All'));
        // Both bookmarks shown again, search results banner gone.
        expect(await screen.findByText('Toxicity Detector')).toBeInTheDocument();
        expect(screen.getByText('PII Detector')).toBeInTheDocument();
        expect(screen.queryByText(/Search Results/)).not.toBeInTheDocument();
    });

    it('filters by selecting an available category', async () => {
        seedBookmarks();
        api.searchBookmarks.mockResolvedValue({
            data: {
                results: [
                    { id: 102, name: 'PII Detector', content: 'Flags personal info', categories: ['Privacy'], public_id: 'pub-102', is_local_draft: false },
                ],
                total_results: 1,
            },
        });
        renderPage();
        await screen.findByText('Toxicity Detector');

        // Available categories are derived from the bookmark catalog.
        const privacyBtn = screen.getByRole('button', { name: 'Privacy' });
        fireEvent.click(privacyBtn);

        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalledWith(
            expect.objectContaining({ categories: 'Privacy', asset_types: 'ce' }),
        ));
    });

    it('triggers a server search via the SearchPanel Search button', async () => {
        seedBookmarks();
        renderPage();
        await screen.findByText('Toxicity Detector');
        api.searchBookmarks.mockClear();

        const searchBtn = screen.getByRole('button', { name: /Search/ });
        fireEvent.click(searchBtn);
        await waitFor(() => expect(api.searchBookmarks).toHaveBeenCalled());
    });

    it('refetches bookmarks on a library-changed event', async () => {
        seedBookmarks();
        // Subsequent refetch after the event.
        api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [{ ce_id: 101 }] } });
        api.getCognitiveElements.mockResolvedValue({ data: ceCatalog });
        renderPage();
        await screen.findByText('Toxicity Detector');
        const callsBefore = api.getCEBookmarks.mock.calls.length;

        fireEvent(window, new Event('gavel:libraryChanged'));
        await waitFor(() =>
            expect(api.getCEBookmarks.mock.calls.length).toBeGreaterThan(callsBefore),
        );
    });

    it('paginates when there are more bookmarks than the page size', async () => {
        // 12 bookmarks > default topK (10) => Pagination renders.
        const many = Array.from({ length: 12 }, (_, i) => ({
            ce_id: 300 + i,
            name: `CE ${i}`,
            definition: 'd',
            categories: ['Privacy'],
            public_id: `p${i}`,
            is_local_draft: false,
            examples: [],
        }));
        seedBookmarks(many.map((c) => c.ce_id), many);
        const { container } = renderPage();
        // Wait for cards to render (some CE name appears).
        await screen.findByText('CE 5');

        // Page 1 shows exactly topK (10) cards; 12 total => 2 remain for page 2.
        // waitFor so we don't race the full list render under heavy suite load.
        await waitFor(() => {
            expect(container.querySelectorAll('.rules-list .ce-card').length).toBe(10);
        });

        // Pagination control is present (12 > pageSize 10).
        const page2 = screen.getByRole('button', { name: '2' });
        fireEvent.click(page2);

        // Page 2 shows the remaining 2 cards.
        await waitFor(() => {
            const cardsPage2 = container.querySelectorAll('.rules-list .ce-card');
            expect(cardsPage2.length).toBe(2);
        });
    });
});
