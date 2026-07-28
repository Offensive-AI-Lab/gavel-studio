// Behavior tests for Community.jsx (the /community discovery page).
//
// Follows the smoke-test pattern: mock '../api' with benign defaults for
// every export this page + its children might touch, stub the Sidebar
// (rendered inside Layout), and wrap renders in the routing + tutorial
// providers the page relies on.

import React from 'react';
import {
    describe, it, expect, vi, beforeEach, afterEach,
} from 'vitest';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';
import {
    render, screen, fireEvent, waitFor, act,
} from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// --- API mock. searchArtists / getLeaderboard are the two this page uses;
// the rest are benign so the Sidebar/Layout tree never hits the network.
vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return {
        default: {
            get: vi.fn(() => empty()), post: vi.fn(() => empty()),
            delete: vi.fn(() => empty()), put: vi.fn(() => empty()),
        },
        searchArtists: vi.fn(() => empty({ items: [], total: 0 })),
        getLeaderboard: vi.fn(() => empty({ items: [], total: 0 })),
    };
});

// Stub the Sidebar — it has its own data fetches we don't want to run.
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

import Community from '../../src/pages/Community';
import { searchArtists, getLeaderboard } from '../../src/api';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 1, email: 'a@b.c' }));
};

// Render at a URL; the captured location lets tests read the current
// path + query string so URL-driven state changes are observable.
let lastLocation;
const LocationProbe = () => {
    lastLocation = useLocation();
    return null;
};

const renderAt = (path = '/community') => render(
    <TutorialProvider>
        <MemoryRouter initialEntries={[path]}>
            <LocationProbe />
            <Routes>
                <Route path="/community" element={<Community />} />
                <Route path="/workspace" element={<div data-testid="workspace-page" />} />
            </Routes>
        </MemoryRouter>
    </TutorialProvider>,
);

const makeArtist = (overrides = {}) => ({
    username: 'alice',
    display_name: 'Alice A',
    bio: 'I write rules',
    is_team: false,
    contribution_count_rules: 2,
    contribution_count_ces: 3,
    avg_rating_received: 4.25,
    total_rating_count: 8,
    ...overrides,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    searchArtists.mockResolvedValue({ data: { items: [], total: 0 } });
    getLeaderboard.mockResolvedValue({ data: { items: [], total: 0 } });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('Community — initial render & layout', () => {
    it('renders the header, both mode tabs, and mounts inside Layout', async () => {
        renderAt();
        expect(screen.getByRole('heading', { name: 'Community' })).toBeInTheDocument();
        // The header subtitle and the inline page-help summary both contain this
        // sentence, so assert at least one (the header copy) is present.
        expect(screen.getAllByText(/Discover the people behind the public library/i).length).toBeGreaterThan(0);
        expect(screen.getByRole('button', { name: /Search/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Leaderboard/i })).toBeInTheDocument();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/Search by username or display name/i)).toBeInTheDocument();
        await waitFor(() => expect(searchArtists).toHaveBeenCalled());
        expect(getLeaderboard).not.toHaveBeenCalled();
    });

    it('calls searchArtists with the empty url query and page 1 on mount', async () => {
        renderAt();
        await waitFor(() => expect(searchArtists).toHaveBeenCalledWith('', 1, 12));
    });

    it('shows the empty state for search mode with no query', async () => {
        renderAt();
        expect(await screen.findByText(/No artists yet\. Be the first to publish!/i)).toBeInTheDocument();
    });
});

describe('Community — search mode with results', () => {
    it('renders an artist card with display name, handle and contributions', async () => {
        // Ratings were removed with accounts, so the card shows contribution
        // volume only — no average, no rating count.
        searchArtists.mockResolvedValue({ data: { items: [makeArtist()], total: 1 } });
        renderAt();
        expect(await screen.findByText('Alice A')).toBeInTheDocument();
        expect(screen.getByText('@alice')).toBeInTheDocument();
        expect(screen.getByText('5')).toBeInTheDocument();
        expect(screen.getByText('contributions')).toBeInTheDocument();
        expect(screen.queryByText('4.3')).not.toBeInTheDocument();
        expect(screen.queryByText(/No ratings yet/i)).not.toBeInTheDocument();
    });

    it('links each card to the artist profile', async () => {
        searchArtists.mockResolvedValue({ data: { items: [makeArtist()], total: 1 } });
        renderAt();
        const link = await screen.findByRole('link');
        expect(link).toHaveAttribute('href', '/profile/alice');
    });

    it('falls back to username when display_name is missing', async () => {
        searchArtists.mockResolvedValue({
            data: { items: [makeArtist({ display_name: null })], total: 1 },
        });
        renderAt();
        await waitFor(() => expect(screen.getByText('@alice')).toBeInTheDocument());
        expect(screen.getAllByText('A').length).toBeGreaterThanOrEqual(1);
    });

    it('renders the bio when present', async () => {
        searchArtists.mockResolvedValue({
            data: { items: [makeArtist({ bio: 'My custom bio text' })], total: 1 },
        });
        renderAt();
        expect(await screen.findByText('My custom bio text')).toBeInTheDocument();
    });

    it('omits the bio paragraph when bio is empty', async () => {
        searchArtists.mockResolvedValue({
            data: { items: [makeArtist({ bio: '' })], total: 1 },
        });
        renderAt();
        await screen.findByText('Alice A');
        expect(screen.queryByText('My custom bio text')).not.toBeInTheDocument();
    });

    it('renders the team award icon for team accounts', async () => {
        searchArtists.mockResolvedValue({
            data: { items: [makeArtist({ is_team: true })], total: 1 },
        });
        const { container } = renderAt();
        await screen.findByText('Alice A');
        // react-icons renders the `title` prop as a <title> SVG child node.
        const titles = Array.from(container.querySelectorAll('title'))
            .map((t) => t.textContent);
        expect(titles).toContain('Team account');
    });

    it('does not render the team badge for non-team accounts', async () => {
        searchArtists.mockResolvedValue({
            data: { items: [makeArtist({ is_team: false })], total: 1 },
        });
        const { container } = renderAt();
        await screen.findByText('Alice A');
        const titles = Array.from(container.querySelectorAll('title'))
            .map((t) => t.textContent);
        expect(titles).not.toContain('Team account');
    });

    it('renders multiple artist cards keyed by username', async () => {
        searchArtists.mockResolvedValue({
            data: {
                items: [
                    makeArtist({ username: 'alice', display_name: 'Alice' }),
                    makeArtist({ username: 'bob', display_name: 'Bob' }),
                ],
                total: 2,
            },
        });
        renderAt();
        expect(await screen.findByText('Alice')).toBeInTheDocument();
        expect(screen.getByText('Bob')).toBeInTheDocument();
        expect(screen.getAllByRole('link')).toHaveLength(2);
    });
});

describe('Community — query-driven empty state', () => {
    it('shows "No artists match" when a query yields nothing', async () => {
        searchArtists.mockResolvedValue({ data: { items: [], total: 0 } });
        renderAt('/community?q=zzz');
        expect(await screen.findByText(/No artists match "zzz"\./i)).toBeInTheDocument();
        expect(searchArtists).toHaveBeenCalledWith('zzz', 1, 12);
    });

    it('seeds the input from the url query', async () => {
        renderAt('/community?q=hello');
        expect(screen.getByPlaceholderText(/Search by username/i)).toHaveValue('hello');
    });
});

describe('Community — typing & debounced URL sync', () => {
    it('debounces input into the q query param after 350ms', () => {
        vi.useFakeTimers();
        renderAt();
        const input = screen.getByPlaceholderText(/Search by username/i);
        act(() => {
            fireEvent.change(input, { target: { value: 'ali' } });
        });
        // Before the debounce window elapses the URL has no q param.
        expect(lastLocation.search).not.toContain('q=ali');
        // Advancing past the debounce fires setSearchParams synchronously.
        act(() => { vi.advanceTimersByTime(350); });
        expect(lastLocation.search).toContain('q=ali');
    });

    it('does not sync to the URL before the debounce elapses', () => {
        vi.useFakeTimers();
        renderAt();
        const input = screen.getByPlaceholderText(/Search by username/i);
        act(() => {
            fireEvent.change(input, { target: { value: 'bob' } });
        });
        act(() => { vi.advanceTimersByTime(200); });
        expect(lastLocation.search).not.toContain('q=bob');
    });

    it('removes the q param when the input is cleared', () => {
        vi.useFakeTimers();
        renderAt('/community?q=ali');
        const input = screen.getByPlaceholderText(/Search by username/i);
        expect(input).toHaveValue('ali');
        act(() => {
            fireEvent.change(input, { target: { value: '' } });
        });
        act(() => { vi.advanceTimersByTime(350); });
        expect(lastLocation.search).not.toContain('q=');
    });
});

describe('Community — leaderboard mode', () => {
    it('switches to leaderboard mode and calls getLeaderboard', async () => {
        renderAt();
        fireEvent.click(screen.getByRole('button', { name: /Leaderboard/i }));
        await waitFor(() => expect(getLeaderboard).toHaveBeenCalledWith('avg_rating', 1, 12, 0));
        expect(screen.queryByPlaceholderText(/Search by username/i)).not.toBeInTheDocument();
        // The "Highest rated" / "Most contributions" / "Min ratings" controls
        // went away with ratings — the leaderboard has one ranking now.
        expect(screen.queryByRole('button', { name: /Highest rated/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Min ratings/i })).not.toBeInTheDocument();
        expect(lastLocation.search).toContain('mode=leaderboard');
    });

    it('renders the leaderboard empty state', async () => {
        getLeaderboard.mockResolvedValue({ data: { items: [], total: 0 } });
        renderAt('/community?mode=leaderboard');
        expect(await screen.findByText(/No leaderboard data yet/i)).toBeInTheDocument();
    });



    it('renders leaderboard artist cards', async () => {
        getLeaderboard.mockResolvedValue({
            data: { items: [makeArtist({ username: 'champ', display_name: 'Champ' })], total: 1 },
        });
        renderAt('/community?mode=leaderboard');
        expect(await screen.findByText('Champ')).toBeInTheDocument();
    });


    it('switching back to search from leaderboard restores the input', async () => {
        renderAt('/community?mode=leaderboard');
        // Anchor on the leaderboard's caption — the "Highest rated" pill this
        // used to wait for went away with the rating sort.
        await screen.findByText(/Contributors in your synced library/i);
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));
        await waitFor(() =>
            expect(screen.getByPlaceholderText(/Search by username/i)).toBeInTheDocument());
        expect(lastLocation.search).toContain('mode=search');
    });
});

describe('Community — loading & error states', () => {
    it('shows the loading indicator while the fetch is pending', async () => {
        let resolve;
        searchArtists.mockReturnValue(new Promise((r) => { resolve = r; }));
        renderAt();
        expect(screen.getByText(/Loading…/i)).toBeInTheDocument();
        await act(async () => {
            resolve({ data: { items: [], total: 0 } });
        });
        await waitFor(() => expect(screen.queryByText(/Loading…/i)).not.toBeInTheDocument());
    });

    it('falls back to an empty result set when the fetch rejects', async () => {
        searchArtists.mockRejectedValue(new Error('boom'));
        renderAt();
        expect(await screen.findByText(/No artists yet/i)).toBeInTheDocument();
    });
});

describe('Community — pagination', () => {
    it('renders pagination when total exceeds the page size', async () => {
        const items = Array.from({ length: 12 }, (_, i) =>
            makeArtist({ username: `u${i}`, display_name: `User ${i}` }));
        searchArtists.mockResolvedValue({ data: { items, total: 30 } });
        renderAt();
        await screen.findByText('User 0');
        expect(screen.getByRole('button', { name: '2' })).toBeInTheDocument();
    });

    it('does not render pagination when total fits in one page', async () => {
        searchArtists.mockResolvedValue({ data: { items: [makeArtist()], total: 1 } });
        renderAt();
        await screen.findByText('Alice A');
        expect(screen.queryByRole('button', { name: '2' })).not.toBeInTheDocument();
    });

    it('fetches the next page when a page button is clicked', async () => {
        const items = Array.from({ length: 12 }, (_, i) =>
            makeArtist({ username: `u${i}`, display_name: `User ${i}` }));
        searchArtists.mockResolvedValue({ data: { items, total: 30 } });
        renderAt();
        await screen.findByText('User 0');
        fireEvent.click(screen.getByRole('button', { name: '2' }));
        await waitFor(() => expect(searchArtists).toHaveBeenCalledWith('', 2, 12));
    });
});

describe('Community — navigation', () => {
    it('navigates back to the workspace', async () => {
        renderAt();
        fireEvent.click(screen.getByText('Hub'));
        await waitFor(() => expect(screen.getByTestId('workspace-page')).toBeInTheDocument());
        expect(lastLocation.pathname).toBe('/workspace');
    });
});
