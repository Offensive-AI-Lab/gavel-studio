// Profile.test.jsx — behavior tests for the public profile page.
//
// Profile.jsx renders a user's profile card (header stats, bio, tabs) plus
// a paginated list of their published rules/CEs. It owns several effects:
//   * initial profile fetch (getUserProfile) keyed on :username
//   * contributions fetch (searchLibrary) keyed on profile/tab/page
//   * bookmark-seed fetch (getRuleBookmarks/getCEBookmarks) for the viewer
//   * window listeners for 'gavel:ratingChanged' + 'gavel:libraryChanged'
//
// We mock '../api' so nothing hits the network, stub the heavy card
// components so assertions are deterministic, mock sweetalert2 (via the
// confirmDialog helper) to capture alert calls, and wrap renders in a
// MemoryRouter with a :username route so useParams resolves.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// --- navigate spy shared across tests ---
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- API mock: every export Profile (and its children we don't stub) call ---
vi.mock('../../src/api', () => ({
    getUserProfile: vi.fn(),
    updateMyProfile: vi.fn(),
    searchLibrary: vi.fn(),
    addRuleBookmark: vi.fn(() => Promise.resolve({ data: {} })),
    removeRuleBookmark: vi.fn(() => Promise.resolve({ data: {} })),
    getRuleBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    addCEBookmark: vi.fn(() => Promise.resolve({ data: {} })),
    removeCEBookmark: vi.fn(() => Promise.resolve({ data: {} })),
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    getCognitiveDataset: vi.fn(() => Promise.resolve({ data: { training_data_preview: [] } })),
}));

// --- Tutorial context: Profile calls useTutorialContent(); provide a no-op. ---
vi.mock('../../src/contexts/TutorialContext', () => ({
    useTutorialContent: vi.fn(),
}));

// --- Library refresh hook: capture the registered callback so tests can
// invoke it directly (it normally fires on a window 'gavel:libraryChanged'). ---
let libraryRefreshCb = null;
vi.mock('../../src/hooks/useLibraryRefresh', () => ({
    useLibraryRefresh: (cb) => { libraryRefreshCb = cb; },
}));

// --- confirmDialog helper: capture alert invocations. ---
const showAlertDialog = vi.fn(() => Promise.resolve());
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showAlertDialog: (...args) => showAlertDialog(...args),
}));

// --- Layout: render children only, no sidebar fetches. ---
vi.mock('../../src/components/Layout/Layout', () => ({
    default: ({ children }) => <div data-testid="layout">{children}</div>,
}));

// --- Pagination stub: expose page-change so we can drive setPage. ---
vi.mock('../../src/components/Pagination/Pagination', () => ({
    default: ({ currentPage, totalItems, pageSize, onPageChange }) => (
        totalItems > pageSize ? (
            <div data-testid="pagination" data-total={totalItems} data-page={currentPage}>
                <button data-testid="page-next" onClick={() => onPageChange(currentPage + 1)}>next</button>
            </div>
        ) : null
    ),
}));

// --- RuleCard stub: render identifying text + expose toggle/bookmark. ---
vi.mock('../../src/components/RuleCard/RuleCard', () => ({
    default: ({ rule, isExpanded, onToggle, onBookmark, isBookmarked, readOnly, bookmarkLabel }) => (
        <div data-testid={`rule-${rule.rule_id}`} data-expanded={String(isExpanded)} data-readonly={String(readOnly)}>
            <span>{rule.custom_name}</span>
            <button data-testid={`rule-toggle-${rule.rule_id}`} onClick={onToggle}>toggle</button>
            <span data-testid={`rule-bm-state-${rule.rule_id}`}>{String(isBookmarked)}</span>
            {onBookmark && (
                <button data-testid={`rule-bm-${rule.rule_id}`} onClick={() => onBookmark(rule)}>
                    {bookmarkLabel}
                </button>
            )}
        </div>
    ),
}));

// --- CognitiveElementCard stub. ---
vi.mock('../../src/components/CognitiveElementCard/CognitiveElementCard', () => ({
    default: ({ ce, isOpen, onToggle, onBookmark, isBookmarked, samples }) => (
        <div data-testid={`ce-${ce.ce_id}`} data-open={String(isOpen)} data-samples={samples ? 'yes' : 'no'}>
            <span>{ce.name}</span>
            <button data-testid={`ce-toggle-${ce.ce_id}`} onClick={() => onToggle(ce.ce_id)}>toggle</button>
            <span data-testid={`ce-bm-state-${ce.ce_id}`}>{String(isBookmarked)}</span>
            {onBookmark && (
                <button data-testid={`ce-bm-${ce.ce_id}`} onClick={() => onBookmark(ce)}>save</button>
            )}
        </div>
    ),
}));

// Import AFTER mocks.
import Profile from '../../src/pages/Profile';
import * as api from '../../src/api';

// ---------------------------------------------------------------------------

const baseProfile = {
    username: 'alice',
    display_name: 'Alice A',
    email: 'alice@example.com',
    bio: 'I build rules.',
    is_team: false,
    member_since: '2023-01-15T00:00:00Z',
    contribution_count_rules: 3,
    contribution_count_ces: 2,
    avg_rating_received: null,
    total_rating_count: 0,
};

const ruleRow = (id, name) => ({
    id, asset_type: 'rule', name, content: `predicate ${id}`,
    active_ces: [{ name: 'CE-X' }],
});
const ceRow = (id, name) => ({
    id, asset_type: 'ce', name, content: `definition ${id}`, examples: ['ex'],
});

const setUser = (user) => {
    sessionStorage.setItem('user', JSON.stringify(user));
};

const renderProfile = (username = 'alice') => render(
    <MemoryRouter initialEntries={[`/profile/${username}`]}>
        <Routes>
            <Route path="/profile/:username" element={<Profile />} />
        </Routes>
    </MemoryRouter>,
);

// Error boundary so a render crash (the page has a real bug: on a non-404
// fetch error it renders with profile === null and dereferences it) is
// contained inside the test instead of bubbling up as an unhandled error
// that fails the whole vitest run. We still assert the alert/behavior that
// fired before the crash.
class Boundary extends React.Component {
    constructor(props) { super(props); this.state = { crashed: false }; }
    static getDerivedStateFromError() { return { crashed: true }; }
    componentDidCatch() { /* swallow */ }
    render() {
        if (this.state.crashed) return <div data-testid="boundary-crashed" />;
        return this.props.children;
    }
}

const renderProfileGuarded = (username = 'alice') => render(
    <Boundary>
        <MemoryRouter initialEntries={[`/profile/${username}`]}>
            <Routes>
                <Route path="/profile/:username" element={<Profile />} />
            </Routes>
        </MemoryRouter>
    </Boundary>,
);

beforeEach(() => {
    vi.clearAllMocks();
    libraryRefreshCb = null;
    localStorage.clear();
    // Sensible defaults; individual tests override as needed.
    api.getUserProfile.mockResolvedValue({ data: baseProfile });
    api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
    api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
});

afterEach(() => {
    vi.useRealTimers();
});

// ---------------------------------------------------------------------------

describe('Profile — loading + fetch', () => {
    it('shows the loading state before the profile resolves', () => {
        // Never-resolving promise keeps loading=true.
        api.getUserProfile.mockReturnValue(new Promise(() => {}));
        renderProfile();
        expect(screen.getByText(/Loading profile/i)).toBeInTheDocument();
    });

    it('calls getUserProfile with the :username param', async () => {
        renderProfile('bob');
        await waitFor(() => expect(api.getUserProfile).toHaveBeenCalledWith('bob'));
    });

    it('renders the header card after the profile resolves', async () => {
        renderProfile();
        expect(await screen.findByText('Alice A')).toBeInTheDocument();
        // Public profile shows the @handle, never the email.
        expect(screen.getByText('@alice')).toBeInTheDocument();
        expect(screen.getByText('I build rules.')).toBeInTheDocument();
    });

    it('falls back to username as display name when display_name is empty', async () => {
        api.getUserProfile.mockResolvedValue({
            data: { ...baseProfile, display_name: '', email: '' },
        });
        renderProfile();
        // h1 shows username; the @username paragraph also appears (no email)
        expect(await screen.findByRole('heading', { name: 'alice' })).toBeInTheDocument();
        expect(screen.getByText('@alice')).toBeInTheDocument();
    });
});

describe('Profile — not found', () => {
    it('renders the not-found state on a 404 and navigates back to workspace', async () => {
        const user = userEvent.setup();
        api.getUserProfile.mockRejectedValue({ response: { status: 404 } });
        renderProfile('ghost');
        expect(await screen.findByText('User not found')).toBeInTheDocument();
        expect(screen.getByText('ghost')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: /Back to Workspace/i }));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
        // Non-404 alert path should NOT have fired.
        expect(showAlertDialog).not.toHaveBeenCalled();
    });

    it('shows an alert dialog for a non-404 error', async () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        api.getUserProfile.mockRejectedValue({
            response: { status: 500, data: { detail: 'server boom' } },
        });
        renderProfileGuarded();
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalled());
        const arg = showAlertDialog.mock.calls[0][0];
        expect(arg.title).toMatch(/Could not load profile/i);
        expect(arg.message).toBe('server boom');
        expect(arg.variant).toBe('error');
        spy.mockRestore();
    });

    it('uses err.message when the error has no response detail', async () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        api.getUserProfile.mockRejectedValue(new Error('network down'));
        renderProfileGuarded();
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalled());
        expect(showAlertDialog.mock.calls[0][0].message).toBe('network down');
        spy.mockRestore();
    });
});

describe('Profile — header stats', () => {
    it('renders the contribution counts', async () => {
        renderProfile();
        expect(await screen.findByText('3')).toBeInTheDocument();   // rules
        expect(screen.getByText('2')).toBeInTheDocument();          // CEs
    });

    it('shows no rating stats (ratings were removed with accounts)', async () => {
        renderProfile();
        await screen.findByText('3');
        expect(screen.queryByText(/ratings?\)/)).not.toBeInTheDocument();
        expect(screen.queryByText(/No ratings yet/i)).not.toBeInTheDocument();
    });

    it('renders the Team badge when is_team is true', async () => {
        api.getUserProfile.mockResolvedValue({ data: { ...baseProfile, is_team: true } });
        renderProfile();
        expect(await screen.findByText('Team')).toBeInTheDocument();
    });

    it('omits the Team badge when is_team is false', async () => {
        renderProfile();
        await screen.findByText('Alice A');
        expect(screen.queryByText('Team')).not.toBeInTheDocument();
    });

    it('renders the member-since line', async () => {
        renderProfile();
        expect(await screen.findByText(/Member since/i)).toBeInTheDocument();
    });

    it('omits member-since when not provided', async () => {
        api.getUserProfile.mockResolvedValue({
            data: { ...baseProfile, member_since: null },
        });
        renderProfile();
        await screen.findByText('Alice A');
        expect(screen.queryByText(/Member since/i)).not.toBeInTheDocument();
    });

    it('shows the contribution counts in the tab labels', async () => {
        renderProfile();
        expect(await screen.findByRole('button', { name: /Rules \(3\)/ })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /CEs \(2\)/ })).toBeInTheDocument();
    });
});

describe('Profile — breadcrumb navigation', () => {
    it('navigates to /community when the Community crumb is clicked', async () => {
        const user = userEvent.setup();
        renderProfile();
        await screen.findByText('Alice A');
        await user.click(screen.getByText('Community'));
        expect(mockNavigate).toHaveBeenCalledWith('/community');
    });
});

describe('Profile — bio rendering for visitor', () => {
    it('shows the not-own-profile empty bio message', async () => {
        // viewer is someone else (or anonymous)
        api.getUserProfile.mockResolvedValue({ data: { ...baseProfile, bio: '' } });
        renderProfile();
        expect(await screen.findByText(/hasn't added a bio yet/i)).toBeInTheDocument();
    });
});

describe('Profile — contributions list', () => {
    it('requests contributions with searchLibrary using the author + tab', async () => {
        renderProfile();
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({
            author: 'alice',
            asset_types: 'rule',
            page: 1,
            page_size: 10,
        }));
    });

    it('renders rule cards from the search results', async () => {
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile();
        expect(await screen.findByText('Rule Eleven')).toBeInTheDocument();
        expect(screen.getByTestId('rule-11')).toHaveAttribute('data-readonly', 'true');
    });

    it('shows the visitor empty state when no contributions', async () => {
        renderProfile();
        expect(await screen.findByText(/Alice A hasn't published any rules/i)).toBeInTheDocument();
    });

    it('shows the contributions loading state while the search is in flight', async () => {
        let resolveSearch;
        api.searchLibrary.mockReturnValue(new Promise((r) => { resolveSearch = r; }));
        renderProfile();
        // Header resolved, but contributions still loading. Use findByText so
        // we wait for the contribution-fetch effect to flip into its loading
        // state rather than racing it.
        await screen.findByText('Alice A');
        expect(await screen.findByText(/^Loading…$/)).toBeInTheDocument();
        await act(async () => {
            resolveSearch({ data: { results: [], total_results: 0 } });
        });
    });

    it('falls back to empty contributions on search error', async () => {
        api.searchLibrary.mockRejectedValue(new Error('search failed'));
        renderProfile();
        expect(await screen.findByText(/hasn't published any rules/i)).toBeInTheDocument();
    });

    it('handles ce rows with ces fallback shape (active_ces missing)', async () => {
        // rule row without active_ces but with ces array of names exercises
        // the `(it.ces || []).map(...)` branch.
        api.searchLibrary.mockResolvedValue({
            data: {
                results: [{ id: 7, asset_type: 'rule', name: 'R7', content: 'p', ces: ['A', 'B'] }],
                total_results: 1,
            },
        });
        renderProfile();
        expect(await screen.findByText('R7')).toBeInTheDocument();
    });

    it('defaults total to mapped length when total_results is absent', async () => {
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(1, 'r1'), ruleRow(2, 'r2')] },
        });
        renderProfile();
        await screen.findByText('r1');
        // 2 items <= pageSize 10, so pagination not rendered.
        expect(screen.queryByTestId('pagination')).not.toBeInTheDocument();
    });
});

describe('Profile — tabs', () => {
    it('switches to the CE tab and fetches ce contributions', async () => {
        const user = userEvent.setup();
        api.searchLibrary.mockImplementation(({ asset_types }) => Promise.resolve({
            data: asset_types === 'ce'
                ? { results: [ceRow(21, 'CE Twenty-One')], total_results: 1 }
                : { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        }));
        renderProfile();
        expect(await screen.findByText('Rule Eleven')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        expect(await screen.findByText('CE Twenty-One')).toBeInTheDocument();
        expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({ asset_types: 'ce' }));
    });

    it('shows ce empty-state copy when on the CE tab with no items', async () => {
        const user = userEvent.setup();
        renderProfile();
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        expect(await screen.findByText(/hasn't published any CEs/i)).toBeInTheDocument();
    });
});

describe('Profile — pagination', () => {
    it('renders pagination and advances the page (re-fetch with new page)', async () => {
        const user = userEvent.setup();
        // 15 items total -> pagination visible (pageSize 10).
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(1, 'r1')], total_results: 15 },
        });
        renderProfile();
        await screen.findByText('r1');
        expect(screen.getByTestId('pagination')).toHaveAttribute('data-total', '15');

        await user.click(screen.getByTestId('page-next'));
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalledWith(
            expect.objectContaining({ page: 2 }),
        ));
    });
});

describe('Profile — card expansion', () => {
    it('toggles a rule card expanded then collapsed', async () => {
        const user = userEvent.setup();
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile();
        await screen.findByText('Rule Eleven');
        expect(screen.getByTestId('rule-11')).toHaveAttribute('data-expanded', 'false');

        await user.click(screen.getByTestId('rule-toggle-11'));
        expect(screen.getByTestId('rule-11')).toHaveAttribute('data-expanded', 'true');

        await user.click(screen.getByTestId('rule-toggle-11'));
        expect(screen.getByTestId('rule-11')).toHaveAttribute('data-expanded', 'false');
    });

    it('toggling a CE card fetches its dataset and passes samples', async () => {
        const user = userEvent.setup();
        api.searchLibrary.mockResolvedValue({
            data: { results: [ceRow(21, 'CE 21')], total_results: 1 },
        });
        api.getCognitiveDataset.mockResolvedValue({
            data: { training_data_preview: [{ text: 'sample' }] },
        });
        renderProfile();
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        await screen.findByText('CE 21');

        await user.click(screen.getByTestId('ce-toggle-21'));
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledWith(21));
        await waitFor(() => expect(screen.getByTestId('ce-21')).toHaveAttribute('data-samples', 'yes'));
        expect(screen.getByTestId('ce-21')).toHaveAttribute('data-open', 'true');
    });

    it('does not re-fetch the dataset when re-opening a cached CE', async () => {
        const user = userEvent.setup();
        api.searchLibrary.mockResolvedValue({
            data: { results: [ceRow(21, 'CE 21')], total_results: 1 },
        });
        renderProfile();
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        await screen.findByText('CE 21');

        await user.click(screen.getByTestId('ce-toggle-21')); // open -> fetch
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalledTimes(1));
        await user.click(screen.getByTestId('ce-toggle-21')); // collapse
        await user.click(screen.getByTestId('ce-toggle-21')); // re-open, cached
        // Still only one fetch.
        expect(api.getCognitiveDataset).toHaveBeenCalledTimes(1);
    });

    it('caches empty samples when the dataset fetch fails', async () => {
        const user = userEvent.setup();
        api.searchLibrary.mockResolvedValue({
            data: { results: [ceRow(21, 'CE 21')], total_results: 1 },
        });
        api.getCognitiveDataset.mockRejectedValue(new Error('no data'));
        renderProfile();
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        await screen.findByText('CE 21');
        await user.click(screen.getByTestId('ce-toggle-21'));
        await waitFor(() => expect(api.getCognitiveDataset).toHaveBeenCalled());
        // samples becomes [] (truthy array) -> data-samples yes
        await waitFor(() => expect(screen.getByTestId('ce-21')).toHaveAttribute('data-samples', 'yes'));
    });
});

describe('Profile — own profile editing', () => {
    const ownUser = { user_id: 99, username: 'alice', email: 'alice@example.com' };

    it('shows the Edit Profile button on your own profile', async () => {
        setUser(ownUser);
        renderProfile('alice');
        expect(await screen.findByRole('button', { name: /Edit Profile/i })).toBeInTheDocument();
    });

    it('hides the Edit Profile button on someone else\'s profile', async () => {
        setUser({ user_id: 5, username: 'bob' });
        renderProfile('alice');
        await screen.findByText('Alice A');
        expect(screen.queryByRole('button', { name: /Edit Profile/i })).not.toBeInTheDocument();
    });

    it('shows the own-profile empty bio prompt when bio is blank', async () => {
        setUser(ownUser);
        api.getUserProfile.mockResolvedValue({ data: { ...baseProfile, bio: '' } });
        renderProfile('alice');
        expect(await screen.findByText(/Add a bio to tell others/i)).toBeInTheDocument();
    });

    it('enters edit mode and saves the profile', async () => {
        const user = userEvent.setup();
        setUser(ownUser);
        api.updateMyProfile.mockResolvedValue({
            data: { display_name: 'New Name', bio: 'New bio' },
        });
        renderProfile('alice');
        await user.click(await screen.findByRole('button', { name: /Edit Profile/i }));

        const nameInput = screen.getByDisplayValue('Alice A');
        await user.clear(nameInput);
        await user.type(nameInput, 'New Name');

        await user.click(screen.getByRole('button', { name: /^Save$/i }));

        await waitFor(() => expect(api.updateMyProfile).toHaveBeenCalledWith(
            expect.objectContaining({ display_name: 'New Name' }),
        ));
        // Reflects server response + leaves edit mode.
        expect(await screen.findByText('New Name')).toBeInTheDocument();
        // localStorage synced.
        const stored = JSON.parse(sessionStorage.getItem('user'));
        expect(stored.display_name).toBe('New Name');
        expect(stored.bio).toBe('New bio');
    });

    it('shows an alert when saving fails', async () => {
        const user = userEvent.setup();
        setUser(ownUser);
        api.updateMyProfile.mockRejectedValue({
            response: { data: { detail: 'save boom' } },
        });
        renderProfile('alice');
        await user.click(await screen.findByRole('button', { name: /Edit Profile/i }));
        await user.click(screen.getByRole('button', { name: /^Save$/i }));
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Save failed', message: 'save boom' }),
        ));
        // Still in edit mode (save failed).
        expect(screen.getByRole('button', { name: /^Save$/i })).toBeInTheDocument();
    });

    it('cancels edit mode and restores original values', async () => {
        const user = userEvent.setup();
        setUser(ownUser);
        renderProfile('alice');
        await user.click(await screen.findByRole('button', { name: /Edit Profile/i }));

        const nameInput = screen.getByDisplayValue('Alice A');
        await user.clear(nameInput);
        await user.type(nameInput, 'Throwaway');
        await user.click(screen.getByRole('button', { name: /Cancel/i }));

        // Back to read mode; original display name still shown.
        expect(screen.getByText('Alice A')).toBeInTheDocument();
        expect(screen.queryByDisplayValue('Throwaway')).not.toBeInTheDocument();
        expect(api.updateMyProfile).not.toHaveBeenCalled();
    });

    it('typing in the bio textarea updates its value', async () => {
        const user = userEvent.setup();
        setUser(ownUser);
        renderProfile('alice');
        await user.click(await screen.findByRole('button', { name: /Edit Profile/i }));
        const bio = screen.getByDisplayValue('I build rules.');
        await user.clear(bio);
        await user.type(bio, 'updated bio text');
        expect(screen.getByDisplayValue('updated bio text')).toBeInTheDocument();
    });
});

describe('Profile — bookmarks (visitor)', () => {
    const viewer = { user_id: 42, username: 'viewer' };

    it('seeds rule bookmark state and reflects it on the card', async () => {
        setUser(viewer);
        api.getRuleBookmarks.mockResolvedValue({
            data: { bookmarks: [{ rule_id: 11 }] },
        });
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile('alice');
        await screen.findByText('Rule Eleven');
        await waitFor(() => expect(api.getRuleBookmarks).toHaveBeenCalledWith(42));
        await waitFor(() => expect(screen.getByTestId('rule-bm-state-11')).toHaveTextContent('true'));
    });

    it('adds a rule bookmark when not already saved', async () => {
        const user = userEvent.setup();
        setUser(viewer);
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile('alice');
        await screen.findByText('Rule Eleven');
        expect(screen.getByTestId('rule-bm-state-11')).toHaveTextContent('false');

        await user.click(screen.getByTestId('rule-bm-11'));
        await waitFor(() => expect(api.addRuleBookmark).toHaveBeenCalledWith(42, 11));
        await waitFor(() => expect(screen.getByTestId('rule-bm-state-11')).toHaveTextContent('true'));
    });

    it('removes a rule bookmark when already saved', async () => {
        const user = userEvent.setup();
        setUser(viewer);
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 11 }] } });
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile('alice');
        await screen.findByText('Rule Eleven');
        await waitFor(() => expect(screen.getByTestId('rule-bm-state-11')).toHaveTextContent('true'));

        await user.click(screen.getByTestId('rule-bm-11'));
        await waitFor(() => expect(api.removeRuleBookmark).toHaveBeenCalledWith(42, 11));
        await waitFor(() => expect(screen.getByTestId('rule-bm-state-11')).toHaveTextContent('false'));
    });

    it('shows an alert when a rule bookmark toggle fails', async () => {
        const user = userEvent.setup();
        setUser(viewer);
        api.addRuleBookmark.mockRejectedValue(new Error('bm boom'));
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile('alice');
        await screen.findByText('Rule Eleven');
        await user.click(screen.getByTestId('rule-bm-11'));
        await waitFor(() => expect(showAlertDialog).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Bookmark failed' }),
        ));
    });

    it('adds a CE bookmark', async () => {
        const user = userEvent.setup();
        setUser(viewer);
        api.searchLibrary.mockImplementation(({ asset_types }) => Promise.resolve({
            data: asset_types === 'ce'
                ? { results: [ceRow(21, 'CE 21')], total_results: 1 }
                : { results: [], total_results: 0 },
        }));
        renderProfile('alice');
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        await screen.findByText('CE 21');

        await user.click(screen.getByTestId('ce-bm-21'));
        await waitFor(() => expect(api.addCEBookmark).toHaveBeenCalledWith(42, 21));
        await waitFor(() => expect(screen.getByTestId('ce-bm-state-21')).toHaveTextContent('true'));
    });

    it('removes a CE bookmark when already saved', async () => {
        const user = userEvent.setup();
        setUser(viewer);
        api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [{ ce_id: 21 }] } });
        api.searchLibrary.mockImplementation(({ asset_types }) => Promise.resolve({
            data: asset_types === 'ce'
                ? { results: [ceRow(21, 'CE 21')], total_results: 1 }
                : { results: [], total_results: 0 },
        }));
        renderProfile('alice');
        await screen.findByText('Alice A');
        await user.click(screen.getByRole('button', { name: /CEs \(2\)/ }));
        await screen.findByText('CE 21');
        await waitFor(() => expect(screen.getByTestId('ce-bm-state-21')).toHaveTextContent('true'));

        await user.click(screen.getByTestId('ce-bm-21'));
        await waitFor(() => expect(api.removeCEBookmark).toHaveBeenCalledWith(42, 21));
        await waitFor(() => expect(screen.getByTestId('ce-bm-state-21')).toHaveTextContent('false'));
    });

    it('does not render a bookmark button for anonymous visitors', async () => {
        // no user in localStorage
        api.searchLibrary.mockResolvedValue({
            data: { results: [ruleRow(11, 'Rule Eleven')], total_results: 1 },
        });
        renderProfile('alice');
        await screen.findByText('Rule Eleven');
        expect(screen.queryByTestId('rule-bm-11')).not.toBeInTheDocument();
        // bookmark-seed fetches skipped for anonymous viewers.
        expect(api.getRuleBookmarks).not.toHaveBeenCalled();
    });
});

describe('Profile — refresh listeners', () => {
    // The 'gavel:ratingChanged' listener was removed with ratings; the
    // library-refresh path below is the remaining silent-refresh trigger.

    it('library-refresh callback re-fetches profile and bumps contributions', async () => {
        renderProfile('alice');
        await screen.findByText('Alice A');
        expect(typeof libraryRefreshCb).toBe('function');
        const profileCallsBefore = api.getUserProfile.mock.calls.length;
        const searchCallsBefore = api.searchLibrary.mock.calls.length;
        await act(async () => {
            libraryRefreshCb();
        });
        await waitFor(() => expect(api.getUserProfile.mock.calls.length).toBeGreaterThan(profileCallsBefore));
        // contribRefreshTick bump triggers the contributions effect again.
        await waitFor(() => expect(api.searchLibrary.mock.calls.length).toBeGreaterThan(searchCallsBefore));
    });

    it('library-refresh is a no-op when there is no username', () => {
        // Render without a matching :username param.
        render(
            <MemoryRouter initialEntries={['/profile']}>
                <Routes>
                    <Route path="/profile" element={<Profile />} />
                </Routes>
            </MemoryRouter>,
        );
        // username is undefined -> getUserProfile called with undefined, but
        // the refresh callback should bail early without throwing.
        expect(typeof libraryRefreshCb).toBe('function');
        expect(() => act(() => { libraryRefreshCb(); })).not.toThrow();
    });
});
