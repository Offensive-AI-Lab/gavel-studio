// Behavior tests for LibrarySearch.jsx.
//
// Follows the established pattern in pages.smoke.test.jsx: mock '../api' with
// benign defaults for every export this page (and its child components —
// Layout/Sidebar, RuleCard, CognitiveElementCard) might touch,
// stub the Sidebar, wrap in MemoryRouter, and seed a logged-in user in
// localStorage so the page renders as authenticated.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// --- navigate spy: capture useNavigate calls without leaving the router ---
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- API mock. searchLibrary is the only call LibrarySearch makes directly,
// too. Everything resolves benign so nothing hits the network. ---
const searchLibrary = vi.fn(() => Promise.resolve({ data: { results: [] } }));
vi.mock('../../src/api', () => ({
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    addCEBookmark: vi.fn(() => Promise.resolve({})),
    removeCEBookmark: vi.fn(() => Promise.resolve({})),
    default: {
        get: vi.fn(() => Promise.resolve({ data: {} })),
        post: vi.fn(() => Promise.resolve({ data: {} })),
        delete: vi.fn(() => Promise.resolve({ data: {} })),
        put: vi.fn(() => Promise.resolve({ data: {} })),
    },
    searchLibrary: (...args) => searchLibrary(...args),
}));

// Stub the Sidebar (rendered inside Layout) — it has its own fetches/routing.
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));

import LibrarySearch from '../../src/pages/LibrarySearch';

const renderPage = () =>
    render(
        <MemoryRouter initialEntries={['/library/search']}>
            <LibrarySearch />
        </MemoryRouter>,
    );

const setUser = (user = { user_id: 1, username: 'alice', email: 'a@b.c' }) => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify(user));
};

const ruleResult = (over = {}) => ({
    id: 1,
    asset_type: 'rule',
    name: 'Finance Policy',
    content: 'A AND B',
    ces: ['CE One', { name: 'CE Two' }],
    ...over,
});

const ceResult = (over = {}) => ({
    id: 7,
    asset_type: 'ce',
    name: 'Greeting',
    content: 'Says hello',
    categories: ['Safety'],
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    searchLibrary.mockResolvedValue({ data: { results: [] } });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('LibrarySearch — initial render', () => {
    it('renders the header, subhead and search input', () => {
        renderPage();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByText('Search Rules & Cognitive Elements')).toBeInTheDocument();
        expect(screen.getByText('Unified Library')).toBeInTheDocument();
        expect(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
        ).toBeInTheDocument();
    });

    it('shows the empty-state hint and no error initially', () => {
        renderPage();
        expect(
            screen.getByText(/No results yet/i),
        ).toBeInTheDocument();
        expect(document.querySelector('.alert')).toBeNull();
    });

    it('does NOT auto-search on mount (query starts empty)', () => {
        renderPage();
        expect(searchLibrary).not.toHaveBeenCalled();
    });

    it('shows the signed-in pill with the username', () => {
        renderPage();
        expect(screen.getByText(/Signed in as alice/)).toBeInTheDocument();
    });

    it('falls back to email when the user has no username', () => {
        setUser({ user_id: 2, email: 'noname@x.io' });
        renderPage();
        expect(screen.getByText(/Signed in as noname@x.io/)).toBeInTheDocument();
    });

    it('hides the signed-in pill when there is no stored user', () => {
        sessionStorage.removeItem('user');
        // user becomes null; component guards with `user &&`
        renderPage();
        expect(screen.queryByText(/Signed in as/)).toBeNull();
    });
});

describe('LibrarySearch — validation', () => {
    it('shows an error and does not search when query is blank', () => {
        renderPage();
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));
        expect(screen.getByText('Enter a query to search')).toBeInTheDocument();
        expect(searchLibrary).not.toHaveBeenCalled();
    });

    it('shows an error when query is only whitespace', () => {
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: '   ' } });
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));
        expect(screen.getByText('Enter a query to search')).toBeInTheDocument();
        expect(searchLibrary).not.toHaveBeenCalled();
    });
});

describe('LibrarySearch — successful search', () => {
    it('calls searchLibrary with trimmed query and default params', async () => {
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: '  finance  ' } });
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        await waitFor(() => expect(searchLibrary).toHaveBeenCalledTimes(1));
        expect(searchLibrary).toHaveBeenCalledWith({
            q: 'finance',
            categories: undefined,
            asset_types: 'rule,ce',
            top_k: 10,
            candidate_limit: 80,
        });
    });

    it('passes categories when provided', async () => {
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'policy' } },
        );
        fireEvent.change(
            screen.getByPlaceholderText('Finance, Security, Safety'),
            { target: { value: ' Finance ' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        expect(searchLibrary.mock.calls[0][0].categories).toBe('Finance');
    });

    it('renders the results count header and both card sections', async () => {
        searchLibrary.mockResolvedValue({
            data: { results: [ruleResult(), ceResult()] },
        });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'something' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        expect(await screen.findByText(/Search Results \(2 found\)/)).toBeInTheDocument();
        expect(screen.getByText('Rules')).toBeInTheDocument();
        expect(screen.getByText('Cognitive Elements')).toBeInTheDocument();
        expect(screen.getByText('Finance Policy')).toBeInTheDocument();
        expect(screen.getByText('Greeting')).toBeInTheDocument();
        // empty-state hint is gone once results exist
        expect(screen.queryByText(/No results yet/i)).toBeNull();
    });

    it('renders only the Rules section when there are no CE results', async () => {
        searchLibrary.mockResolvedValue({ data: { results: [ruleResult()] } });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'rule only' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        await screen.findByText('Finance Policy');
        expect(screen.getByText('Rules')).toBeInTheDocument();
        expect(screen.queryByText('Cognitive Elements')).toBeNull();
    });

    it('treats non-rule asset_types as CEs', async () => {
        searchLibrary.mockResolvedValue({
            data: { results: [ceResult({ asset_type: 'ce' }), ceResult({ id: 9, name: 'Other', asset_type: undefined })] },
        });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'ces' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        await screen.findByText('Greeting');
        expect(screen.getByText('Cognitive Elements')).toBeInTheDocument();
        expect(screen.getByText('Other')).toBeInTheDocument();
        expect(screen.queryByText('Rules')).toBeNull();
    });

    it('handles a response with a missing results array (defaults to [])', async () => {
        searchLibrary.mockResolvedValue({ data: {} });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'empty' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        expect(await screen.findByText(/No results yet/i)).toBeInTheDocument();
    });
});

describe('LibrarySearch — loading state', () => {
    it('shows the searching skeleton and disabled button while in flight', async () => {
        let resolveSearch;
        searchLibrary.mockImplementation(
            () => new Promise((res) => { resolveSearch = res; }),
        );
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'pending' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        // loading=true → skeleton + "Searching..." label
        expect(await screen.findByText(/Running hybrid search/i)).toBeInTheDocument();
        const searchBtn = screen.getByRole('button', { name: /Searching/i });
        expect(searchBtn).toBeDisabled();

        resolveSearch({ data: { results: [] } });
        await waitFor(() =>
            expect(screen.queryByText(/Running hybrid search/i)).toBeNull(),
        );
    });
});

describe('LibrarySearch — error state', () => {
    it('shows a failure message and clears results when search rejects', async () => {
        searchLibrary.mockRejectedValue(new Error('boom'));
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'kaboom' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));

        expect(
            await screen.findByText('Search failed. Please try again.'),
        ).toBeInTheDocument();
        // alert present means the empty-state hint is suppressed (error branch)
        expect(screen.queryByText(/No results yet/i)).toBeNull();
    });

    it('recovers: a successful search after a failed one clears the error', async () => {
        searchLibrary.mockRejectedValueOnce(new Error('boom'));
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: 'fail first' } });
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));
        await screen.findByText('Search failed. Please try again.');

        searchLibrary.mockResolvedValue({ data: { results: [ruleResult()] } });
        fireEvent.click(screen.getByRole('button', { name: /^Search/i }));
        await screen.findByText('Finance Policy');
        expect(screen.queryByText('Search failed. Please try again.')).toBeNull();
    });
});

describe('LibrarySearch — keyboard & buttons', () => {
    it('triggers a search on Enter key in the query input', async () => {
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: 'enter search' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await waitFor(() => expect(searchLibrary).toHaveBeenCalledTimes(1));
    });

    it('does not search on non-Enter keys', () => {
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: 'abc' } });
        fireEvent.keyDown(input, { key: 'a' });
        expect(searchLibrary).not.toHaveBeenCalled();
    });

    it('Clear button empties the query input', () => {
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        fireEvent.change(input, { target: { value: 'temp' } });
        expect(input.value).toBe('temp');
        fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
        expect(input.value).toBe('');
    });

    it('Reset (categories) empties the categories input', () => {
        renderPage();
        const cat = screen.getByPlaceholderText('Finance, Security, Safety');
        fireEvent.change(cat, { target: { value: 'Security' } });
        expect(cat.value).toBe('Security');
        // The category "Reset" lives next to the categories input; the other
        // "Reset" is the section-level secondary button. Scope to the former.
        const catGroupReset = cat.closest('.input-group').querySelector('button');
        fireEvent.click(catGroupReset);
        expect(cat.value).toBe('');
    });

    it('full Reset clears query, categories, results and error', async () => {
        searchLibrary.mockResolvedValue({ data: { results: [ruleResult()] } });
        renderPage();
        const input = screen.getByPlaceholderText('Search intents, policies, detectors...');
        const cat = screen.getByPlaceholderText('Finance, Security, Safety');
        fireEvent.change(input, { target: { value: 'q' } });
        fireEvent.change(cat, { target: { value: 'c' } });
        fireEvent.click(screen.getByRole('button', { name: /^Search$/i }));
        await screen.findByText('Finance Policy');

        // The section-level full reset is the secondary button in .actions.
        const secondaryReset = document.querySelector('.actions .secondary');
        fireEvent.click(secondaryReset);

        expect(input.value).toBe('');
        expect(cat.value).toBe('');
        expect(screen.queryByText('Finance Policy')).toBeNull();
        expect(await screen.findByText(/No results yet/i)).toBeInTheDocument();
    });
});

describe('LibrarySearch — topK control', () => {
    it('updates top_k from the number input', async () => {
        renderPage();
        const number = document.querySelector('input[type="number"]');
        expect(number.value).toBe('10');
        fireEvent.change(number, { target: { value: '25' } });

        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'topk' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search$/i }));
        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        expect(searchLibrary.mock.calls[0][0].top_k).toBe(25);
    });

    it('falls back to 10 when the number input is cleared/NaN', async () => {
        renderPage();
        const number = document.querySelector('input[type="number"]');
        fireEvent.change(number, { target: { value: '' } });
        expect(number.value).toBe('10');
    });
});

describe('LibrarySearch — header navigation', () => {
    it('Hub breadcrumb navigates to /workspace', () => {
        renderPage();
        fireEvent.click(screen.getByText('Hub'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
    });
});

describe('LibrarySearch — card expansion', () => {
    it('toggles a rule card open and closed', async () => {
        searchLibrary.mockResolvedValue({ data: { results: [ruleResult()] } });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'rule expand' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search$/i }));
        await screen.findByText('Finance Policy');

        // predicate (content) is only visible once expanded
        expect(screen.queryByText('A AND B')).toBeNull();
        fireEvent.click(screen.getByText('Finance Policy'));
        expect(await screen.findByText('A AND B')).toBeInTheDocument();

        // collapse again
        fireEvent.click(screen.getByText('Finance Policy'));
        await waitFor(() => expect(screen.queryByText('A AND B')).toBeNull());
    });

    it('toggles a CE card open', async () => {
        searchLibrary.mockResolvedValue({ data: { results: [ceResult()] } });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'ce expand' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search$/i }));
        await screen.findByText('Greeting');

        // "Examples" label appears only in expanded CE content
        expect(screen.queryByText('Examples')).toBeNull();
        fireEvent.click(screen.getByText('Greeting'));
        expect(await screen.findByText('Examples')).toBeInTheDocument();
    });

    it('renders a rule whose ces is empty without crashing', async () => {
        searchLibrary.mockResolvedValue({
            data: { results: [ruleResult({ ces: [], content: '' })] },
        });
        renderPage();
        fireEvent.change(
            screen.getByPlaceholderText('Search intents, policies, detectors...'),
            { target: { value: 'no ces' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^Search$/i }));
        await screen.findByText('Finance Policy');
        // expand to hit the "No predicate available" fallback path
        fireEvent.click(screen.getByText('Finance Policy'));
        expect(await screen.findByText('No predicate available')).toBeInTheDocument();
    });
});
