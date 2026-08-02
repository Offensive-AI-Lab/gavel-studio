// Behavior tests for the useLibrarySearch hook.
//
// The hook owns the debounced, race-safe live-search shared by the Browse
// pages. It:
//   - debounces keystrokes (default 250ms) into a single searchLibrary call
//   - refuses to search for sub-2-char queries (unless categories/author give
//     it something to browse when allowEmptyQuery is true)
//   - maps query/categories/author/page into one API call
//   - drops stale in-flight responses via a request-id ref
//   - maps res.data.results -> results and res.data.total_results -> total
//   - surfaces a friendly error string on failure
//
// We mock '../api' (only searchLibrary is used) and drive the hook with
// renderHook + rerender, using fake timers to step the debounce window.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

vi.mock('../../src/api', () => ({
    searchLibrary: vi.fn(() => Promise.resolve({ data: { results: [], total_results: 0 } })),
}));

import { searchLibrary } from '../../src/api';
import useLibrarySearch from '../../src/hooks/useLibrarySearch';

const baseArgs = (over = {}) => ({
    query: '',
    categories: [],
    page: 1,
    pageSize: 10,
    assetTypes: ['rule'],
    author: '',
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('useLibrarySearch — empty / no-search states', () => {
    it('does not search when there is no query, category, or author', () => {
        vi.useFakeTimers();
        const { result } = renderHook(() => useLibrarySearch(baseArgs()));
        act(() => { vi.advanceTimersByTime(1000); });
        expect(searchLibrary).not.toHaveBeenCalled();
        expect(result.current.results).toEqual([]);
        expect(result.current.totalResults).toBe(0);
        expect(result.current.hasSearched).toBe(false);
        expect(result.current.loading).toBe(false);
        expect(result.current.error).toBe('');
    });

    it('does not search for a sub-minimum (1-char) query', () => {
        vi.useFakeTimers();
        renderHook(() => useLibrarySearch(baseArgs({ query: 'a' })));
        act(() => { vi.advanceTimersByTime(1000); });
        expect(searchLibrary).not.toHaveBeenCalled();
    });

    it('does not browse on categories/author when allowEmptyQuery is false', () => {
        vi.useFakeTimers();
        renderHook(() => useLibrarySearch(baseArgs({
            categories: ['Safety'], author: 'me', allowEmptyQuery: false,
        })));
        act(() => { vi.advanceTimersByTime(1000); });
        expect(searchLibrary).not.toHaveBeenCalled();
    });

    it('clears prior results when filters go empty', async () => {
        const { result, rerender } = renderHook((props) => useLibrarySearch(props), {
            initialProps: baseArgs({ query: 'rule' }),
        });
        searchLibrary.mockResolvedValueOnce({ data: { results: [{ id: 1 }], total_results: 1 } });
        await waitFor(() => expect(result.current.results.length).toBe(1));

        // Now empty the query — the no-search branch resets everything.
        rerender(baseArgs({ query: '' }));
        await waitFor(() => expect(result.current.results).toEqual([]));
        expect(result.current.totalResults).toBe(0);
        expect(result.current.hasSearched).toBe(false);
    });
});

describe('useLibrarySearch — search flow & mapping', () => {
    it('debounces keystrokes into a single request after the window', async () => {
        vi.useFakeTimers();
        const { rerender } = renderHook((props) => useLibrarySearch(props), {
            initialProps: baseArgs({ query: 'ru', debounceMs: 250 }),
        });
        rerender(baseArgs({ query: 'rul', debounceMs: 250 }));
        rerender(baseArgs({ query: 'rule', debounceMs: 250 }));
        // Before the window elapses, no call.
        act(() => { vi.advanceTimersByTime(200); });
        expect(searchLibrary).not.toHaveBeenCalled();
        // After the window, exactly one call for the final value.
        await act(async () => { await vi.advanceTimersByTimeAsync(250); });
        expect(searchLibrary).toHaveBeenCalledTimes(1);
        expect(searchLibrary.mock.calls[0][0]).toMatchObject({ q: 'rule' });
    });

    it('maps query/categories/author/page into the API payload', async () => {
        const { result } = renderHook(() => useLibrarySearch(baseArgs({
            query: '  hello  ',
            categories: ['Safety', 'Bias'],
            author: '  alice  ',
            page: 3,
            pageSize: 25,
            assetTypes: ['rule', 'ce'],
            candidateLimit: 50,
        })));
        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        expect(searchLibrary).toHaveBeenCalledWith({
            q: 'hello',
            categories: 'Safety,Bias',
            asset_types: 'rule,ce',
            author: 'alice',
            page: 3,
            page_size: 25,
            candidate_limit: 50,
        });
        await waitFor(() => expect(result.current.hasSearched).toBe(true));
    });

    it('maps results and total_results from the response', async () => {
        searchLibrary.mockResolvedValue({
            data: { results: [{ id: 1 }, { id: 2 }], total_results: 42 },
        });
        const { result } = renderHook(() => useLibrarySearch(baseArgs({ query: 'rule' })));
        await waitFor(() => expect(result.current.results.length).toBe(2));
        expect(result.current.totalResults).toBe(42);
        expect(result.current.loading).toBe(false);
    });

    it('falls back to the list length when total_results is missing', async () => {
        searchLibrary.mockResolvedValue({ data: { results: [{ id: 1 }, { id: 2 }, { id: 3 }] } });
        const { result } = renderHook(() => useLibrarySearch(baseArgs({ query: 'rule' })));
        await waitFor(() => expect(result.current.totalResults).toBe(3));
    });

    it('tolerates a response with no results key', async () => {
        searchLibrary.mockResolvedValue({ data: {} });
        const { result } = renderHook(() => useLibrarySearch(baseArgs({ query: 'rule' })));
        await waitFor(() => expect(result.current.hasSearched).toBe(true));
        expect(result.current.results).toEqual([]);
        expect(result.current.totalResults).toBe(0);
    });

    it('browses with "*" query when only categories are provided', async () => {
        renderHook(() => useLibrarySearch(baseArgs({ categories: ['Safety'] })));
        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        expect(searchLibrary.mock.calls[0][0]).toMatchObject({ q: '*', categories: 'Safety' });
    });

    it('browses on an author-only filter', async () => {
        renderHook(() => useLibrarySearch(baseArgs({ author: 'bob' })));
        await waitFor(() => expect(searchLibrary).toHaveBeenCalled());
        const payload = searchLibrary.mock.calls[0][0];
        expect(payload).toMatchObject({ q: '*', author: 'bob' });
        expect(payload.categories).toBeUndefined();
    });
});

describe('useLibrarySearch — error & race handling', () => {
    it('sets a friendly error and clears results when the request fails', async () => {
        searchLibrary.mockRejectedValue(new Error('boom'));
        const { result } = renderHook(() => useLibrarySearch(baseArgs({ query: 'rule' })));
        await waitFor(() => expect(result.current.error).toBe('Search failed. Please try again.'));
        expect(result.current.results).toEqual([]);
        expect(result.current.totalResults).toBe(0);
        expect(result.current.loading).toBe(false);
    });

    it('drops a stale response so it cannot clobber a newer query', async () => {
        let resolveSlow;
        const slow = new Promise((res) => { resolveSlow = res; });
        searchLibrary
            .mockReturnValueOnce(slow) // first request — slow, will resolve last
            .mockResolvedValueOnce({ data: { results: [{ id: 'fresh' }], total_results: 1 } }); // second — fast

        // Use fake timers so we can let the first debounce timer actually fire
        // (kicking off the slow request) BEFORE the second query supersedes it.
        vi.useFakeTimers();
        const { result, rerender } = renderHook((props) => useLibrarySearch(props), {
            initialProps: baseArgs({ query: 'rule', debounceMs: 250 }),
        });
        // Let the first debounce window elapse so the slow request is in flight.
        await act(async () => { await vi.advanceTimersByTimeAsync(250); });
        expect(searchLibrary).toHaveBeenCalledTimes(1);

        // Now change the query — its (fast) response will arrive and win.
        rerender(baseArgs({ query: 'rules', debounceMs: 250 }));
        await act(async () => { await vi.advanceTimersByTimeAsync(250); });
        await vi.waitFor(() => expect(result.current.results).toEqual([{ id: 'fresh' }]));

        // Finally let the stale first request resolve — it must be ignored.
        await act(async () => {
            resolveSlow({ data: { results: [{ id: 'stale' }], total_results: 99 } });
            await slow;
        });
        expect(result.current.results).toEqual([{ id: 'fresh' }]);
        expect(result.current.totalResults).toBe(1);
    });
});
