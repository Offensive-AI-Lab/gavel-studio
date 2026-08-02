// useLibraryRefresh — runs a refetch callback whenever any mutation
// across the app dispatches `gavel:libraryChanged`.
//
// Pages that show library content (models, guardrails, rules, CEs,
// bookmarks, drafts) call this hook with their own refetch function.
// Mutations live in api.js, where every mutating endpoint is wrapped
// with `withNotify(...)` so the event fires automatically on success —
// callers don't have to remember to dispatch.
//
// Why a hook + ref pattern (and not just useEffect with a deps array):
// passing the refetch callback as a deps-array value would re-register
// the listener on every render, which is wasteful and races during
// rapid state churn. Using a ref means we register exactly once on
// mount, but the listener still calls the LATEST refetch closure each
// time it fires — so the callback always reads current state.
//
// Without this ref pattern we hit the same stale-closure bug the
// Sidebar had: a listener bound on first render kept calling a
// fetchModels that closed over the first render's expandedModels=[],
// so child rows never refreshed even though the model list did.

import { useEffect, useRef } from 'react';

export const useLibraryRefresh = (refetch) => {
    const ref = useRef(refetch);
    ref.current = refetch;

    useEffect(() => {
        const handler = () => {
            try {
                ref.current?.();
            } catch (err) {
                // A listener throwing must not break the dispatch chain
                // for any other listening page. Refetch failures are
                // page-local; log and move on.
                console.warn('[useLibraryRefresh] refetch threw:', err);
            }
        };
        window.addEventListener('gavel:libraryChanged', handler);
        return () => window.removeEventListener('gavel:libraryChanged', handler);
    }, []);
};

export default useLibraryRefresh;
