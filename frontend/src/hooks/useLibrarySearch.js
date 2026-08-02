import { useEffect, useRef, useState } from 'react';
import { searchLibrary } from '../api';

// Live-search defaults:
//   * 250ms debounce — issues at most one request per pause in input without
//     introducing perceptible latency on the rendered results.
//   * 2-character minimum — single-character queries are too low-signal for the
//     embedder and would generate spurious requests. Categories-only search is
//     still permitted when allowEmptyQuery is true.
const DEFAULT_DEBOUNCE_MS = 250;
const MIN_QUERY_LEN = 2;

/**
 * Live, debounced library search shared by Rules and CEs.
 *
 * Both Browse pages drive their results through this hook. They differ only by
 * the `assetTypes` argument — `['rule']` for the rules page, `['ce']` for the CE
 * page. The hook is the single place that owns:
 *   - debouncing keystrokes
 *   - dropping stale responses (race-safe via a request-id ref)
 *   - clearing state when filters go empty
 *   - mapping query/category/page changes into a single API call
 *
 * Returns `{ results, totalResults, loading, error, hasSearched }`. Components
 * only read these — they never call `searchLibrary` themselves.
 *
 * @param {object} args
 * @param {string} args.query             — the user's typed query
 * @param {string[]} args.categories      — selected category names
 * @param {number} args.page              — current page (1-based)
 * @param {number} args.pageSize          — items per page
 * @param {string[]} args.assetTypes      — ['rule'] | ['ce'] | ['rule','ce']
 * @param {number} [args.candidateLimit]  — backend retrieval pool size
 * @param {boolean} [args.allowEmptyQuery] — if true, categories-only search is OK
 * @param {number} [args.debounceMs]      — override debounce window
 */
export default function useLibrarySearch({
    query,
    categories,
    page,
    pageSize,
    assetTypes,
    author,
    candidateLimit = 80,
    allowEmptyQuery = true,
    debounceMs = DEFAULT_DEBOUNCE_MS,
    // Bump this (e.g. from a "Try again" button) to force the effect to
    // re-issue the current search even when no query/filter changed — used to
    // recover from a transient fetch error without making the user re-type.
    reloadKey = 0,
}) {
    const [results, setResults] = useState([]);
    const [totalResults, setTotalResults] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [hasSearched, setHasSearched] = useState(false);

    // Each call increments this. When a response comes back, it's only applied
    // if its id still matches — older in-flight requests are silently dropped
    // so a slow "ru" response can't clobber a fast "rule" response.
    const reqIdRef = useRef(0);

    // Stable string keys so the effect deps array doesn't change identity on
    // every parent re-render even when contents are equal.
    const categoriesKey = (categories || []).join(',');
    const assetTypesKey = (assetTypes || []).join(',');

    useEffect(() => {
        const trimmed = (query || '').trim();
        const hasQuery = trimmed.length >= MIN_QUERY_LEN;
        const hasCategories = (categories?.length || 0) > 0;
        const hasAuthor = !!(author && author.trim());
        // An author filter alone is a valid search — the empty-query
        // browse path on the backend accepts (categories OR author).
        const wantSearch = hasQuery || (allowEmptyQuery && (hasCategories || hasAuthor));

        if (!wantSearch) {
            setResults([]);
            setTotalResults(0);
            setError('');
            setHasSearched(false);
            setLoading(false);
            return undefined;
        }

        const myReqId = ++reqIdRef.current;
        const timer = setTimeout(async () => {
            setLoading(true);
            setHasSearched(true);
            setError('');
            try {
                const res = await searchLibrary({
                    // '*' tells the backend "categories-only browse" — preserved
                    // for backward compat with the existing route.
                    q: trimmed || '*',
                    categories: hasCategories ? categories.join(',') : undefined,
                    asset_types: (assetTypes || []).join(','),
                    author: hasAuthor ? author.trim() : undefined,
                    page,
                    page_size: pageSize,
                    candidate_limit: candidateLimit,
                });
                if (myReqId !== reqIdRef.current) return;
                const list = res.data?.results || [];
                setResults(list);
                setTotalResults(res.data?.total_results ?? list.length);
            } catch {
                if (myReqId !== reqIdRef.current) return;
                setError('Search failed. Please try again.');
                setResults([]);
                setTotalResults(0);
            } finally {
                if (myReqId === reqIdRef.current) setLoading(false);
            }
        }, debounceMs);

        return () => clearTimeout(timer);
        // categoriesKey / assetTypesKey are derived from the array contents above.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [query, categoriesKey, page, pageSize, assetTypesKey, author, candidateLimit, allowEmptyQuery, debounceMs, reloadKey]);

    return { results, totalResults, loading, error, hasSearched };
}
