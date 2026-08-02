// Global test setup. Runs once per test file before any test.
//
// What's here and why:
//   * @testing-library/jest-dom — adds matchers like toBeInTheDocument().
//   * fail-fast on console.error — silent React warnings during tests almost
//     always indicate a real bug; we surface them as test failures so a stray
//     "act() warning" or "key prop missing" doesn't slip through.
//   * a clean localStorage between tests so cross-test bleed doesn't lie about
//     "the user is logged in".

import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// jsdom doesn't implement scrollIntoView. Pages like RealtimeViewer call
// it in mount-effects to keep a chat scrolled to bottom. A no-op stub
// keeps those effects silent in tests.
if (typeof window !== 'undefined' && !window.HTMLElement.prototype.scrollIntoView) {
    window.HTMLElement.prototype.scrollIntoView = function () {};
}

beforeEach(() => {
    // Clean storage between tests so a previous test's seeded auth state can't
    // leak into the next. Auth (user/token/models) lives in sessionStorage so
    // separate tabs are separate users; other prefs (e.g. sidebar state) stay
    // in localStorage — clear both.
    localStorage.clear();
    sessionStorage.clear();
});

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
});
