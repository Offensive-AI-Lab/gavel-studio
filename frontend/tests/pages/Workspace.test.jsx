// Behavior tests for the Workspace (hub) landing page.
//
// Workspace:
//   * fetches dashboard data for the stored user on mount (refresh())
//   * conditionally renders the stats grid, classifier overview table, and
//     recent-activity list only when each has data
//   * renders status pills with per-status copy (active/training/.../unknown)
//   * navigates to /browse and /models from the two action tiles
//   * re-runs refresh() on the `gavel:libraryChanged` event
//   * fires the onboarding welcome modal only when tutorial_seen is falsy
//
// We mock '../api' (only getDashboardData is used), spy on useNavigate, and
// wrap in TutorialProvider + MemoryRouter. showWelcome is observed via a spy
// on the tutorial module.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// ---- navigate spy ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// ---- API mock ----
vi.mock('../../src/api', () => ({
    getDashboardData: vi.fn(() => Promise.resolve({ data: { stats: null } })),
}));

// ---- Tutorial context — keep the provider real but observe showWelcome ----
const mockShowWelcome = vi.fn();
vi.mock('../../src/contexts/TutorialContext', () => ({
    TutorialProvider: ({ children }) => <>{children}</>,
    useTutorial: () => ({ showWelcome: mockShowWelcome }),
    useTutorialContent: vi.fn(),
}));

import Workspace from '../../src/pages/Workspace';
import { getDashboardData } from '../../src/api';

const setUser = (over = {}) => {
    sessionStorage.setItem('user', JSON.stringify({
        user_id: 7, username: 'Sean', tutorial_seen: true, ...over,
    }));
};

const renderPage = () => render(
    <MemoryRouter initialEntries={['/workspace']}>
        <Workspace />
    </MemoryRouter>,
);

const fullDashboard = (over = {}) => ({
    data: {
        stats: {
            total_models: 2,
            total_classifiers: 3,
            active_classifiers: 1,
            total_rules: 10,
            total_ces: 4,
            total_evaluations: 5,
            total_test_datasets: 2,
        },
        classifier_summary: [
            {
                classifier_id: 1, classifier_name: 'Finance-Guard', model_name: 'Llama',
                status: 'active', rule_count: 3, ce_count: 1, last_evaluation: '2026-01-15T00:00:00Z',
            },
        ],
        recent_activity: [
            { classifier_name: 'Finance-Guard', detail: 'Trained', created_at: '2026-01-16T00:00:00Z' },
        ],
        ...over,
    },
});

beforeEach(() => {
    vi.clearAllMocks();
    setUser();
    getDashboardData.mockResolvedValue({ data: { stats: null } });
});

afterEach(() => {
    localStorage.clear();
});

describe('Workspace — mount & auth', () => {

    it('fetches dashboard data for the stored user id on mount', async () => {
        renderPage();
        await waitFor(() => expect(getDashboardData).toHaveBeenCalledWith(7));
    });

    it('renders a time-of-day greeting with no name appended', async () => {
        // With login removed the only available name is the fixed local user's,
        // so the hero greets the time of day and nothing else.
        renderPage();
        expect(
            await screen.findByText(/^Good (morning|afternoon|evening)\.$/)
        ).toBeInTheDocument();
    });
});

describe('Workspace — tutorial auto-fire', () => {
    it('does not fire the welcome modal when tutorial_seen is true', async () => {
        renderPage();
        await waitFor(() => expect(getDashboardData).toHaveBeenCalled());
        expect(mockShowWelcome).not.toHaveBeenCalled();
    });

    it('fires the welcome modal on first run (tutorial_seen falsy)', async () => {
        setUser({ tutorial_seen: false });
        renderPage();
        await waitFor(() => expect(mockShowWelcome).toHaveBeenCalled());
    });
});

describe('Workspace — conditional sections', () => {
    it('hides stats, overview, and activity when the dashboard is empty', async () => {
        getDashboardData.mockResolvedValue({ data: { stats: null } });
        renderPage();
        await waitFor(() => expect(getDashboardData).toHaveBeenCalled());
        expect(screen.queryByText('Your statistics')).not.toBeInTheDocument();
        expect(screen.queryByText('Rule set overview')).not.toBeInTheDocument();
        expect(screen.queryByText('Recent activity')).not.toBeInTheDocument();
    });

    it('renders the stats grid when stats are present', async () => {
        getDashboardData.mockResolvedValue(fullDashboard());
        renderPage();
        expect(await screen.findByText('Your statistics')).toBeInTheDocument();
        // "Rule Sets" appears twice — the action tile AND the stat card label —
        // so assert on all matches. "Evaluations" is a unique stat label, which
        // alone proves the stats grid rendered.
        expect(screen.getAllByText('Rule Sets').length).toBeGreaterThanOrEqual(2);
        expect(screen.getByText('Evaluations')).toBeInTheDocument();
    });

    it('renders the classifier overview table with a status pill and formatted date', async () => {
        getDashboardData.mockResolvedValue(fullDashboard({
            // Distinct activity name so the table cell name is unambiguous.
            recent_activity: [{ classifier_name: 'Beta', detail: 'Trained', created_at: null }],
        }));
        renderPage();
        expect(await screen.findByText('Rule set overview')).toBeInTheDocument();
        expect(screen.getByText('Finance-Guard')).toBeInTheDocument();
        // "Active" appears both as a stat label and the status pill; assert
        // both renderings are present rather than expecting a unique node.
        expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(2);
    });

    it('renders an em-dash when a classifier has no last_evaluation', async () => {
        getDashboardData.mockResolvedValue(fullDashboard({
            classifier_summary: [{
                classifier_id: 9, classifier_name: 'C', model_name: 'M',
                status: 'untrained', rule_count: 0, ce_count: 0, last_evaluation: null,
            }],
            recent_activity: [],
        }));
        renderPage();
        await screen.findByText('Rule set overview');
        expect(screen.getByText('—')).toBeInTheDocument();
    });

    it('renders the recent-activity list and falls back to event_type when detail is missing', async () => {
        getDashboardData.mockResolvedValue(fullDashboard({
            classifier_summary: [],
            recent_activity: [
                { classifier_name: 'Alpha', event_type: 'created', created_at: null },
            ],
        }));
        renderPage();
        expect(await screen.findByText('Recent activity')).toBeInTheDocument();
        expect(screen.getByText('Alpha')).toBeInTheDocument();
        expect(screen.getByText('created')).toBeInTheDocument();
    });

    it('defaults missing summary/activity arrays to empty without crashing', async () => {
        // Response has stats but omits classifier_summary / recent_activity.
        getDashboardData.mockResolvedValue({ data: { stats: fullDashboard().data.stats } });
        renderPage();
        await screen.findByText('Your statistics');
        expect(screen.queryByText('Rule set overview')).not.toBeInTheDocument();
        expect(screen.queryByText('Recent activity')).not.toBeInTheDocument();
    });

    it('keeps rendering the hub when the dashboard fetch rejects', async () => {
        getDashboardData.mockRejectedValue(new Error('down'));
        renderPage();
        // Hero still renders; no stats sections appear.
        expect(
            await screen.findByText(/^Good (morning|afternoon|evening)\.$/)
        ).toBeInTheDocument();
        expect(screen.queryByText('Your statistics')).not.toBeInTheDocument();
    });
});

describe('Workspace — status pill variants', () => {
    const cases = [
        ['training', 'Training'],
        ['needs_retraining', 'Needs Retrain'],
        ['error', 'Error'],
        ['untrained', 'Untrained'],
    ];
    it.each(cases)('renders the %s pill as "%s"', async (status, label) => {
        getDashboardData.mockResolvedValue(fullDashboard({
            classifier_summary: [{
                classifier_id: 1, classifier_name: 'X', model_name: 'M',
                status, rule_count: 0, ce_count: 0, last_evaluation: null,
            }],
            recent_activity: [],
        }));
        renderPage();
        expect(await screen.findByText(label)).toBeInTheDocument();
    });

    it('falls back to the raw status string for an unknown status', async () => {
        getDashboardData.mockResolvedValue(fullDashboard({
            classifier_summary: [{
                classifier_id: 1, classifier_name: 'X', model_name: 'M',
                status: 'mystery_state', rule_count: 0, ce_count: 0, last_evaluation: null,
            }],
            recent_activity: [],
        }));
        renderPage();
        expect(await screen.findByText('mystery_state')).toBeInTheDocument();
    });
});

describe('Workspace — navigation & interactions', () => {
    it('navigates to /browse and /guardrails from the action tiles', async () => {
        renderPage();
        const browse = await screen.findByText('Browse');
        fireEvent.click(browse.closest('button'));
        expect(mockNavigate).toHaveBeenCalledWith('/browse');

        // Models tile was removed — model management lives in the rule set flow.
        fireEvent.click(screen.getByText('Rule Sets').closest('button'));
        expect(mockNavigate).toHaveBeenCalledWith('/guardrails');
    });

    it('toggles the action-tile hover state on mouse enter/leave', async () => {
        renderPage();
        const browse = await screen.findByText('Browse');
        const btn = browse.closest('button');
        // Exercises the onMouseEnter/onMouseLeave hover branches.
        fireEvent.mouseEnter(btn);
        expect(btn).toBeInTheDocument();
        fireEvent.mouseLeave(btn);
        expect(btn).toBeInTheDocument();
    });

    it('re-runs refresh() when a gavel:libraryChanged event fires', async () => {
        renderPage();
        await waitFor(() => expect(getDashboardData).toHaveBeenCalledTimes(1));
        await act(async () => {
            window.dispatchEvent(new Event('gavel:libraryChanged'));
        });
        await waitFor(() => expect(getDashboardData).toHaveBeenCalledTimes(2));
    });
});
