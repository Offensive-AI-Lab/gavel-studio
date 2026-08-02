// Behavior tests for RulePage — covers loading/empty/error states, the
// optional public/bookmark header, predicate block, Cognitive Element rows
// (expand/collapse + definition/examples/empty), and the Test Set
// view (bucket chip statuses + sample dialogue toggle).
//
// Follows the smoke-test pattern: mock EVERY '../api' export the page and its
// children might touch, stub nothing else heavy, set a local
// user in localStorage, and wrap in MemoryRouter with a matching Route so
// useParams resolves :ruleId.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// --- navigate spy ---------------------------------------------------------
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- API mock. Covers RulePage's own calls. -----------------------------
const getRuleDetail = vi.fn(() => Promise.resolve({ data: null }));
const previewRuleTestSets = vi.fn(() => Promise.resolve({ data: { default: { buckets: [] } } }));
const getRuleBookmarks = vi.fn(() => Promise.resolve({ data: { bookmarks: [] } }));
const addRuleBookmark = vi.fn(() => Promise.resolve({ data: {} }));
const removeRuleBookmark = vi.fn(() => Promise.resolve({ data: {} }));

vi.mock('../../src/api', () => ({
    default: { get: vi.fn(() => Promise.resolve({ data: {} })) },
    getCEBookmarks: vi.fn(() => Promise.resolve({ data: { bookmarks: [] } })),
    addCEBookmark: vi.fn(() => Promise.resolve({})),
    removeCEBookmark: vi.fn(() => Promise.resolve({})),
    getRuleDetail: (...a) => getRuleDetail(...a),
    previewRuleTestSets: (...a) => previewRuleTestSets(...a),
    getRuleBookmarks: (...a) => getRuleBookmarks(...a),
    addRuleBookmark: (...a) => addRuleBookmark(...a),
    removeRuleBookmark: (...a) => removeRuleBookmark(...a),
}));

import RulePage from '../../src/pages/RulePage';

const setUser = (user = { user_id: 7, username: 'alice', email: 'a@b.c' }) => {
    sessionStorage.setItem('token', 'fake-token');
    if (user === null) sessionStorage.removeItem('user');
    else sessionStorage.setItem('user', JSON.stringify(user));
};

const renderRule = (ruleId = '42') =>
    render(
        <MemoryRouter initialEntries={[`/rules/${ruleId}`]}>
            <Routes>
                <Route path="/rules/:ruleId" element={<RulePage />} />
            </Routes>
        </MemoryRouter>,
    );

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    // Reset default resolved values (clearAllMocks keeps impls).
    getRuleDetail.mockResolvedValue({ data: null });
    previewRuleTestSets.mockResolvedValue({ data: { default: { buckets: [] } } });
    getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    addRuleBookmark.mockResolvedValue({ data: {} });
    removeRuleBookmark.mockResolvedValue({ data: {} });
});

describe('RulePage — loading & basic states', () => {
    it('shows the loading indicator before data resolves', () => {
        // Make the detail fetch hang so loading stays true.
        getRuleDetail.mockReturnValue(new Promise(() => {}));
        renderRule('42');
        expect(screen.getByText('Loading…')).toBeInTheDocument();
    });

    it('falls back to "Rule #<id>" when detail has no name', async () => {
        renderRule('42');
        // Name appears in both the breadcrumb and the heading — target the heading.
        expect(await screen.findByRole('heading', { name: 'Rule #42' })).toBeInTheDocument();
    });

    it('renders the rule name from detail', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'No Self Harm' } });
        renderRule('5');
        expect(await screen.findByRole('heading', { name: 'No Self Harm' })).toBeInTheDocument();
    });

    it('handles a getRuleDetail rejection by showing the empty header', async () => {
        getRuleDetail.mockRejectedValue(new Error('boom'));
        renderRule('99');
        expect(await screen.findByRole('heading', { name: 'Rule #99' })).toBeInTheDocument();
        // Loading must have cleared even on error.
        await waitFor(() => expect(screen.queryByText('Loading…')).not.toBeInTheDocument());
    });

    it('handles a previewRuleTestSets rejection (preview stays null)', async () => {
        previewRuleTestSets.mockRejectedValue(new Error('nope'));
        renderRule('1');
        // No-test-set message renders because preview failed.
        expect(await screen.findByText(/No test set yet/i)).toBeInTheDocument();
    });
});

describe('RulePage — breadcrumb navigation', () => {
    it('navigates to /community when the Community crumb is clicked', async () => {
        renderRule('42');
        await screen.findByRole('heading', { name: 'Rule #42' });
        fireEvent.click(screen.getByText('Community'));
        expect(mockNavigate).toHaveBeenCalledWith('/community');
    });
});

describe('RulePage — firing logic block', () => {
    it('renders groups + condition when the rule carries them', async () => {
        // GET /rules/{id}/detail nests the v2 logic under `logic`.
        getRuleDetail.mockResolvedValue({ data: {
            name: 'R',
            predicate: 'A AND B',
            logic: {
                groups: { required: [{ ce_id: 1, name: 'A' }], option_1: [{ ce_id: 2, name: 'B' }] },
                condition: 'all of required and 1 of option_1',
                predicate: 'A AND B',
            },
        } });
        renderRule('3');
        expect(await screen.findByText('Firing Logic')).toBeInTheDocument();
        // Group names appear twice — the group pill and tinted inside the condition.
        expect(screen.getAllByText('required')).toHaveLength(2);
        expect(screen.getAllByText('option_1')).toHaveLength(2);
        expect(document.querySelector('code')).toHaveTextContent('all of required and 1 of option_1');
        // Raw predicate is hidden when groups exist.
        expect(screen.queryByText('A AND B')).not.toBeInTheDocument();
    });

    it('falls back to the predicate string for legacy rows without groups', async () => {
        // Legacy rows come back with logic.groups = {} — predicate display only.
        getRuleDetail.mockResolvedValue({ data: {
            name: 'R',
            predicate: 'A AND B',
            logic: { groups: {}, condition: '', predicate: 'A AND B' },
        } });
        renderRule('3');
        expect(await screen.findByText('Firing Logic')).toBeInTheDocument();
        expect(screen.getByText('A AND B')).toBeInTheDocument();
    });

    it('omits the firing-logic block when neither groups nor predicate exist', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R' } });
        renderRule('3');
        await screen.findByRole('heading', { name: 'R' });
        expect(screen.queryByText('Firing Logic')).not.toBeInTheDocument();
    });
});

describe('RulePage — cognitive elements', () => {
    it('shows the empty CE message when there are none', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R', ces: [] } });
        renderRule('3');
        expect(await screen.findByText('No cognitive elements linked to this rule.')).toBeInTheDocument();
        expect(screen.getByText('Cognitive Elements (0)')).toBeInTheDocument();
    });

    it('lists CEs with their count and group-membership chips', async () => {
        getRuleDetail.mockResolvedValue({
            data: {
                name: 'R',
                logic: {
                    groups: {
                        required: [{ ce_id: 1, name: 'CE Alpha' }],
                        option_1: [{ ce_id: 2, name: 'CE Beta' }],
                    },
                    condition: 'all of required and 1 of option_1',
                    predicate: 'CE Alpha AND CE Beta',
                },
                // Detail `ces` rows carry their group memberships too.
                ces: [
                    { ce_id: 1, name: 'CE Alpha', definition: 'd1', groups: ['required'] },
                    { ce_id: 2, name: 'CE Beta', groups: ['option_1'] },
                    { ce_id: 3, name: 'CE Ungrouped', groups: [] },
                ],
            },
        });
        renderRule('3');
        expect(await screen.findByText('Cognitive Elements (3)')).toBeInTheDocument();
        // Member names render in both the Firing Logic chips and the CE list.
        expect(screen.getAllByText('CE Alpha').length).toBeGreaterThanOrEqual(2);
        expect(screen.getByText('CE Ungrouped')).toBeInTheDocument();
        // Each CE row shows the group(s) it belongs to (chips render in the
        // logic block too, hence getAllByText).
        expect(screen.getAllByText('required').length).toBeGreaterThanOrEqual(2);
        expect(screen.getAllByText('option_1').length).toBeGreaterThanOrEqual(2);
    });

    it('expands a CE to reveal definition and examples, then collapses', async () => {
        getRuleDetail.mockResolvedValue({
            data: {
                name: 'R',
                ces: [{ ce_id: 1, name: 'My CE', definition: 'The definition text', examples: ['first ex', { input: 'obj ex' }] }],
            },
        });
        renderRule('3');
        const ceBtn = await screen.findByRole('button', { name: /My CE/i });
        // Collapsed: definition not visible yet.
        expect(screen.queryByText('The definition text')).not.toBeInTheDocument();
        fireEvent.click(ceBtn);
        expect(screen.getByText('The definition text')).toBeInTheDocument();
        expect(screen.getByText('Examples')).toBeInTheDocument();
        expect(screen.getByText('first ex')).toBeInTheDocument();
        // Object example renders its `input`.
        expect(screen.getByText('obj ex')).toBeInTheDocument();
        // Collapse again.
        fireEvent.click(ceBtn);
        expect(screen.queryByText('The definition text')).not.toBeInTheDocument();
    });

    it('shows "No definition or examples on file." for a bare CE when expanded', async () => {
        getRuleDetail.mockResolvedValue({
            data: { name: 'R', ces: [{ ce_id: 9, name: 'Bare CE' }] },
        });
        renderRule('3');
        const ceBtn = await screen.findByRole('button', { name: /Bare CE/i });
        fireEvent.click(ceBtn);
        expect(screen.getByText('No definition or examples on file.')).toBeInTheDocument();
    });
});

describe('RulePage — test set', () => {
    it('shows the no-test-set message when preview has no buckets', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R' } });
        previewRuleTestSets.mockResolvedValue({ data: { default: { buckets: [] } } });
        renderRule('3');
        expect(await screen.findByText(/No test set yet/i)).toBeInTheDocument();
    });

    it('renders the recommended set name, scenario, and bucket chips', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R' } });
        previewRuleTestSets.mockResolvedValue({
            data: {
                default: {
                    scenario_instructions: 'Test the rule under scenario X.',
                    buckets: [
                        { dataset_type: 'positive', status: 'ready', count: 12, samples: [] },
                        { dataset_type: 'negative', status: 'generating', samples: [] },
                        { dataset_type: 'positive_calibration', status: 'queued', samples: [] },
                    ],
                },
            },
        });
        renderRule('3');
        expect(await screen.findByText('Test Set')).toBeInTheDocument();
        expect(screen.getByText('Test the rule under scenario X.')).toBeInTheDocument();
        // Ready bucket shows count.
        expect(screen.getByText(/Positive: 12/)).toBeInTheDocument();
        // Generating shows ellipsis.
        expect(screen.getByText(/Negative: …/)).toBeInTheDocument();
        // Other status shows the raw status string.
        expect(screen.getByText(/Calibration: queued/)).toBeInTheDocument();
    });

    it('toggles the sample dialogues open and renders convo turns', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R' } });
        previewRuleTestSets.mockResolvedValue({
            data: {
                default: {
                    buckets: [
                        {
                            dataset_type: 'positive',
                            status: 'ready',
                            count: 1,
                            samples: [[
                                { role: 'user', content: 'Hello there' },
                                { role: 'assistant', content: 'Hi, how can I help?' },
                            ]],
                        },
                    ],
                },
            },
        });
        renderRule('3');
        const toggle = await screen.findByRole('button', { name: /Sample dialogues/i });
        // Convo content hidden until opened.
        expect(screen.queryByText(/Hello there/)).not.toBeInTheDocument();
        fireEvent.click(toggle);
        expect(screen.getByText(/Hello there/)).toBeInTheDocument();
        expect(screen.getByText(/Hi, how can I help\?/)).toBeInTheDocument();
        // Roles rendered as bold labels.
        expect(screen.getByText('user:')).toBeInTheDocument();
        expect(screen.getByText('assistant:')).toBeInTheDocument();
        // Close again.
        fireEvent.click(toggle);
        expect(screen.queryByText(/Hello there/)).not.toBeInTheDocument();
    });

    it('does not render the sample-dialogues toggle when no samples exist', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'R' } });
        previewRuleTestSets.mockResolvedValue({
            data: { default: { buckets: [{ dataset_type: 'positive', status: 'ready', count: 3, samples: [] }] } },
        });
        renderRule('3');
        await screen.findByText('Test Set');
        expect(screen.queryByRole('button', { name: /Sample dialogues/i })).not.toBeInTheDocument();
    });
});

describe('RulePage — public header & bookmarks', () => {
    it('renders the Save button for a published rule', async () => {
        getRuleDetail.mockResolvedValue({
            data: { name: 'Pub Rule', public_id: 'pub-123', created_by_username: 'bob' },
        });
        renderRule('42');
        // The bookmark ("Save") action is the published-rule affordance.
        expect(await screen.findByRole('button', { name: /Save/i })).toBeInTheDocument();
    });

    it('omits the public header entirely for an unpublished rule', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Draft Rule' } });
        renderRule('42');
        await screen.findByRole('heading', { name: 'Draft Rule' });
        expect(screen.queryByRole('button', { name: /Save/i })).not.toBeInTheDocument();
    });

    it('shows "Remove" when the rule is already bookmarked', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 42 }] } });
        renderRule('42');
        expect(await screen.findByRole('button', { name: /Remove/i })).toBeInTheDocument();
    });

    it('adds a bookmark when Save is clicked', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        renderRule('42');
        const saveBtn = await screen.findByRole('button', { name: /Save/i });
        fireEvent.click(saveBtn);
        await waitFor(() => expect(addRuleBookmark).toHaveBeenCalledWith(7, 42));
        expect(await screen.findByRole('button', { name: /Remove/i })).toBeInTheDocument();
    });

    it('removes a bookmark when Remove is clicked', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 42 }] } });
        renderRule('42');
        const removeBtn = await screen.findByRole('button', { name: /Remove/i });
        fireEvent.click(removeBtn);
        await waitFor(() => expect(removeRuleBookmark).toHaveBeenCalledWith(7, 42));
        expect(await screen.findByRole('button', { name: /Save/i })).toBeInTheDocument();
    });

    it('swallows a bookmark API error without crashing', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        addRuleBookmark.mockRejectedValue(new Error('fail'));
        renderRule('42');
        const saveBtn = await screen.findByRole('button', { name: /Save/i });
        fireEvent.click(saveBtn);
        await waitFor(() => expect(addRuleBookmark).toHaveBeenCalled());
        // State reverts: still shows Save (bmBusy cleared, bookmarked stayed false).
        expect(await screen.findByRole('button', { name: /Save/i })).toBeInTheDocument();
    });

    it('does not fetch bookmarks when no user is logged in', async () => {
        setUser(null);
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        renderRule('42');
        await screen.findByRole('heading', { name: 'Pub' });
        expect(getRuleBookmarks).not.toHaveBeenCalled();
        // No Save button without a user.
        expect(screen.queryByRole('button', { name: /Save/i })).not.toBeInTheDocument();
    });

    it('ignores a getRuleBookmarks rejection (stays not-bookmarked)', async () => {
        getRuleDetail.mockResolvedValue({ data: { name: 'Pub', public_id: 'p1' } });
        getRuleBookmarks.mockRejectedValue(new Error('down'));
        renderRule('42');
        expect(await screen.findByRole('button', { name: /Save/i })).toBeInTheDocument();
    });
});

describe('RulePage — fetch wiring', () => {
    it('parses the ruleId param to an int for the detail fetch', async () => {
        renderRule('77');
        await waitFor(() => expect(getRuleDetail).toHaveBeenCalledWith(77));
        expect(previewRuleTestSets).toHaveBeenCalledWith(77);
    });
});
