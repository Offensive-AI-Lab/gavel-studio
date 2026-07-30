// Behavior tests for RealtimeViewer.
//
// Follows the established pages.smoke.test.jsx pattern: mock '../api' with
// benign defaults for every export the page (and Layout/Sidebar children)
// might call, stub the Sidebar, wrap in MemoryRouter (+ matching Route so
// useParams reads :classifierId), and set a logged-in user in localStorage.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// --- Hoisted mock fns so individual tests can override per-call behavior.
const mocks = vi.hoisted(() => ({
    analyzeRealtime: vi.fn(),
    analyzeStored: vi.fn(),
    listSampleGroups: vi.fn(),
    getSampleGroup: vi.fn(),
    getClassifierDetails: vi.fn(),
    // Warm-session lifecycle. Default (set in beforeEach) is the local-fallback
    // path, so the component runs the model in-process — exactly what these tests
    // exercise via analyzeRealtime/analyzeStored.
    startRealtimeSession: vi.fn(),
    getRealtimeSessionStatus: vi.fn(),
    realtimeSessionKeepalive: vi.fn(),
    endRealtimeSession: vi.fn(),
    endRealtimeSessionUnload: vi.fn(),
    sessionAnalyzeStored: vi.fn(),
    sessionAnalyzeLive: vi.fn(),
    navigate: vi.fn(),
}));

vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return {
        default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()), delete: vi.fn(() => empty()), put: vi.fn(() => empty()) },
        analyzeRealtime: mocks.analyzeRealtime,
        analyzeStored: mocks.analyzeStored,
        listSampleGroups: mocks.listSampleGroups,
        getSampleGroup: mocks.getSampleGroup,
        getClassifierDetails: mocks.getClassifierDetails,
        startRealtimeSession: mocks.startRealtimeSession,
        getRealtimeSessionStatus: mocks.getRealtimeSessionStatus,
        realtimeSessionKeepalive: mocks.realtimeSessionKeepalive,
        endRealtimeSession: mocks.endRealtimeSession,
        endRealtimeSessionUnload: mocks.endRealtimeSessionUnload,
        sessionAnalyzeStored: mocks.sessionAnalyzeStored,
        sessionAnalyzeLive: mocks.sessionAnalyzeLive,
    };
});

// Stub the Sidebar — its own data fetches/routing are irrelevant here.
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));

// Keep useNavigate observable while leaving the rest of react-router-dom intact.
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mocks.navigate };
});

import RealtimeViewer from '../../src/pages/RealtimeViewer';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 1, email: 'a@b.c' }));
};

const renderViewer = () => render(
    <MemoryRouter initialEntries={['/classifiers/42/monitor']}>
        <Routes>
            <Route path="/classifiers/:classifierId/monitor" element={<RealtimeViewer />} />
        </Routes>
    </MemoryRouter>,
);

// "Live Chat" / "Test Samples" also appear in the Explainer copy, so target
// the actual toggle buttons inside the mode toggle group by label substring.
const clickMode = (label) => {
    const toggle = document.querySelector('.rtv-mode-toggle');
    const btn = within(toggle).getByText(label).closest('button');
    fireEvent.click(btn);
};

// A CE name appears in both the "Cognitive Elements" list and the
// "Calibrated thresholds" list; scope sidebar lookups to the CE item rows.
const ceItem = (name) => {
    const sidebar = document.querySelector('.rtv-sidebar');
    return within(sidebar).getAllByText(name).find(el => el.classList.contains('rtv-ce-name'));
};

// A fully-populated live analysis exercising tokens, chart, thresholds,
// triggered CEs and rule triggers.
const richAnalysis = {
    generated_text: 'hello world',
    labels: { Toxicity: {}, Bias: {} },
    triggered_ces: ['Toxicity'],
    thresholds_used: { Toxicity: { threshold: 0.7, patience: 3 } },
    tokens: [
        { token: 'hel', token_index: 0, triggered_ces: ['Toxicity'], probabilities: { Toxicity: 0.9, Bias: 0.1 } },
        { token: 'lo', token_index: 1, triggered_ces: [], probabilities: { Toxicity: 0.2, Bias: 0.3 } },
        { token: ' world', token_index: 2, triggered_ces: ['Bias'], probabilities: { Toxicity: 0.1, Bias: 0.8 } },
    ],
    rule_triggers: [
        {
            rule_name: 'RuleFired',
            fired: true,
            condition: 'all of required',
            groups: {
                required: { members: ['Toxicity'], hits: ['Toxicity'], quantifier: 'all', satisfied: true },
            },
            groups_unmet: [],
        },
        {
            rule_name: 'RuleMissed',
            fired: false,
            condition: 'all of required and 2 of solicited_action',
            groups: {
                required: { members: ['Toxicity'], hits: ['Toxicity'], quantifier: 'all', satisfied: true },
                solicited_action: { members: ['a', 'b', 'c'], hits: ['a'], quantifier: 2, satisfied: false },
                supporting: { members: ['x'], hits: [], quantifier: null, satisfied: true },
            },
            groups_unmet: ['solicited_action'],
        },
    ],
};

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    mocks.getClassifierDetails.mockResolvedValue({ data: { name: 'My Classifier' } });
    mocks.listSampleGroups.mockResolvedValue({ data: { groups: [] } });
    mocks.getSampleGroup.mockResolvedValue({ data: { samples: [] } });
    mocks.analyzeRealtime.mockResolvedValue({ data: { generated_text: 'reply' } });
    mocks.analyzeStored.mockResolvedValue({ data: { generated_text: 'stored reply' } });
    // Run in local mode by default: the backend signals fallback:'local', so the
    // viewer loads the model in-process and the chat UI renders immediately.
    mocks.startRealtimeSession.mockResolvedValue({ data: { fallback: 'local' } });
    mocks.getRealtimeSessionStatus.mockResolvedValue({ data: { status: 'ready' } });
    mocks.realtimeSessionKeepalive.mockResolvedValue({ data: { alive: true } });
    mocks.endRealtimeSession.mockResolvedValue({ data: { ok: true } });
    mocks.endRealtimeSessionUnload.mockResolvedValue({ data: { ok: true } });
    mocks.sessionAnalyzeStored.mockResolvedValue({ data: {} });
    mocks.sessionAnalyzeLive.mockResolvedValue({ data: {} });
});

describe('RealtimeViewer — initial render', () => {
    it('renders header, layout and empty live state', async () => {
        renderViewer();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByText('Realtime CE Monitor')).toBeInTheDocument();
        expect(screen.getByText(/Send a message to see per-token CE activations/i)).toBeInTheDocument();
        expect(screen.getByText('No analysis yet.')).toBeInTheDocument();
        // Classifier loaded → subtitle shows its name.
        expect(await screen.findByText(/Rule Set: My Classifier/)).toBeInTheDocument();
    });

    it('falls back to #id subtitle when classifier has no name', async () => {
        mocks.getClassifierDetails.mockResolvedValue({ data: {} });
        renderViewer();
        expect(await screen.findByText(/Rule Set: #42/)).toBeInTheDocument();
    });

    it('renders without subtitle when classifier fetch fails', async () => {
        mocks.getClassifierDetails.mockRejectedValue(new Error('nope'));
        renderViewer();
        await waitFor(() => expect(mocks.getClassifierDetails).toHaveBeenCalledWith('42'));
        expect(screen.queryByText(/Rule Set:/)).not.toBeInTheDocument();
    });

    it('rule set breadcrumb navigates to the rules page', () => {
        renderViewer();
        // Clicked synchronously before the details fetch resolves, so the
        // middle crumb still shows the fallback "Rule Set" label.
        fireEvent.click(screen.getByText('Rule Set'));
        expect(mocks.navigate).toHaveBeenCalledWith('/classifiers/42/rules');
    });
});

describe('RealtimeViewer — settings panel', () => {
    it('toggles the settings panel and edits the system prompt', () => {
        renderViewer();
        expect(screen.queryByText('System Prompt')).not.toBeInTheDocument();
        fireEvent.click(screen.getByTitle('Settings'));
        const prompt = screen.getByDisplayValue('You are a helpful assistant.');
        fireEvent.change(prompt, { target: { value: 'New prompt' } });
        expect(screen.getByDisplayValue('New prompt')).toBeInTheDocument();
        // Toggle closed again.
        fireEvent.click(screen.getByTitle('Settings'));
        expect(screen.queryByText('System Prompt')).not.toBeInTheDocument();
    });

    it('clamps max tokens to a minimum of 1', () => {
        renderViewer();
        fireEvent.click(screen.getByTitle('Settings'));
        const num = screen.getByDisplayValue('128');
        fireEvent.change(num, { target: { value: '0' } });
        expect(screen.getByDisplayValue('1')).toBeInTheDocument();
        fireEvent.change(num, { target: { value: 'abc' } });
        expect(screen.getByDisplayValue('1')).toBeInTheDocument();
        fireEvent.change(num, { target: { value: '256' } });
        expect(screen.getByDisplayValue('256')).toBeInTheDocument();
    });
});

describe('RealtimeViewer — live chat send flow', () => {
    it('does not send empty / whitespace-only input', async () => {
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: '   ' } });
        // Send button disabled for whitespace-only.
        const sendBtn = input.parentElement.querySelector('button');
        expect(sendBtn).toBeDisabled();
        fireEvent.click(sendBtn);
        expect(mocks.analyzeRealtime).not.toHaveBeenCalled();
    });

    it('sends a message and renders the assistant reply with analysis', async () => {
        mocks.analyzeRealtime.mockResolvedValue({ data: richAnalysis });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'hi there' } });
        fireEvent.keyDown(input, { key: 'Enter' });

        // User message echoed.
        expect(await screen.findByText('hi there')).toBeInTheDocument();
        // Analysis arrives → tokens render.
        await screen.findByText('hel');
        expect(screen.getByText('lo')).toBeInTheDocument();
        expect(screen.getByText(/world/)).toBeInTheDocument();

        // API called with expected payload (history undefined for first send).
        expect(mocks.analyzeRealtime).toHaveBeenCalledWith('42', expect.objectContaining({
            system_prompt: 'You are a helpful assistant.',
            user_message: 'hi there',
            history: undefined,
            max_new_tokens: 128,
        }));

        // Sidebar lists CEs derived from the analysis.
        const sidebar = document.querySelector('.rtv-sidebar');
        expect(ceItem('Toxicity')).toBeTruthy();
        expect(ceItem('Bias')).toBeTruthy();
        // Triggered badge for the triggered CE.
        expect(within(sidebar).getByText('TRIGGERED')).toBeInTheDocument();
        // Threshold now shown inline next to each CE.
        expect(within(sidebar).getByText(/value = threshold/)).toBeInTheDocument();
        expect(within(sidebar).getByText('0.70')).toBeInTheDocument();

        // Rule triggers sidebar section.
        expect(within(sidebar).getByText('RuleFired')).toBeInTheDocument();
        expect(within(sidebar).getByText('FIRED')).toBeInTheDocument();
        expect(within(sidebar).getByText('NOT FIRED')).toBeInTheDocument();
        // Per-group progress: hits vs. the condition's demand, per group.
        expect(within(sidebar).getByText(/1\/2 of solicited_action/)).toBeInTheDocument();
        // Unreferenced (supporting) groups are not listed as progress rows.
        expect(within(sidebar).queryByText(/of supporting/)).not.toBeInTheDocument();
    });

    it('passes prior history on the second send', async () => {
        mocks.analyzeRealtime.mockResolvedValue({ data: { generated_text: 'r1', labels: {}, tokens: [] } });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'first' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('first');

        fireEvent.change(input, { target: { value: 'second' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await waitFor(() => expect(mocks.analyzeRealtime).toHaveBeenCalledTimes(2));
        const secondCall = mocks.analyzeRealtime.mock.calls[1][1];
        expect(secondCall.history).toEqual([
            { role: 'user', content: 'first' },
            { role: 'assistant', content: 'r1' },
        ]);
    });

    it('renders an error message when analysis fails (with detail)', async () => {
        mocks.analyzeRealtime.mockRejectedValue({ response: { data: { detail: 'boom' } } });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('Error: boom')).toBeInTheDocument();
    });

    it('renders a generic error message when analysis fails without detail', async () => {
        mocks.analyzeRealtime.mockRejectedValue(new Error('network'));
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('Error: Analysis failed')).toBeInTheDocument();
    });

    it('Shift+Enter does not send', async () => {
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'keep typing' } });
        fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
        expect(mocks.analyzeRealtime).not.toHaveBeenCalled();
    });

    it('clears the conversation', async () => {
        mocks.analyzeRealtime.mockResolvedValue({ data: { generated_text: 'reply', labels: {}, tokens: [] } });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'hello' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('hello');

        fireEvent.click(screen.getByTitle('Clear conversation'));
        expect(screen.queryByText('hello')).not.toBeInTheDocument();
        expect(screen.getByText(/Send a message to see per-token/i)).toBeInTheDocument();
    });
});

describe('RealtimeViewer — CE selection & threshold defaults', () => {
    it('selecting/deselecting a CE toggles its active state and chart focus', async () => {
        mocks.analyzeRealtime.mockResolvedValue({ data: richAnalysis });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('hel');

        const toxItem = ceItem('Toxicity').closest('.rtv-ce-item');
        fireEvent.click(toxItem);
        expect(toxItem.className).toContain('active');
        // Chart title reflects the selected CE.
        expect(screen.getByText(/CE activation over tokens · Toxicity/)).toBeInTheDocument();
        // Click again to deselect.
        fireEvent.click(toxItem);
        expect(toxItem.className).not.toContain('active');
    });

    it('shows default thresholds label when calibration not used', async () => {
        mocks.analyzeRealtime.mockResolvedValue({
            data: { generated_text: 'x', labels: { CeA: {} }, tokens: [{ token: 'a', token_index: 0, triggered_ces: [], probabilities: { CeA: 0.1 } }] },
        });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('a');
        const sidebar = document.querySelector('.rtv-sidebar');
        // Threshold shown inline; defaults to 0.50 when calibration isn't used.
        expect(within(sidebar).getByText(/value = threshold/)).toBeInTheDocument();
        expect(within(sidebar).getByText('0.50')).toBeInTheDocument();
    });

    it('falls back to window display when no per-token data', async () => {
        mocks.analyzeRealtime.mockResolvedValue({
            data: {
                generated_text: 'fulltext',
                labels: { CeA: {} },
                tokens: [],
                windows: [
                    { window_index: 0, token_count: 2, text: 'win-one ', probabilities: { CeA: 0.9 }, window_triggered_ces: ['CeA'] },
                    { window_index: 1, token_count: 1, text: 'win-two', probabilities: { CeA: 0.1 }, window_triggered_ces: [] },
                ],
            },
        });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('win-one')).toBeInTheDocument();
        expect(screen.getByText('win-two')).toBeInTheDocument();
    });

    it('window display falls back to generated_text when no windows', async () => {
        mocks.analyzeRealtime.mockResolvedValue({
            data: { generated_text: 'plain-reply', labels: { CeA: {} }, tokens: [] },
        });
        renderViewer();
        const input = await screen.findByPlaceholderText('Type a message...');
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('plain-reply')).toBeInTheDocument();
    });
});

describe('RealtimeViewer — stored (test samples) mode', () => {
    it('shows loading then empty groups state', async () => {
        let resolveGroups;
        mocks.listSampleGroups.mockReturnValue(new Promise(r => { resolveGroups = r; }));
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        // While the promise is pending, groups === null → "Loading…".
        expect(screen.getByText('Loading…')).toBeInTheDocument();
        resolveGroups({ data: { groups: [] } });
        expect(await screen.findByText(/No stored conversations for this rule set yet/)).toBeInTheDocument();
    });

    it('sets empty groups when listSampleGroups rejects', async () => {
        mocks.listSampleGroups.mockRejectedValue(new Error('fail'));
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        expect(await screen.findByText(/No stored conversations/)).toBeInTheDocument();
    });

    it('lists groups, picks one, lists samples, picks a sample and analyzes it', async () => {
        mocks.listSampleGroups.mockResolvedValue({
            data: { groups: [
                { key: 'g1', label: 'Group One', count: 2 },
                { key: 'g2', label: 'Empty Group', count: 0 },
            ] },
        });
        mocks.getSampleGroup.mockResolvedValue({
            data: { samples: [
                {
                    index: 0,
                    messages: [{ role: 'user', content: 'the full user question text' }],
                    assistant_preview: 'preview A',
                    // Truncated preview is for the picker list only — the analysis
                    // pane must show the FULL message content, not this.
                    user_preview: 'the full user q…',
                },
            ] },
        });
        mocks.analyzeStored.mockResolvedValue({ data: richAnalysis });

        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');

        const groupBtn = await screen.findByText('Group One');
        // Empty group is rendered but disabled.
        const emptyBtn = screen.getByText('Empty Group').closest('button');
        expect(emptyBtn).toBeDisabled();

        // Conversation column starts with a hint.
        expect(screen.getByText('Pick a dataset first.')).toBeInTheDocument();

        fireEvent.click(groupBtn.closest('button'));
        const sampleBtn = await screen.findByText(/Conversation 1/);
        expect(mocks.getSampleGroup).toHaveBeenCalledWith('42', 'g1');

        fireEvent.click(sampleBtn.closest('button'));
        // Analysis renders the rich tokens.
        await screen.findByText('hel');
        expect(mocks.analyzeStored).toHaveBeenCalledWith('42', [{ role: 'user', content: 'the full user question text' }]);
        // FULL user message shown above the assistant analysis (not the
        // truncated 160-char preview used in the picker list).
        expect(screen.getByText('the full user question text')).toBeInTheDocument();
        expect(screen.queryByText('the full user q…')).not.toBeInTheDocument();
    });

    it('renders a multi-turn dialogue as ping-pong with a graph per assistant turn', async () => {
        mocks.listSampleGroups.mockResolvedValue({
            data: { groups: [{ key: 'g1', label: 'Group One', count: 1 }] },
        });
        mocks.getSampleGroup.mockResolvedValue({
            data: { samples: [{ index: 0, assistant_preview: 'preview A', user_preview: 'u', messages: [] }] },
        });
        // New backend shape: an ordered `turns` array, each assistant turn
        // carrying its own tokens + rule_triggers.
        mocks.analyzeStored.mockResolvedValue({
            data: {
                generated_text: null,
                labels: { Toxicity: {}, Bias: {} },
                thresholds_used: { Toxicity: { threshold: 0.7, patience: 1 }, Bias: { threshold: 0.5, patience: 1 } },
                triggered_ces: ['Toxicity'],
                rule_triggers: [{ rule_name: 'AggRule', fired: true }],
                turns: [
                    { role: 'user', content: 'first question text' },
                    {
                        role: 'assistant', content: 'first answer',
                        tokens: [
                            { token: 'A1', token_index: 0, triggered_ces: ['Toxicity'], probabilities: { Toxicity: 0.9, Bias: 0.1 } },
                            { token: 'A2', token_index: 1, triggered_ces: [], probabilities: { Toxicity: 0.2, Bias: 0.2 } },
                        ],
                        rule_triggers: [{ rule_name: 'RuleTurnOne', fired: true }],
                    },
                    { role: 'user', content: 'second question text' },
                    {
                        role: 'assistant', content: 'second answer',
                        tokens: [
                            { token: 'B1', token_index: 0, triggered_ces: ['Bias'], probabilities: { Toxicity: 0.1, Bias: 0.8 } },
                            { token: 'B2', token_index: 1, triggered_ces: [], probabilities: { Toxicity: 0.1, Bias: 0.1 } },
                        ],
                        rule_triggers: [{ rule_name: 'RuleTurnTwo', fired: false, condition: 'all of required', groups: { required: { members: ['a'], hits: [], quantifier: 'all', satisfied: false } }, groups_unmet: ['required'] }],
                    },
                ],
            },
        });

        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        fireEvent.click((await screen.findByText('Group One')).closest('button'));
        fireEvent.click((await screen.findByText(/Conversation 1/)).closest('button'));

        // Both user turns render in order (ping-pong), not concatenated.
        expect(await screen.findByText('first question text')).toBeInTheDocument();
        expect(screen.getByText('second question text')).toBeInTheDocument();
        // Each assistant turn renders its own tokens...
        expect(screen.getByText('A1')).toBeInTheDocument();
        expect(screen.getByText('B1')).toBeInTheDocument();
        // ...and its own rule strip (per-turn, not just an aggregate).
        expect(screen.getByText(/RuleTurnOne/)).toBeInTheDocument();
        expect(screen.getByText(/RuleTurnTwo/)).toBeInTheDocument();
    });

    it('shows "No conversations." when a picked group has no samples', async () => {
        mocks.listSampleGroups.mockResolvedValue({ data: { groups: [{ key: 'g1', label: 'G1', count: 3 }] } });
        mocks.getSampleGroup.mockResolvedValue({ data: { samples: [] } });
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        fireEvent.click((await screen.findByText('G1')).closest('button'));
        expect(await screen.findByText('No conversations.')).toBeInTheDocument();
    });

    it('handles getSampleGroup rejection by clearing samples', async () => {
        mocks.listSampleGroups.mockResolvedValue({ data: { groups: [{ key: 'g1', label: 'G1', count: 3 }] } });
        mocks.getSampleGroup.mockRejectedValue(new Error('boom'));
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        fireEvent.click((await screen.findByText('G1')).closest('button'));
        expect(await screen.findByText('No conversations.')).toBeInTheDocument();
    });

    it('renders an error in stored analysis when analyzeStored fails', async () => {
        mocks.listSampleGroups.mockResolvedValue({ data: { groups: [{ key: 'g1', label: 'G1', count: 1 }] } });
        mocks.getSampleGroup.mockResolvedValue({
            data: { samples: [{ index: 0, messages: [], assistant_preview: 'AP' }] },
        });
        mocks.analyzeStored.mockRejectedValue({ response: { data: { detail: 'stored boom' } } });
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        fireEvent.click((await screen.findByText('G1')).closest('button'));
        fireEvent.click((await screen.findByText(/Conversation 1/)).closest('button'));
        expect(await screen.findByText('Error: stored boom')).toBeInTheDocument();
    });

    it('shows the stored empty prompt before any sample is analyzed', async () => {
        mocks.listSampleGroups.mockResolvedValue({ data: { groups: [] } });
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        expect(await screen.findByText(/Pick a CE and one of its dialogues to analyze/)).toBeInTheDocument();
        // Settings/clear buttons hidden in stored mode.
        expect(screen.queryByTitle('Settings')).not.toBeInTheDocument();
        expect(screen.queryByTitle('Clear conversation')).not.toBeInTheDocument();
    });

    it('does not re-fetch groups when toggling back to stored mode', async () => {
        mocks.listSampleGroups.mockResolvedValue({ data: { groups: [] } });
        renderViewer();
        // Wait for the session to settle to local mode before interacting, so
        // stored analysis goes through the local path (not the warm-session one).
        await screen.findByPlaceholderText('Type a message...');
        clickMode('Test Samples');
        await screen.findByText(/No stored conversations/);
        expect(mocks.listSampleGroups).toHaveBeenCalledTimes(1);
        clickMode('Live Chat');
        clickMode('Test Samples');
        // sampleGroups already loaded (non-null) → effect short-circuits.
        expect(mocks.listSampleGroups).toHaveBeenCalledTimes(1);
    });
});

describe('RealtimeViewer — warm-session provider labeling', () => {
    it('labels a remote GPU worker session as "remote GPU", never "cluster"', async () => {
        mocks.startRealtimeSession.mockResolvedValue({
            data: { ok: true, mode: 'remote_worker', session_id: 's1', status: 'loading' },
        });
        // Keep it in the loading state so the startup banner stays visible.
        mocks.getRealtimeSessionStatus.mockResolvedValue({ data: { status: 'loading' } });
        renderViewer();
        // Banner reflects the remote GPU (heading + sub-text), cluster wording gone.
        expect((await screen.findAllByText(/remote GPU/i)).length).toBeGreaterThan(0);
        expect(screen.queryByText(/cluster/i)).not.toBeInTheDocument();
    });

    it('labels a SLURM session as "cluster GPU"', async () => {
        // No `mode` and no `fallback` → the cluster tier.
        mocks.startRealtimeSession.mockResolvedValue({
            data: { ok: true, session_id: 'c1', status: 'queued' },
        });
        mocks.getRealtimeSessionStatus.mockResolvedValue({ data: { status: 'queued' } });
        renderViewer();
        expect((await screen.findAllByText(/cluster GPU/i)).length).toBeGreaterThan(0);
    });

    it('reads the backend `status` field so a loading session reaches ready (no infinite wait)', async () => {
        mocks.startRealtimeSession.mockResolvedValue({
            data: { ok: true, mode: 'remote_worker', session_id: 's2', status: 'loading' },
        });
        // The poll returns {status: 'ready'} — the viewer must read `.status`
        // (not `.state`) to transition, or it would hang forever.
        mocks.getRealtimeSessionStatus.mockResolvedValue({ data: { status: 'ready' } });
        renderViewer();
        // Once ready, the live chat input is enabled with its normal placeholder.
        expect(await screen.findByPlaceholderText('Type a message...', {}, { timeout: 4000 }))
            .toBeInTheDocument();
    });
});
