// Behavior tests for RuleDefaultsStep — the final wizard step that derives a
// misuse scenario for a fresh rule, lets the user edit it, and kicks off
// background generation of the rule's test + calibration sets via the task tray.
//
// Strategy:
//   * Mock '../../api' so deriveScenario / generateRuleDefaults /
//     getRuleDefaultsStatus never hit the network and we control their results.
//   * Use the REAL TaskTrayProvider + REAL runInTray so the background job
//     (generate + poll loop) actually runs — driven with fake timers since the
//     poll uses sleep(2500).
//   * Stub sweetalert2 defensively (component doesn't use it, but children might).

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { TaskTrayProvider } from '../../../src/contexts/TaskTrayContext';

// --- API mock. Cover the three exports this file uses. Each test overrides
// the resolved values it cares about via mockResolvedValueOnce / mockImplementation.
vi.mock('../../../src/api', () => ({
    deriveScenario: vi.fn(() => Promise.resolve({ data: { scenario: '' } })),
    generateRuleDefaults: vi.fn(() => Promise.resolve({ data: {} })),
    getRuleDefaultsStatus: vi.fn(() => Promise.resolve({ data: { state: 'ready', datasets: [] } })),
}));

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));

import Swal from 'sweetalert2';
import RuleDefaultsStep from '../../../src/components/RuleDefaults/RuleDefaultsStep';
import { useTaskTray } from '../../../src/contexts/TaskTrayContext';
import { deriveScenario, generateRuleDefaults, getRuleDefaultsStatus } from '../../../src/api';

// TaskTrayProvider only provides context — it does NOT render the chip UI.
// This probe surfaces the in-memory task list so tests can assert on the
// background job's lifecycle (title / subtitle / status). It also mirrors the
// real TaskTray's click affordance: a chip with an openHandler gets an "open"
// button that calls tray.open(id) — how the error chip's dialog is reached.
function TrayProbe() {
    const { tasks, open } = useTaskTray();
    return (
        <ul data-testid="tray-probe">
            {tasks.map((t) => (
                <li key={t.id} data-status={t.status} data-has-open={t.openHandler ? 'yes' : 'no'}>
                    {t.title} :: {t.subtitle} :: {t.status}
                    {t.openHandler && (
                        <button data-testid="chip-open" onClick={() => open(t.id)}>open</button>
                    )}
                </li>
            ))}
        </ul>
    );
}

const renderStep = (props = {}) => {
    const onDone = props.onDone || vi.fn();
    const finalize = props.finalize || vi.fn(() => Promise.resolve());
    const utils = render(
        <TaskTrayProvider>
            <RuleDefaultsStep ruleId={props.ruleId ?? 7} onDone={onDone} finalize={finalize} />
            <TrayProbe />
        </TaskTrayProvider>,
    );
    return { ...utils, onDone, finalize };
};

beforeEach(() => {
    vi.clearAllMocks();
    // default benign resolutions
    deriveScenario.mockResolvedValue({ data: { scenario: '' } });
    generateRuleDefaults.mockResolvedValue({ data: {} });
    getRuleDefaultsStatus.mockResolvedValue({ data: { state: 'ready', datasets: [] } });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('RuleDefaultsStep — initial render & derive', () => {
    it('shows the loading row while deriving the scenario', async () => {
        // deriveScenario never resolves during this assertion window.
        let resolveDerive;
        deriveScenario.mockReturnValue(new Promise((r) => { resolveDerive = r; }));

        renderStep();

        // Loading text is visible; textarea is NOT rendered yet.
        expect(screen.getByText(/Analyzing the rule/i)).toBeInTheDocument();
        expect(screen.queryByPlaceholderText(/Example: A user poses as/i)).toBeNull();

        // Static info box renders regardless of derive state.
        expect(screen.getByText(/Test Set & calibration/i)).toBeInTheDocument();

        // resolve so the effect's setState doesn't fire after unmount warnings.
        await act(async () => { resolveDerive({ data: { scenario: '' } }); });
    });

    it('prefills the textarea with the derived scenario on success', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: 'Prefilled misuse text' } });

        renderStep({ ruleId: 42 });

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        expect(textarea).toHaveValue('Prefilled misuse text');
        // Called with the ruleId.
        expect(deriveScenario).toHaveBeenCalledWith(42);
        // No loading row anymore.
        expect(screen.queryByText(/Analyzing the rule/i)).toBeNull();
        // No error / warning.
        expect(screen.queryByText(/Could not auto-write/i)).toBeNull();
    });

    it('falls back to an empty textarea when derive returns no scenario', async () => {
        deriveScenario.mockResolvedValue({ data: {} });

        renderStep();

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        expect(textarea).toHaveValue('');
        // Generate button disabled because the scenario is empty.
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();
    });

    it('handles a null data object from derive without crashing', async () => {
        deriveScenario.mockResolvedValue({});

        renderStep();

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        expect(textarea).toHaveValue('');
    });

    it('shows the soft warning when derive rejects, but still renders an editable textarea', async () => {
        deriveScenario.mockRejectedValue(new Error('boom'));

        renderStep();

        expect(await screen.findByText(/Could not auto-write a scenario/i)).toBeInTheDocument();
        // Textarea is still present and empty so the user can type their own.
        const textarea = screen.getByPlaceholderText(/Example: A user poses as/i);
        expect(textarea).toHaveValue('');
        // Loading row gone.
        expect(screen.queryByText(/Analyzing the rule/i)).toBeNull();
    });

    it('names the missing OpenAI key when derive fails with the backend 503 key error', async () => {
        deriveScenario.mockRejectedValue({
            response: {
                status: 503,
                data: { detail: 'OPENAI_API_KEY is not configured. Add it to backend/.env and restart the backend.' },
            },
        });

        renderStep();

        const warning = await screen.findByText(/Could not auto-write a scenario/i);
        expect(warning).toHaveTextContent(/OpenAI key/i);
        expect(warning).toHaveTextContent(/OPENAI_API_KEY/);
        expect(warning).toHaveTextContent(/backend\/\.env/);
        // Manual-typing path stays available.
        expect(screen.getByPlaceholderText(/Example: A user poses as/i)).toHaveValue('');
    });

    it('shows the key guidance when a non-503 error detail mentions OPENAI_API_KEY', async () => {
        // litellm auth failures surface as 500 with the key name in the detail.
        deriveScenario.mockRejectedValue({
            response: {
                status: 500,
                data: { detail: 'Scenario derivation failed: AuthenticationError - set OPENAI_API_KEY' },
            },
        });

        renderStep();

        const warning = await screen.findByText(/Could not auto-write a scenario/i);
        expect(warning).toHaveTextContent(/OpenAI key/i);
    });

    it('says the backend is unreachable when derive fails without a response', async () => {
        // axios network error: no err.response at all.
        deriveScenario.mockRejectedValue(new Error('Network Error'));

        renderStep();

        const warning = await screen.findByText(/Could not auto-write a scenario/i);
        expect(warning).toHaveTextContent(/backend is unreachable/i);
        expect(screen.getByPlaceholderText(/Example: A user poses as/i)).toHaveValue('');
    });

    it('includes the backend detail for other derive failures', async () => {
        deriveScenario.mockRejectedValue({
            response: { status: 500, data: { detail: 'scenario_deriver.md not found' } },
        });

        renderStep();

        const warning = await screen.findByText(/Could not auto-write a scenario/i);
        expect(warning).toHaveTextContent(/scenario_deriver\.md not found/);
    });

    it('shows a soft warning instead of a silent empty box when derive returns an empty scenario', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: '   ' } });

        renderStep();

        expect(await screen.findByText(/returned an empty scenario/i)).toBeInTheDocument();
        // Textarea still editable; button stays disabled on whitespace-only.
        expect(screen.getByPlaceholderText(/Example: A user poses as/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Generate test/i })).toBeDisabled();
    });

    it('re-derives when ruleId changes', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: 'first' } });
        const onDone = vi.fn();
        const { rerender } = render(
            <TaskTrayProvider>
                <RuleDefaultsStep ruleId={1} onDone={onDone} />
            </TaskTrayProvider>,
        );
        await screen.findByDisplayValue('first');
        expect(deriveScenario).toHaveBeenCalledWith(1);

        deriveScenario.mockResolvedValue({ data: { scenario: 'second' } });
        rerender(
            <TaskTrayProvider>
                <RuleDefaultsStep ruleId={2} onDone={onDone} />
            </TaskTrayProvider>,
        );
        await screen.findByDisplayValue('second');
        expect(deriveScenario).toHaveBeenCalledWith(2);
        expect(deriveScenario).toHaveBeenCalledTimes(2);
    });
});

describe('RuleDefaultsStep — typing & validation', () => {
    it('enables the Generate button once a non-empty scenario is typed', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: '' } });
        renderStep();

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();

        fireEvent.change(textarea, { target: { value: 'catch jailbreaks' } });
        expect(textarea).toHaveValue('catch jailbreaks');
        expect(btn).not.toBeDisabled();
    });

    it('keeps the button disabled when the scenario is only whitespace', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: '' } });
        renderStep();

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        fireEvent.change(textarea, { target: { value: '    ' } });
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();
    });
});

describe('RuleDefaultsStep — generate flow (background tray job)', () => {
    const flush = async () => { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); };

    it('starts generation, calls onDone immediately, and leaves a running tray task', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: '  leading/trailing  ' } });
        // Keep the poll pending so the task stays in the running state.
        getRuleDefaultsStatus.mockReturnValue(new Promise(() => {}));

        const { onDone } = renderStep({ ruleId: 99 });

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        expect(textarea).toHaveValue('  leading/trailing  ');

        const btn = screen.getByRole('button', { name: /Generate test/i });
        await act(async () => { fireEvent.click(btn); });

        // onDone fired synchronously (non-blocking design).
        expect(onDone).toHaveBeenCalledTimes(1);

        // generateRuleDefaults kicked off with the TRIMMED scenario.
        await flush();
        expect(generateRuleDefaults).toHaveBeenCalledWith(99, 'leading/trailing');
        expect(generateRuleDefaults).toHaveBeenCalledTimes(1);

        // A running task is present in the tray.
        const probe = screen.getByTestId('tray-probe');
        expect(probe).toHaveTextContent('Generating test & calibration set');
        expect(probe).toHaveTextContent('running');
    });

    it('does nothing when clicking the disabled button with an empty scenario', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: 'something' } });
        const { onDone } = renderStep();

        const textarea = await screen.findByPlaceholderText(/Example: A user poses as/i);
        fireEvent.change(textarea, { target: { value: '' } });
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();

        fireEvent.click(btn);
        expect(onDone).not.toHaveBeenCalled();
        expect(generateRuleDefaults).not.toHaveBeenCalled();
        // No background task was created.
        expect(screen.getByTestId('tray-probe')).toBeEmptyDOMElement();
    });

    it('polls until state is ready, updating the subtitle, then marks the task success', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus
            .mockResolvedValueOnce({ data: { state: 'generating', datasets: [{ status: 'ready' }, { status: 'pending' }, { status: 'pending' }] } })
            .mockResolvedValueOnce({ data: { state: 'ready', datasets: [] } });

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        // First sleep(2500) → first poll: 1/3 ready.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(getRuleDefaultsStatus).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('1/3 sets ready');

        // Second sleep → second poll returns ready → task success.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(getRuleDefaultsStatus).toHaveBeenCalledTimes(2);
        const probe = screen.getByTestId('tray-probe');
        expect(probe).toHaveTextContent('Rule ready. Find it in Your Library → Rules.');
        expect(probe).toHaveTextContent('success');
    });

    it('calls finalize() to REVEAL the rule only AFTER the sets report ready', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus
            .mockResolvedValueOnce({ data: { state: 'generating', datasets: [] } })
            .mockResolvedValueOnce({ data: { state: 'ready', datasets: [] } });
        const finalize = vi.fn(() => Promise.resolve());

        renderStep({ ruleId: 5, finalize });
        await act(async () => { await Promise.resolve(); });
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        // First poll: still generating → the rule must NOT be revealed yet.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(finalize).not.toHaveBeenCalled();

        // Second poll: ready → finalize reveals the rule, then task succeeds.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(finalize).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('success');
    });

    it('marks the task error (rule stays hidden) when finalize rejects', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus.mockResolvedValue({ data: { state: 'ready', datasets: [] } });
        const finalize = vi.fn(() => Promise.reject(new Error('reveal failed')));

        renderStep({ ruleId: 5, finalize });
        await act(async () => { await Promise.resolve(); });
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        // First poll returns ready → finalize() runs and rejects → error chip.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        await act(async () => { await Promise.resolve(); await Promise.resolve(); });
        expect(finalize).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('error');
    });

    it('counts ready datasets as 0/3 when the status payload omits datasets', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus
            .mockResolvedValueOnce({ data: { state: 'generating' } }) // no datasets → defaults to []
            .mockResolvedValueOnce({ data: { state: 'ready' } });

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('0/3 sets ready');

        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('success');
    });

    it('flips the task to error when status reports state === "error"', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus.mockResolvedValue({ data: { state: 'error', datasets: [] } });

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });

        const probe = screen.getByTestId('tray-probe');
        expect(probe).toHaveTextContent('Generation failed for one or more sets.');
        expect(probe).toHaveTextContent('error');
    });

    it('surfaces the backend-reported failure reason on the error chip and mentions Drafts', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        // Backend status now carries the per-bucket reason in `error`.
        getRuleDefaultsStatus.mockResolvedValue({
            data: { state: 'error', error: 'negative: LLM quota exceeded', datasets: [] },
        });

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });

        const probe = screen.getByTestId('tray-probe');
        // The WHY, not just "failed".
        expect(probe).toHaveTextContent('negative: LLM quota exceeded');
        // And where the rule went (it was revealed, not stranded).
        expect(probe).toHaveTextContent(/saved to Your Library/i);
        expect(probe).toHaveTextContent('error');
        // The failed chip is clickable (has an openHandler → real tray makes it a button).
        expect(probe.querySelector('li')).toHaveAttribute('data-has-open', 'yes');
    });

    it('clicking the failed chip and confirming the dialog retries the generation', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus
            .mockResolvedValueOnce({ data: { state: 'error', error: 'positive: boom', datasets: [] } })
            .mockResolvedValue({ data: { state: 'ready', datasets: [] } });
        // The confirm dialog (Swal under the hood) resolves as CONFIRMED once.
        Swal.fire.mockResolvedValueOnce({ isConfirmed: true });
        const finalize = vi.fn(() => Promise.resolve());

        renderStep({ ruleId: 5, finalize });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });

        // First poll → error chip with the reason + an open affordance.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('error');
        expect(generateRuleDefaults).toHaveBeenCalledTimes(1);

        // Click the chip → dialog confirmed → old chip closed, job re-started.
        await act(async () => {
            fireEvent.click(screen.getByTestId('chip-open'));
            await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
        });
        expect(generateRuleDefaults).toHaveBeenCalledTimes(2);
        expect(generateRuleDefaults).toHaveBeenLastCalledWith(5, 'scenario');
        // The retried job is running again (the errored chip was closed).
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('running');
        expect(screen.getByTestId('tray-probe')).not.toHaveTextContent('error');

        // Retry's poll reports ready → finalize runs → success chip.
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
        expect(finalize).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('success');
    });

    it('leaves the error chip in place when the retry dialog is declined', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        getRuleDefaultsStatus.mockResolvedValue({ data: { state: 'error', error: 'positive: boom', datasets: [] } });
        // Default Swal mock resolves { isConfirmed: false } → decline.

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });
        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        await act(async () => { await Promise.resolve(); });
        await act(async () => { await vi.advanceTimersByTimeAsync(2500); });

        await act(async () => {
            fireEvent.click(screen.getByTestId('chip-open'));
            await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
        });
        // No retry kicked off; the failed chip is still there for later.
        expect(generateRuleDefaults).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('tray-probe')).toHaveTextContent('error');
    });

    it('flips the task to error (with backend detail) when generateRuleDefaults rejects', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'scenario' } });
        generateRuleDefaults.mockRejectedValue({ response: { data: { detail: 'kickoff exploded' } } });

        renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        // Let the rejection propagate through runInTray's catch.
        await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); });

        const probe = screen.getByTestId('tray-probe');
        expect(probe).toHaveTextContent('kickoff exploded');
        expect(probe).toHaveTextContent('error');
        // Poll never happened because kickoff failed.
        expect(getRuleDefaultsStatus).not.toHaveBeenCalled();
    });

    it('does not show the inline "Write a scenario first." error on a valid start', async () => {
        vi.useFakeTimers();
        deriveScenario.mockResolvedValue({ data: { scenario: 'valid scenario' } });
        getRuleDefaultsStatus.mockReturnValue(new Promise(() => {}));

        const { onDone } = renderStep({ ruleId: 5 });
        await act(async () => { await Promise.resolve(); });

        await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Generate test/i })); });
        expect(screen.queryByText('Write a scenario first.')).toBeNull();
        expect(onDone).toHaveBeenCalledTimes(1);
    });
});

describe('RuleDefaultsStep — scenario auto-write shows the REAL reason', () => {
    // #2B: a missing key is only one way this fails. An invalid key, a rate
    // limit, a provider outage and a timeout all surface as the backend's
    // detail — the message must carry it, not flatten everything into
    // "could not auto-write a scenario".
    it.each([
        ['invalid key', 502, 'Scenario auto-write failed: AuthenticationError: Incorrect API key provided.'],
        ['rate limit', 502, 'Scenario auto-write failed: RateLimitError: quota exceeded.'],
        ['timeout', 502, 'Scenario auto-write failed: Request timed out.'],
    ])('surfaces a %s failure verbatim', async (_label, status, detail) => {
        deriveScenario.mockRejectedValue({ response: { status, data: { detail } } });
        renderStep();
        expect(await screen.findByText(new RegExp(detail.slice(0, 40).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')))
            .toBeInTheDocument();
    });

    it('is honest when the backend gives no reason at all', async () => {
        deriveScenario.mockRejectedValue({ response: { status: 500, data: {} } });
        renderStep();
        expect(await screen.findByText(/returned 500 with no reason/i)).toBeInTheDocument();
    });
});
