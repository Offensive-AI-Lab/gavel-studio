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

import RuleDefaultsStep from '../../../src/components/RuleDefaults/RuleDefaultsStep';
import { useTaskTray } from '../../../src/contexts/TaskTrayContext';
import { deriveScenario, generateRuleDefaults, getRuleDefaultsStatus } from '../../../src/api';

// TaskTrayProvider only provides context — it does NOT render the chip UI.
// This probe surfaces the in-memory task list so tests can assert on the
// background job's lifecycle (title / subtitle / status).
function TrayProbe() {
    const { tasks } = useTaskTray();
    return (
        <ul data-testid="tray-probe">
            {tasks.map((t) => (
                <li key={t.id} data-status={t.status}>
                    {t.title} :: {t.subtitle} :: {t.status}
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
        expect(screen.queryByPlaceholderText(/Describe the misuse/i)).toBeNull();

        // Static info box renders regardless of derive state.
        expect(screen.getByText(/Test Set & calibration/i)).toBeInTheDocument();

        // resolve so the effect's setState doesn't fire after unmount warnings.
        await act(async () => { resolveDerive({ data: { scenario: '' } }); });
    });

    it('prefills the textarea with the derived scenario on success', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: 'Prefilled misuse text' } });

        renderStep({ ruleId: 42 });

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
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

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
        expect(textarea).toHaveValue('');
        // Generate button disabled because the scenario is empty.
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();
    });

    it('handles a null data object from derive without crashing', async () => {
        deriveScenario.mockResolvedValue({});

        renderStep();

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
        expect(textarea).toHaveValue('');
    });

    it('shows the soft warning when derive rejects, but still renders an editable textarea', async () => {
        deriveScenario.mockRejectedValue(new Error('boom'));

        renderStep();

        expect(await screen.findByText(/Could not auto-write a scenario/i)).toBeInTheDocument();
        // Textarea is still present and empty so the user can type their own.
        const textarea = screen.getByPlaceholderText(/Describe the misuse/i);
        expect(textarea).toHaveValue('');
        // Loading row gone.
        expect(screen.queryByText(/Analyzing the rule/i)).toBeNull();
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

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
        const btn = screen.getByRole('button', { name: /Generate test/i });
        expect(btn).toBeDisabled();

        fireEvent.change(textarea, { target: { value: 'catch jailbreaks' } });
        expect(textarea).toHaveValue('catch jailbreaks');
        expect(btn).not.toBeDisabled();
    });

    it('keeps the button disabled when the scenario is only whitespace', async () => {
        deriveScenario.mockResolvedValue({ data: { scenario: '' } });
        renderStep();

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
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

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
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

        const textarea = await screen.findByPlaceholderText(/Describe the misuse/i);
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
        expect(probe).toHaveTextContent('Rule ready — find it in Drafts.');
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
