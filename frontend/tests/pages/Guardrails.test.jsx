// Behavior tests for the Rule Sets page (pages/Guardrails.jsx).
//
// Focus: the "Cancel training" action on a card whose run is in progress
// (issue #20). The page lists rule sets, polls training-status for the ones
// that are training, and renders a training banner per card — the cancel
// button lives in that banner.
//
// We mock the network (../api), the confirm-dialog helpers, sweetalert2 and
// the Sidebar (which Layout renders and which has its own fetches).

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// ---- navigate spy ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// ---- API mock: every export Guardrails (and children) touch ----
const ok = (extra = {}) => Promise.resolve({ data: extra });
vi.mock('../../src/api', () => ({
    default: { get: vi.fn(() => ok()), post: vi.fn(() => ok()), delete: vi.fn(() => ok()), put: vi.fn(() => ok()) },
    getUserGuardrails: vi.fn(() => ok({ classifiers: [] })),
    createGuardrail: vi.fn(() => ok({ classifier: { classifier_id: 1 } })),
    cloneClassifierToModel: vi.fn(() => ok({ classifier: { classifier_id: 2 } })),
    getUserModels: vi.fn(() => ok({ models: [] })),
    deleteClassifier: vi.fn(() => ok()),
    getTrainingStatus: vi.fn(() => ok({ classifier_id: 5, status: 'training', is_training: true })),
    cancelTraining: vi.fn(() => ok({ success: true, status: 'needs_retraining' })),
    startImport: vi.fn(() => ok({ job_id: 1 })),
    getBundleJob: vi.fn(() => ok({ status: 'running' })),
    getGuardrailFolders: vi.fn(() => ok({ folders: [] })),
    createGuardrailFolder: vi.fn(() => ok()),
    renameGuardrailFolder: vi.fn(() => ok()),
    deleteGuardrailFolder: vi.fn(() => ok()),
    assignGuardrailFolder: vi.fn(() => ok()),
    getRuleSetBookmarks: vi.fn(() => ok({ bookmarks: [] })),
    forkPublicRuleSet: vi.fn(() => ok({ classifier: { classifier_id: 3 } })),
    getPublicRuleSets: vi.fn(() => ok({ rule_sets: [] })),
    createModel: vi.fn(() => ok()),
}));

// ---- confirm dialog helpers, controllable per test ----
const mockConfirm = vi.fn(() => Promise.resolve(true));
const mockAlert = vi.fn(() => Promise.resolve());
const mockLoading = vi.fn(() => vi.fn());
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showConfirmDialog: (...a) => mockConfirm(...a),
    showAlertDialog: (...a) => mockAlert(...a),
    showLoadingDialog: (...a) => mockLoading(...a),
}));

// ---- sweetalert2 (import-started toast) ----
vi.mock('sweetalert2', () => ({ default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })), close: vi.fn() } }));

// ---- Sidebar stub (Layout renders it; it has its own fetches) ----
vi.mock('../../src/components/Sidebar/Sidebar', () => ({ default: () => <aside data-testid="sidebar-stub" /> }));

import Guardrails from '../../src/pages/Guardrails';
import * as api from '../../src/api';

const setUser = () => {
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderPage = () =>
    render(
        <TutorialProvider>
            <MemoryRouter initialEntries={['/classifiers']}>
                <Guardrails />
            </MemoryRouter>
        </TutorialProvider>,
    );

const guardrailFixture = (over = {}) => ({
    classifier_id: 5,
    name: 'Rule Set A',
    model_id: 1,
    model_name: 'M',
    status: 'training',
    training_phase_detail: 'Training RNN — Epoch 3 of 10',
    folder_id: null,
    ...over,
});

// Render with one rule set mid-training and wait for its banner.
const renderTraining = async (over = {}) => {
    api.getUserGuardrails.mockResolvedValue({ data: { classifiers: [guardrailFixture(over)] } });
    renderPage();
    return screen.findByText('Cancel training');
};

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    api.getUserGuardrails.mockResolvedValue({ data: { classifiers: [] } });
    api.getGuardrailFolders.mockResolvedValue({ data: { folders: [] } });
    api.getUserModels.mockResolvedValue({ data: { models: [] } });
    api.cancelTraining.mockResolvedValue({ data: { success: true, status: 'needs_retraining' } });
    mockConfirm.mockResolvedValue(true);
    mockAlert.mockResolvedValue();
});

afterEach(() => {
    sessionStorage.clear();
});

describe('Guardrails — cancel training visibility', () => {
    it('shows the cancel action on a card whose run is in progress', async () => {
        await renderTraining();
        expect(screen.getByText('Training RNN — Epoch 3 of 10')).toBeInTheDocument();
        expect(screen.getByTitle('Stop this training run')).toBeInTheDocument();
    });

    it.each(['untrained', 'active', 'needs_retraining', 'error'])(
        'does not show the cancel action when the status is %s',
        async (status) => {
            api.getUserGuardrails.mockResolvedValue({
                data: { classifiers: [guardrailFixture({ status, training_phase_detail: null })] },
            });
            renderPage();
            expect(await screen.findByText('Rule Set A')).toBeInTheDocument();
            expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
        },
    );
});

describe('Guardrails — cancel training flow', () => {
    it('confirms first, then cancels and refetches', async () => {
        const btn = await renderTraining();
        expect(api.getUserGuardrails).toHaveBeenCalledTimes(1);
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Cancel training?', variant: 'danger' }),
        ));
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledWith(5));
        // Refetched so the card shows its new state.
        await waitFor(() => expect(api.getUserGuardrails).toHaveBeenCalledTimes(2));
        expect(mockAlert).not.toHaveBeenCalled();
    });

    it('does not cancel when the confirm dialog is dismissed', async () => {
        mockConfirm.mockResolvedValue(false);
        const btn = await renderTraining();
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalled());
        expect(api.cancelTraining).not.toHaveBeenCalled();
        expect(api.getUserGuardrails).toHaveBeenCalledTimes(1);
    });

    it('treats a 409 as "already over": refetches, no error dialog', async () => {
        api.cancelTraining.mockRejectedValue({ response: { status: 409, data: { detail: 'Not training' } } });
        const btn = await renderTraining();
        fireEvent.click(btn);
        await waitFor(() => expect(api.getUserGuardrails).toHaveBeenCalledTimes(2));
        expect(mockAlert).not.toHaveBeenCalled();
    });

    it('shows an error dialog when the cancel fails for any other reason', async () => {
        api.cancelTraining.mockRejectedValue({ response: { status: 500, data: { detail: 'Boom' } } });
        const btn = await renderTraining();
        fireEvent.click(btn);
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Error', message: 'Boom', variant: 'error' }),
        ));
        // Failed cancel → no refetch; the run is still going.
        expect(api.getUserGuardrails).toHaveBeenCalledTimes(1);
    });

    it('locks the button while the cancel is in flight (no double submit)', async () => {
        let resolveCancel;
        api.cancelTraining.mockReturnValue(new Promise((res) => { resolveCancel = res; }));
        const btn = await renderTraining();
        fireEvent.click(btn);
        expect(await screen.findByText('Cancelling...')).toBeInTheDocument();
        fireEvent.click(screen.getByTitle('Stop this training run'));
        expect(api.cancelTraining).toHaveBeenCalledTimes(1);
        resolveCancel({ data: { success: true, status: 'needs_retraining' } });
        await waitFor(() => expect(api.getUserGuardrails).toHaveBeenCalledTimes(2));
    });

    it('two clicks in the same tick open ONE confirm dialog', async () => {
        // The guard has to be set before the dialog is shown: state updates are
        // async, so a second click that lands in the same tick would otherwise
        // still see "no cancel in flight" and stack a second dialog.
        let resolveConfirm;
        mockConfirm.mockReturnValue(new Promise((res) => { resolveConfirm = res; }));
        const btn = await renderTraining();
        fireEvent.click(btn);
        fireEvent.click(btn);
        expect(mockConfirm).toHaveBeenCalledTimes(1);
        resolveConfirm(true);
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledTimes(1));
    });

    it('re-arms the button when the confirm is declined', async () => {
        mockConfirm.mockResolvedValue(false);
        const btn = await renderTraining();
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledTimes(1));
        // Not stuck on "Cancelling..." — the user can change their mind.
        await waitFor(() => expect(screen.getByTitle('Stop this training run')).not.toBeDisabled());
        mockConfirm.mockResolvedValue(true);
        fireEvent.click(screen.getByTitle('Stop this training run'));
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledTimes(1));
    });
});

describe('Guardrails — a poll that was already in flight when the cancel landed', () => {
    it('is dropped instead of putting the card back into "Training..."', async () => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
        try {
            // The 5s poll answers slowly and reports the run as still training —
            // it left before the cancel did. Applying it would repaint the card.
            let answerPoll;
            api.getTrainingStatus.mockReturnValue(new Promise((res) => { answerPoll = res; }));
            const btn = await renderTraining();

            await vi.advanceTimersByTimeAsync(5000);      // poll fires, hangs
            expect(api.getTrainingStatus).toHaveBeenCalledTimes(1);

            // The cancel completes while that poll is still out; the refetch
            // reports the run as written off.
            api.getUserGuardrails.mockResolvedValue({
                data: { classifiers: [guardrailFixture({ status: 'error', training_phase_detail: null, training_log: 'Training was cancelled.' })] },
            });
            fireEvent.click(btn);
            await waitFor(() => expect(api.getUserGuardrails).toHaveBeenCalledTimes(2));
            expect(await screen.findByText('Training was cancelled.')).toBeInTheDocument();

            // ...and only now does the stale poll answer.
            await act(async () => {
                answerPoll({ data: { classifier_id: 5, status: 'training', is_training: true, training_phase_detail: 'Training RNN — Epoch 4 of 10' } });
            });
            expect(screen.queryByText('Training RNN — Epoch 4 of 10')).not.toBeInTheDocument();
            expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
            expect(screen.getByText('Training was cancelled.')).toBeInTheDocument();
        } finally {
            vi.useRealTimers();
        }
    });
});

describe('Guardrails — how a stopped run reads on the card', () => {
    it('says a cancelled run was cancelled, without the failure banner', async () => {
        api.getUserGuardrails.mockResolvedValue({
            data: {
                classifiers: [guardrailFixture({
                    status: 'error', training_phase_detail: null,
                    training_log: 'Training was cancelled.',
                })],
            },
        });
        renderPage();
        expect(await screen.findByText('Training was cancelled.')).toBeInTheDocument();
    });

    it('still shows the real failure for a run that actually failed', async () => {
        api.getUserGuardrails.mockResolvedValue({
            data: {
                classifiers: [guardrailFixture({
                    status: 'error',
                    training_phase_detail: 'GPU worker unreachable for 900s',
                    training_log: null,
                })],
            },
        });
        renderPage();
        expect(await screen.findByText('GPU worker unreachable for 900s')).toBeInTheDocument();
        expect(screen.queryByText('Training was cancelled.')).not.toBeInTheDocument();
    });
});
