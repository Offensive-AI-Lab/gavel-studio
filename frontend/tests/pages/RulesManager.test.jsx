// Behavior tests for RulesManager (the Policy Logic Manager).
//
// RulesManager loads a classifier's rules + sidebar context + bookmarks +
// training status on mount, renders a RuleCard per rule, and exposes a
// snapshot-driven train button with a three-way policy state machine
// (empty / aligned / drifted). It also wires the "Add an Existing Rule"
// and "Add CE to Rule" modals, per-rule delete, CE add/remove, predicate
// edit/save (fork vs in-place), training trigger and download.
//
// We mock the network (../api), the CE-removal service,
// the confirm-dialog helpers, sweetalert2 and the Sidebar (which
// Layout renders and which has its own fetches).
// Router useNavigate/useParams come from a real
// MemoryRouter route so classifierId reads a value.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within, act } from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// ---- navigate spy; useParams comes from the real route ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, useNavigate: () => mockNavigate };
});

// ---- API mock: every export RulesManager (and children) touch ----
const ok = (extra = {}) => Promise.resolve({ data: extra });
vi.mock('../../src/api', () => ({
    default: { get: vi.fn(() => ok()), post: vi.fn(() => ok()), delete: vi.fn(() => ok()), put: vi.fn(() => ok()) },
    getClassifierRules: vi.fn(() => ok({ rules: [] })),
    deleteRuleSetup: vi.fn(() => ok()),
    addRuleToClassifier: vi.fn(() => ok()),
    getClassifierDetails: vi.fn(() => ok({ model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [], trained_rule_names: [] })),
    updateRuleLogic: vi.fn(() => ok({ predicate: 'NEW PRED' })),
    checkRuleDuplicate: vi.fn(() => ok({ exists: false })),
    saveEditedRule: vi.fn(() => ok({ predicate: 'SAVED PRED' })),
    getRuleBookmarks: vi.fn(() => ok({ bookmarks: [] })),
    getCEBookmarks: vi.fn(() => ok({ bookmarks: [] })),
    trainClassifier: vi.fn(() => ok()),
    getTrainingStatus: vi.fn(() => ok({ status: 'untrained', is_training: false, training_phase: null, training_phase_detail: null })),
    cancelTraining: vi.fn(() => ok({ success: true, status: 'needs_retraining' })),
    downloadClassifier: vi.fn(() => Promise.resolve()),
    listLocalDrafts: vi.fn(() => ok({ rules: [] })),
    // The "Add a Rule" picker searches the public library inline.
    searchLibrary: vi.fn(() => ok({ results: [], total_results: 0 })),
    // Was missing from this mock, so the page's call threw and was swallowed.
    // Defined now so the models loading/empty split is actually testable.
    getUserModels: vi.fn(() => ok({ models: [] })),
    getComputeStatus: vi.fn(() => Promise.resolve({ data: { workloads: {} } })),
    // Machine picker — default to a single target so training proceeds directly.
    getComputeTargets: vi.fn(() => Promise.resolve({ data: { targets: [{ name: 'local', label: 'This machine' }] } })),
}));

// ---- CE-removal service: assert calls, never hit the real flow ----
vi.mock('../../src/services/CEService', () => ({
    handleRemoveCEFlow: vi.fn((setupId, ceId, ceName, rules, ruleIndex, cb) => {
        // Mirror the real flow: invoke the callback with an updated copy.
        const next = rules.map((r, i) =>
            i === ruleIndex ? { ...r, active_ces: r.active_ces.filter((c) => c.ce_id !== ceId) } : r);
        return cb(next);
    }),
}));


// ---- confirm dialog helpers, controllable per test ----
const mockConfirm = vi.fn(() => Promise.resolve(true));
const mockAlert = vi.fn(() => Promise.resolve());
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showConfirmDialog: (...a) => mockConfirm(...a),
    showAlertDialog: (...a) => mockAlert(...a),
}));

// ---- sweetalert2 (fork-name prompt) ----
const swalFire = vi.fn(() => Promise.resolve({ isConfirmed: false }));
vi.mock('sweetalert2', () => ({ default: { fire: (...a) => swalFire(...a), close: vi.fn() } }));

// ---- Sidebar stub (Layout renders it; it has its own fetches) ----
vi.mock('../../src/components/Sidebar/Sidebar', () => ({ default: () => <aside data-testid="sidebar-stub" /> }));

import RulesManager from '../../src/pages/RulesManager';
import * as api from '../../src/api';
import * as CEService from '../../src/services/CEService';

const setUser = () => {
    sessionStorage.setItem('token', 'tok');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderPage = (classifierId = '5') =>
    render(
        <TutorialProvider>
            <MemoryRouter initialEntries={[`/classifiers/${classifierId}/rules`]}>
                <Routes>
                    <Route path="/classifiers/:classifierId/rules" element={<RulesManager />} />
                </Routes>
            </MemoryRouter>
        </TutorialProvider>,
    );

// Mirrors a get_classifier_rules row: v2 logic nested under `logic`
// ({groups: {gname: [{ce_id, name}]}, condition, predicate}) plus a flat
// membership-only `active_ces` list ([{ce_id, name}] — no roles). The base
// fixture is a legacy row (empty groups → predicate-string display).
const ruleFixture = (over = {}) => ({
    setup_id: 1,
    rule_id: 100,
    source_rule_id: 100,
    custom_name: 'Rule Alpha',
    predicate: 'A AND B',
    is_local_draft: true,
    logic: { groups: {}, condition: '', predicate: 'A AND B' },
    active_ces: [
        { ce_id: 11, name: 'CE One' },
        { ce_id: 12, name: 'CE Two' },
    ],
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    // benign defaults
    api.getClassifierRules.mockResolvedValue({ data: { rules: [] } });
    api.getClassifierDetails.mockResolvedValue({ data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [], trained_rule_names: [] } });
    api.getTrainingStatus.mockResolvedValue({ data: { status: 'untrained', is_training: false, training_phase: null, training_phase_detail: null } });
    api.cancelTraining.mockResolvedValue({ data: { success: true, status: 'needs_retraining' } });
    api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.listLocalDrafts.mockResolvedValue({ data: { rules: [] } });
    api.searchLibrary.mockResolvedValue({ data: { results: [], total_results: 0 } });
    api.updateRuleLogic.mockResolvedValue({ data: { predicate: 'NEW PRED' } });
    api.checkRuleDuplicate.mockResolvedValue({ data: { exists: false } });
    api.saveEditedRule.mockResolvedValue({ data: { predicate: 'SAVED PRED' } });
    mockConfirm.mockResolvedValue(true);
    mockAlert.mockResolvedValue();
    swalFire.mockResolvedValue({ isConfirmed: false });
});

afterEach(() => {
    vi.useRealTimers();
    sessionStorage.clear();
});

describe('RulesManager — mount & auth', () => {
    it('returns null and skips fetches when no user', async () => {
        sessionStorage.removeItem('user');
        const { container } = renderPage();
        // user null → component returns null; nothing but providers rendered.
        expect(container.querySelector('.rules-header')).not.toBeInTheDocument();
        expect(api.getClassifierRules).not.toHaveBeenCalled();
    });

    it('loads rules, sidebar context, bookmarks and training status for the param', async () => {
        renderPage('5');
        await waitFor(() => expect(api.getClassifierRules).toHaveBeenCalledWith('5'));
        expect(api.getClassifierDetails).toHaveBeenCalledWith('5');
        expect(api.getTrainingStatus).toHaveBeenCalledWith('5');
        expect(api.getRuleBookmarks).toHaveBeenCalledWith(7);
    });

    it('renders the header heading (the rule set name)', async () => {
        renderPage();
        // Heading is now the rule set's own name from the details fetch.
        expect(await screen.findByRole('heading', { name: 'C' })).toBeInTheDocument();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });
});

describe('RulesManager — empty state', () => {
    it('shows the empty state with both add affordances when there are no rules', async () => {
        renderPage();
        expect(await screen.findByText('No Rules Defined')).toBeInTheDocument();
        // The empty-state buttons (ReactiveButton) duplicate the action cards.
        expect(screen.getAllByText('Add an Existing Rule').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Create a New Rule').length).toBeGreaterThan(0);
    });

    it('tolerates a bare array payload (res.data is the array)', async () => {
        api.getClassifierRules.mockResolvedValue({ data: [ruleFixture()] });
        renderPage();
        expect(await screen.findByText('Rule Alpha')).toBeInTheDocument();
    });

    it('alerts when the rules fetch rejects', async () => {
        api.getClassifierRules.mockRejectedValue(new Error('boom'));
        renderPage();
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(
            expect.objectContaining({ message: 'Failed to load rules', variant: 'error' }),
        ));
    });

    it('falls back to empty bookmarks when the bookmarks fetch rejects', async () => {
        api.getRuleBookmarks.mockRejectedValue(new Error('x'));
        renderPage();
        // Still renders without crashing.
        expect(await screen.findByText('No Rules Defined')).toBeInTheDocument();
    });
});

describe('RulesManager — rule list rendering', () => {
    it('renders a RuleCard per rule with its name', async () => {
        api.getClassifierRules.mockResolvedValue({
            data: { rules: [ruleFixture(), ruleFixture({ setup_id: 2, rule_id: 101, custom_name: 'Rule Beta' })] },
        });
        renderPage();
        expect(await screen.findByText('Rule Alpha')).toBeInTheDocument();
        expect(screen.getByText('Rule Beta')).toBeInTheDocument();
    });

    it('expands a rule on header click to show its predicate', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        const { container } = renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(container.querySelector('.rule-header'));
        expect(await screen.findByText('A AND B')).toBeInTheDocument();
    });
});

describe('RulesManager — policy state machine & train button', () => {
    it('empty: shows the no-rules banner and a disabled Train Classifier button', async () => {
        renderPage();
        expect(await screen.findByText(/No rules in this rule set/)).toBeInTheDocument();
        const btn = screen.getByText('Train Rule Set');
        // Empty state → disabled (no onClick).
        fireEvent.click(btn);
        expect(api.trainClassifier).not.toHaveBeenCalled();
    });

    it('empty-after-training: banner explains every trained rule was removed', async () => {
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [9], trained_rule_names: ['Old Rule'] },
        });
        renderPage();
        expect(await screen.findByText(/removed every rule this rule set was trained on/)).toBeInTheDocument();
    });

    it('aligned-never-trained: rules present, no snapshot → Train Classifier active, no banner', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        await screen.findByText('Rule Alpha');
        expect(screen.getByText('Train Rule Set')).toBeInTheDocument();
        expect(screen.queryByText(/No rules in this rule set/)).not.toBeInTheDocument();
        expect(screen.queryByText(/Rule set differs from the trained model/)).not.toBeInTheDocument();
    });

    it('aligned-trained: current names match snapshot → Up to Date (disabled, no train)', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ custom_name: 'Same' })] } });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [1], trained_rule_names: ['Same'] },
        });
        renderPage();
        expect(await screen.findByText('Up to Date')).toBeInTheDocument();
        fireEvent.click(screen.getByText('Up to Date'));
        expect(api.trainClassifier).not.toHaveBeenCalled();
    });

    it('drifted: selection differs from snapshot → Retrain Classifier + drift banner', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ custom_name: 'New Rule' })] } });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [9], trained_rule_names: ['Trained Rule'] },
        });
        renderPage();
        expect(await screen.findByText('Retrain Rule Set')).toBeInTheDocument();
        expect(screen.getByText(/Rule set differs from the trained model/)).toBeInTheDocument();
    });

    it('drifted by size: more current rules than trained → drifted', async () => {
        api.getClassifierRules.mockResolvedValue({
            data: { rules: [ruleFixture({ custom_name: 'A' }), ruleFixture({ setup_id: 2, custom_name: 'B' })] },
        });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [1], trained_rule_names: ['A'] },
        });
        renderPage();
        expect(await screen.findByText('Retrain Rule Set')).toBeInTheDocument();
    });
});

describe('RulesManager — training trigger', () => {
    const oneRule = () => api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });

    it('confirms then calls trainClassifier and flips to a training banner', async () => {
        oneRule();
        renderPage();
        const btn = await screen.findByText('Train Rule Set');
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Train rule set?' }),
        ));
        await waitFor(() => expect(api.trainClassifier).toHaveBeenCalledWith('5', null));
        // Status flips to training → indigo phase banner appears.
        expect(await screen.findByText('Training in progress')).toBeInTheDocument();
    });

    it('does not train when the confirm dialog is cancelled', async () => {
        mockConfirm.mockResolvedValue(false);
        oneRule();
        renderPage();
        fireEvent.click(await screen.findByText('Train Rule Set'));
        await waitFor(() => expect(mockConfirm).toHaveBeenCalled());
        expect(api.trainClassifier).not.toHaveBeenCalled();
    });

    it('locks the button + shows the progress banner from the moment of click (before confirm resolves)', async () => {
        oneRule();
        // Hold the confirm dialog open so we can inspect the state in between.
        let resolveConfirm;
        mockConfirm.mockReturnValue(new Promise((res) => { resolveConfirm = res; }));
        renderPage();
        fireEvent.click(await screen.findByText('Train Rule Set'));
        // While the confirm is still open: button already locked to "Submitting..."
        // and the progress banner is up — and nothing has been submitted yet.
        expect(await screen.findByText('Submitting...')).toBeInTheDocument();
        expect(screen.getByText('Looking for a GPU')).toBeInTheDocument();
        expect(api.trainClassifier).not.toHaveBeenCalled();
        // Confirm → it proceeds to actually submit.
        resolveConfirm(true);
        await waitFor(() => expect(api.trainClassifier).toHaveBeenCalledWith('5', null));
    });

    it('unlocks the button when the confirm is cancelled (no stuck Submitting…)', async () => {
        mockConfirm.mockResolvedValue(false);
        oneRule();
        renderPage();
        fireEvent.click(await screen.findByText('Train Rule Set'));
        // Cancel rolls submitting back, so the Train button returns.
        expect(await screen.findByText('Train Rule Set')).toBeInTheDocument();
        expect(screen.queryByText('Submitting...')).not.toBeInTheDocument();
    });

    it('uses the retrain (destructive) confirm copy when a snapshot exists', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ custom_name: 'New' })] } });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [9], trained_rule_names: ['Old'] },
        });
        renderPage();
        fireEvent.click(await screen.findByText('Retrain Rule Set'));
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Retrain rule set?', variant: 'danger' }),
        ));
    });

    it('alerts with the server detail when training submission fails', async () => {
        oneRule();
        api.trainClassifier.mockRejectedValue({ response: { data: { detail: 'No GPU' } } });
        renderPage();
        fireEvent.click(await screen.findByText('Train Rule Set'));
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Error', message: 'No GPU', variant: 'error' }),
        ));
    });
});

describe('RulesManager — training status (in-flight)', () => {
    it('shows the live phase + detail banner while training', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: { status: 'training', is_training: true, training_phase: 'Training RNN', training_phase_detail: 'Epoch 3 of 10' },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Training RNN')).toBeInTheDocument();
        expect(screen.getByText(/Epoch 3 of 10/)).toBeInTheDocument();
        // Train button reads Training... while in flight.
        expect(screen.getByText('Training...')).toBeInTheDocument();
    });

    it('shows a clear, friendly error banner when a no-chat-template model fails', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'error', is_training: false, has_error: true,
                training_phase: 'failed',
                training_phase_detail:
                    'Cannot use chat template functions because tokenizer.chat_template '
                    + 'is not set and no template argument was passed!',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText(/Training failed/i)).toBeInTheDocument();
        expect(screen.getByText(/no chat template/i)).toBeInTheDocument();
        // The raw error is still available under "Details:".
        expect(screen.getByText(/Details:/)).toBeInTheDocument();
    });

    it('shows the raw error text for an unrecognized training failure', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'error', is_training: false, has_error: true,
                training_phase: 'oom', training_phase_detail: 'CUDA out of memory',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText(/Training failed/i)).toBeInTheDocument();
        expect(screen.getByText(/CUDA out of memory/)).toBeInTheDocument();
        // No "Details:" line when the friendly message is the raw text.
        expect(screen.queryByText(/Details:/)).not.toBeInTheDocument();
    });

    it('seeds the training banner instantly from sessionStorage cache', async () => {
        sessionStorage.setItem('trainStatus_5', 'training');
        sessionStorage.setItem('trainPhase_5', 'Cached Phase');
        sessionStorage.setItem('trainDetail_5', 'Cached Detail');
        // Keep the status fetch pending so the seed is what we observe.
        api.getTrainingStatus.mockReturnValue(new Promise(() => {}));
        renderPage();
        expect(await screen.findByText('Cached Phase')).toBeInTheDocument();
        expect(screen.getByText(/Cached Detail/)).toBeInTheDocument();
    });
});

describe('RulesManager — cancel training', () => {
    const training = (over = {}) => ({
        data: {
            status: 'training', is_training: true,
            training_phase: 'Training RNN', training_phase_detail: 'Epoch 3 of 10',
            ...over,
        },
    });
    // Mount mid-training and wait for the cancel action in the banner.
    const renderTraining = async () => {
        api.getTrainingStatus.mockResolvedValue(training());
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        return screen.findByText('Cancel training');
    };

    it('offers the cancel action while a run is in progress', async () => {
        await renderTraining();
        expect(screen.getByTitle('Stop this training run')).toBeInTheDocument();
    });

    it('does not offer it when nothing is running', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Train Rule Set')).toBeInTheDocument();
        expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
    });

    it('does not offer it during the submit round-trip (nothing to stop yet)', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        mockConfirm.mockReturnValue(new Promise(() => {}));   // hold the train confirm open
        renderPage();
        fireEvent.click(await screen.findByText('Train Rule Set'));
        expect(await screen.findByText('Looking for a GPU')).toBeInTheDocument();
        expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
    });

    it('does not offer it for the post-training chain (calibration is not cancellable here)', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'active', is_training: false,
                training_phase: null, training_phase_detail: null,
                post_training_phase: 'Calibrating', post_training_phase_detail: 'Loading datasets…',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Calibrating')).toBeInTheDocument();
        expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
    });

    it('confirms first, then cancels and re-reads the status', async () => {
        const btn = await renderTraining();
        // The next status read reports the row as crash recovery leaves it.
        api.getTrainingStatus.mockResolvedValue({
            data: { status: 'needs_retraining', is_training: false, training_phase: null, training_phase_detail: null },
        });
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Cancel training?', variant: 'danger' }),
        ));
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledWith('5'));
        // Banner (and the run) gone; the train button is live again.
        expect(await screen.findByText('Train Rule Set')).toBeInTheDocument();
        expect(screen.queryByText('Cancel training')).not.toBeInTheDocument();
        expect(mockAlert).not.toHaveBeenCalled();
    });

    it('does not cancel when the confirm dialog is dismissed', async () => {
        const btn = await renderTraining();
        mockConfirm.mockResolvedValue(false);
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalled());
        expect(api.cancelTraining).not.toHaveBeenCalled();
        expect(screen.getByText('Cancel training')).toBeInTheDocument();
    });

    it('treats a 409 as "already over": re-reads the status, no error dialog', async () => {
        const btn = await renderTraining();
        api.cancelTraining.mockRejectedValue({ response: { status: 409, data: { detail: 'Not training' } } });
        const readsBefore = api.getTrainingStatus.mock.calls.length;
        api.getTrainingStatus.mockResolvedValue({
            data: { status: 'active', is_training: false, training_phase: null, training_phase_detail: null },
        });
        fireEvent.click(btn);
        await waitFor(() => expect(api.getTrainingStatus.mock.calls.length).toBeGreaterThan(readsBefore));
        await waitFor(() => expect(screen.queryByText('Cancel training')).not.toBeInTheDocument());
        expect(mockAlert).not.toHaveBeenCalled();
    });

    it('alerts with the server detail when the cancel fails for another reason', async () => {
        const btn = await renderTraining();
        api.cancelTraining.mockRejectedValue({ response: { status: 500, data: { detail: 'Could not reach the worker' } } });
        fireEvent.click(btn);
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Error', message: 'Could not reach the worker', variant: 'error' }),
        ));
        // The run is still going, so the action stays available.
        expect(screen.getByText('Cancel training')).toBeInTheDocument();
    });

    it('locks the button while the cancel is in flight (no double submit)', async () => {
        const btn = await renderTraining();
        let resolveCancel;
        api.cancelTraining.mockReturnValue(new Promise((res) => { resolveCancel = res; }));
        fireEvent.click(btn);
        expect(await screen.findByText('Cancelling...')).toBeInTheDocument();
        fireEvent.click(screen.getByTitle('Stop this training run'));
        expect(api.cancelTraining).toHaveBeenCalledTimes(1);
        resolveCancel({ data: { success: true, status: 'needs_retraining' } });
        await waitFor(() => expect(screen.queryByText('Cancelling...')).not.toBeInTheDocument());
    });

    it('two clicks in the same tick open ONE confirm dialog', async () => {
        // The lock goes on before the dialog: a second click in the same tick
        // still sees the pre-render state, so only a ref can stop it stacking
        // a second dialog on the first.
        const btn = await renderTraining();
        let resolveConfirm;
        mockConfirm.mockReturnValue(new Promise((res) => { resolveConfirm = res; }));
        fireEvent.click(btn);
        fireEvent.click(btn);
        expect(mockConfirm).toHaveBeenCalledTimes(1);
        resolveConfirm(true);
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledTimes(1));
    });

    it('re-arms the button when the confirm is declined', async () => {
        const btn = await renderTraining();
        mockConfirm.mockResolvedValue(false);
        fireEvent.click(btn);
        await waitFor(() => expect(mockConfirm).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(screen.getByTitle('Stop this training run')).not.toBeDisabled());
        mockConfirm.mockResolvedValue(true);
        fireEvent.click(screen.getByTitle('Stop this training run'));
        await waitFor(() => expect(api.cancelTraining).toHaveBeenCalledTimes(1));
    });
});

describe('RulesManager — how a stopped run reads', () => {
    // A cancelled run and a failed run leave the same status ('error'); only
    // the reason in training_log tells them apart, and they must not read the
    // same to the user.
    it('says a cancelled run was cancelled, without the failure banner', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'error', is_training: false, has_error: true,
                training_phase: null, training_phase_detail: null,
                training_log: 'Training was cancelled.',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Training was cancelled.')).toBeInTheDocument();
        expect(screen.queryByText('Training failed.')).not.toBeInTheDocument();
    });

    it('still shows the failure banner for a run that actually failed', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'error', is_training: false, has_error: true,
                training_phase: 'failed',
                training_phase_detail: 'GPU worker unreachable for 900s',
                training_log: null,
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Training failed.')).toBeInTheDocument();
        expect(screen.queryByText('Training was cancelled.')).not.toBeInTheDocument();
    });
});

describe('RulesManager — add existing rule modal', () => {
    it('opens the modal from the action card and lists merged bookmarks + drafts', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Bookmarked R', predicate: 'X' }] } });
        api.listLocalDrafts.mockResolvedValue({ data: { rules: [{ rule_id: 60, name: 'Draft R', predicate: 'Y' }] } });
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        expect(await screen.findByText('Bookmarked R')).toBeInTheDocument();
        expect(screen.getByText('Draft R')).toBeInTheDocument();
        expect(screen.getByText('BOOKMARK')).toBeInTheDocument();
        expect(screen.getByText('DRAFT')).toBeInTheDocument();
    });

    it('shows an empty-list message when no bookmarks, drafts or library rules exist', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        expect(await screen.findByText(/No rules to add yet/)).toBeInTheDocument();
    });

    it('adds a selected rule to the classifier and refetches', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Bookmarked R', predicate: 'X' }] } });
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        fireEvent.click(await screen.findByText('Bookmarked R'));
        fireEvent.click(screen.getByText('Add to Rule Set'));
        await waitFor(() => expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '50'));
        // refreshData re-fetches the rules.
        await waitFor(() => expect(api.getClassifierRules).toHaveBeenCalledTimes(2));
    });

    it('warns (without crashing) when one of the selected rules fails to add', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Bookmarked R', predicate: 'X' }] } });
        api.addRuleToClassifier.mockRejectedValue(new Error('nope'));
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        fireEvent.click(await screen.findByText('Bookmarked R'));
        fireEvent.click(screen.getByText('Add to Rule Set'));
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Some rules not added', variant: 'warning' }),
        ));
    });

    it('adds MULTIPLE selected rules in one go (multi-select)', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [
            { rule_id: 50, name: 'Bookmarked R', predicate: 'X' },
            { rule_id: 51, name: 'Second R', predicate: 'Z' },
        ] } });
        api.listLocalDrafts.mockResolvedValue({ data: { rules: [{ rule_id: 60, name: 'Draft R', predicate: 'Y' }] } });
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        // Check three rules across both sources.
        fireEvent.click(await screen.findByText('Bookmarked R'));
        fireEvent.click(screen.getByText('Second R'));
        fireEvent.click(screen.getByText('Draft R'));
        // Button label reflects the count.
        expect(screen.getByText(/Add 3 Rules to Rule Set/)).toBeInTheDocument();
        fireEvent.click(screen.getByText(/Add 3 Rules to Rule Set/));
        await waitFor(() => expect(api.addRuleToClassifier).toHaveBeenCalledTimes(3));
        expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '50');
        expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '51');
        expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '60');
    });

    it('deselecting a checked rule removes it (toggle)', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Bookmarked R', predicate: 'X' }] } });
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        const row = await screen.findByText('Bookmarked R');
        fireEvent.click(row);   // select
        expect(screen.getByText('Add to Rule Set')).not.toBeDisabled();
        fireEvent.click(row);   // deselect (toggle off — proves no double-fire)
        // Add button is disabled again with nothing selected.
        expect(screen.getByText('Add to Rule Set').closest('button')).toBeDisabled();
    });

    it('hides rules already attached to the classifier', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ source_rule_id: 50 })] } });
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Already Attached', predicate: 'X' }] } });
        renderPage();
        await screen.findByText('Rule Alpha');
        // Open via the always-present action card.
        fireEvent.click(screen.getByText('Add an Existing Rule'));
        expect(await screen.findByText(/No rules to add yet/)).toBeInTheDocument();
        expect(screen.queryByText('Already Attached')).not.toBeInTheDocument();
    });
});

// Adding a community rule used to mean leaving for the Community page,
// bookmarking the rule and coming back. The picker searches the public library
// itself now: local bookmarks/drafts narrow to matches, library hits are
// appended, and the existing selection/add machinery attaches them.
describe('RulesManager — add existing rule modal, inline library search', () => {
    const libraryHit = (over = {}) => ({
        id: 70,
        asset_type: 'rule',
        name: 'Toxic Speech',
        content: 'CE One AND CE Two',
        description: 'Flags toxic speech.',
        public_id: 'pub-70',
        is_local_draft: false,
        hybrid_score: 0.9,
        ...over,
    });

    const openModal = async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        return screen.findByLabelText('Search rules');
    };

    const type = (input, value) => fireEvent.change(input, { target: { value } });

    // jsdom does no layout, so the scroll metrics the handler reads are all 0.
    // Stub them to "the user is at the bottom" and fire the event React listens
    // for on the list container.
    const scrollToBottom = () => {
        const el = screen.getByTestId('add-rule-list');
        Object.defineProperty(el, 'scrollHeight', { value: 900, configurable: true });
        Object.defineProperty(el, 'clientHeight', { value: 300, configurable: true });
        Object.defineProperty(el, 'scrollTop', { value: 600, configurable: true });
        fireEvent.scroll(el);
        return el;
    };

    // Serve a different response per requested page.
    const pagedLibrary = (pages, totalResults) =>
        api.searchLibrary.mockImplementation(({ page }) =>
            ok({ results: pages[(page || 1) - 1] || [], total_results: totalResults }));

    it('searches the library for rules and shows the hits with a LIBRARY badge', async () => {
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Toxic Speech')).toBeInTheDocument();
        expect(screen.getByText('LIBRARY')).toBeInTheDocument();
        // The rule's explanation renders, same as for bookmarks/drafts.
        expect(screen.getByText('Flags toxic speech.')).toBeInTheDocument();
        expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({
            q: 'toxic',
            asset_types: 'rule',
        }));
    });

    it('does not search below the two-character minimum', async () => {
        const input = await openModal();
        type(input, 't');
        await new Promise((r) => setTimeout(r, 400));
        // The empty box browses ('*'), so the picker may have asked for the
        // browse list — but never for the half-typed word itself.
        expect(api.searchLibrary).not.toHaveBeenCalledWith(expect.objectContaining({ q: 't' }));
    });

    it('adds a searched library rule through the normal add flow', async () => {
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        const input = await openModal();
        type(input, 'toxic');
        fireEvent.click(await screen.findByText('Toxic Speech'));
        fireEvent.click(screen.getByText('Add to Rule Set'));
        await waitFor(() => expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '70'));
        // No bookmark is created on the way — attaching is the whole point.
        await waitFor(() => expect(api.getClassifierRules).toHaveBeenCalledTimes(2));
    });

    it('hides a library hit that is already attached to this rule set', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ source_rule_id: 70 })] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(screen.getByText('Add an Existing Rule'));
        type(await screen.findByLabelText('Search rules'), 'toxic');
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        expect(await screen.findByText(/No rules match your search/)).toBeInTheDocument();
        expect(screen.queryByText('Toxic Speech')).not.toBeInTheDocument();
    });

    it('dedupes a library hit against the bookmark row for the same rule', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [
            { rule_id: 70, name: 'Toxic Bookmark', predicate: 'X', public_id: 'pub-70' },
        ] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        const input = await openModal();
        type(input, 'toxic');
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        // One row, keeping the badge that belongs to the row's own state.
        expect(await screen.findByText('Toxic Bookmark')).toBeInTheDocument();
        expect(screen.queryByText('Toxic Speech')).not.toBeInTheDocument();
        expect(screen.getByText('BOOKMARK')).toBeInTheDocument();
        expect(screen.queryByText('LIBRARY')).not.toBeInTheDocument();
    });

    it('badges a draft library hit DRAFT, not LIBRARY', async () => {
        api.searchLibrary.mockResolvedValue({
            data: { results: [libraryHit({ id: 71, name: 'Draft Hit', public_id: null, is_local_draft: true })], total_results: 1 },
        });
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Draft Hit')).toBeInTheDocument();
        expect(screen.getByText('DRAFT')).toBeInTheDocument();
        expect(screen.queryByText('LIBRARY')).not.toBeInTheDocument();
    });

    it('narrows the local list while searching and restores it when the query is cleared', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Alpha Rule', predicate: 'X' }] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        const input = await openModal();
        expect(await screen.findByText('Alpha Rule')).toBeInTheDocument();
        type(input, 'toxic');
        expect(await screen.findByText('Toxic Speech')).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByText('Alpha Rule')).not.toBeInTheDocument());
        // Empty query → the local rows are back, unfiltered.
        type(input, '');
        expect(await screen.findByText('Alpha Rule')).toBeInTheDocument();
    });

    it('does not ask the user to narrow the search when more matches exist', async () => {
        // The truncation hint is gone: the rest of the matches arrive by
        // scrolling, so neither a count nor an end-of-list note belongs here.
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 4 } });
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Toxic Speech')).toBeInTheDocument();
        expect(screen.queryByText(/more results/)).not.toBeInTheDocument();
        expect(screen.queryByText(/No more rules to show/)).not.toBeInTheDocument();
    });

    it('appends the next page of library hits when the list is scrolled to the bottom', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Toxic Bookmark', predicate: 'X' }] } });
        pagedLibrary([
            [libraryHit({ id: 70, name: 'Hit One' })],
            [libraryHit({ id: 71, name: 'Hit Two' })],
        ], 2);
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Hit One')).toBeInTheDocument();
        expect(screen.queryByText('Hit Two')).not.toBeInTheDocument();

        scrollToBottom();
        expect(await screen.findByText('Hit Two')).toBeInTheDocument();
        // Page 1 stays put — pages accumulate, they do not replace each other.
        expect(screen.getByText('Hit One')).toBeInTheDocument();
        // Local rows still lead the list, library hits still follow.
        expect(screen.getByText('Toxic Bookmark')).toBeInTheDocument();
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({ q: 'toxic', page: 2 })));
    });

    it('keeps a ticked row ticked when the next page arrives', async () => {
        pagedLibrary([
            [libraryHit({ id: 70, name: 'Hit One' })],
            [libraryHit({ id: 71, name: 'Hit Two' })],
        ], 2);
        const input = await openModal();
        type(input, 'toxic');
        fireEvent.click(await screen.findByText('Hit One'));
        scrollToBottom();
        expect(await screen.findByText('Hit Two')).toBeInTheDocument();
        expect(screen.getByText('Hit One').closest('[role="checkbox"]')).toHaveAttribute('aria-checked', 'true');
    });

    it('stops asking for pages once every match is listed', async () => {
        pagedLibrary([[libraryHit()]], 1);
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Toxic Speech')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByText(/No more rules to show/)).toBeInTheDocument());

        const before = api.searchLibrary.mock.calls.length;
        scrollToBottom();
        scrollToBottom();
        await new Promise((r) => setTimeout(r, 400));
        expect(api.searchLibrary).toHaveBeenCalledTimes(before);
        expect(api.searchLibrary).not.toHaveBeenCalledWith(expect.objectContaining({ page: 2 }));
    });

    it('lists a rule once when two pages overlap', async () => {
        const shared = libraryHit({ id: 70, name: 'Shared Hit' });
        pagedLibrary([
            [shared, libraryHit({ id: 71, name: 'Hit Two' })],
            [shared, libraryHit({ id: 72, name: 'Hit Three' })],
        ], 4);
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Shared Hit')).toBeInTheDocument();

        scrollToBottom();
        expect(await screen.findByText('Hit Three')).toBeInTheDocument();
        expect(screen.getAllByText('Shared Hit')).toHaveLength(1);
        expect(screen.getAllByText('Hit Two')).toHaveLength(1);
    });

    it('drops the pages collected so far when the query changes', async () => {
        api.searchLibrary.mockImplementation(({ q, page }) => {
            if (q === 'toxic') {
                return ok({
                    results: page === 1
                        ? [libraryHit({ id: 70, name: 'Toxic One' })]
                        : [libraryHit({ id: 71, name: 'Toxic Two' })],
                    total_results: 2,
                });
            }
            return ok({ results: [libraryHit({ id: 80, name: 'Spam One' })], total_results: 1 });
        });
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Toxic One')).toBeInTheDocument();
        scrollToBottom();
        expect(await screen.findByText('Toxic Two')).toBeInTheDocument();

        type(input, 'spammy');
        expect(await screen.findByText('Spam One')).toBeInTheDocument();
        expect(screen.queryByText('Toxic One')).not.toBeInTheDocument();
        expect(screen.queryByText('Toxic Two')).not.toBeInTheDocument();
        // The new query starts again at page 1.
        expect(api.searchLibrary).toHaveBeenLastCalledWith(expect.objectContaining({ q: 'spammy', page: 1 }));
    });

    it('shows a loading line while the next page is on its way', async () => {
        let releasePage2;
        api.searchLibrary.mockImplementation(({ page }) => {
            if (page === 1) return ok({ results: [libraryHit({ id: 70, name: 'Hit One' })], total_results: 2 });
            return new Promise((resolve) => {
                releasePage2 = () => resolve({ data: { results: [libraryHit({ id: 71, name: 'Hit Two' })], total_results: 2 } });
            });
        });
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText('Hit One')).toBeInTheDocument();

        scrollToBottom();
        expect(await screen.findByText(/Loading more rules/)).toBeInTheDocument();
        // Still showing once the request is actually out and unanswered.
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({ page: 2 })));
        expect(screen.getByText(/Loading more rules/)).toBeInTheDocument();
        await act(async () => { releasePage2(); });
        expect(await screen.findByText('Hit Two')).toBeInTheDocument();
        expect(screen.queryByText(/Loading more rules/)).not.toBeInTheDocument();
    });

    it('shows the search error without losing the local list', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Toxic Bookmark', predicate: 'X' }] } });
        api.searchLibrary.mockRejectedValue(new Error('down'));
        const input = await openModal();
        type(input, 'toxic');
        expect(await screen.findByText(/Search failed/)).toBeInTheDocument();
        expect(screen.getByText('Toxic Bookmark')).toBeInTheDocument();
    });
});

// An empty box used to list bookmarks and drafts only, so there was nothing to
// scroll into and the paging looked broken. The picker now browses the public
// library from the moment it opens: bookmarks and drafts first, the library
// under them, more of it as the user scrolls.
describe('RulesManager — add existing rule modal, browsing with an empty box', () => {
    const libraryHit = (over = {}) => ({
        id: 70,
        asset_type: 'rule',
        name: 'Toxic Speech',
        content: 'CE One AND CE Two',
        description: 'Flags toxic speech.',
        public_id: 'pub-70',
        is_local_draft: false,
        ...over,
    });

    // Browse (empty box) and search (typed) are told apart by the query the
    // picker sends: '*' is the backend's browse sentinel.
    const BROWSE_Q = '*';

    const openModal = async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        return screen.findByLabelText('Search rules');
    };

    const type = (input, value) => fireEvent.change(input, { target: { value } });

    const scrollToBottom = () => {
        const el = screen.getByTestId('add-rule-list');
        Object.defineProperty(el, 'scrollHeight', { value: 900, configurable: true });
        Object.defineProperty(el, 'clientHeight', { value: 300, configurable: true });
        Object.defineProperty(el, 'scrollTop', { value: 600, configurable: true });
        fireEvent.scroll(el);
    };

    // The rows in list order, top to bottom (name span of each row).
    // Row containers only — the tick box inside each row carries the same role.
    const rowNames = () => Array.from(
        screen.getByTestId('add-rule-list').querySelectorAll('div[role="checkbox"]'),
    ).map((row) => row.querySelector('span').textContent);

    it('lists library rules under the bookmarks and drafts without any typing', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [{ rule_id: 50, name: 'Bookmarked R', predicate: 'X' }] } });
        api.listLocalDrafts.mockResolvedValue({ data: { rules: [{ rule_id: 60, name: 'Draft R', predicate: 'Y' }] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        await openModal();

        expect(await screen.findByText('Toxic Speech')).toBeInTheDocument();
        // Local rows lead, the library follows.
        expect(rowNames()).toEqual(['Bookmarked R', 'Draft R', 'Toxic Speech']);
        expect(screen.getByText('LIBRARY')).toBeInTheDocument();
        expect(api.searchLibrary).toHaveBeenCalledWith(expect.objectContaining({
            q: BROWSE_Q, asset_types: 'rule', page: 1,
        }));
    });

    it('pages in the next batch of library rules as the list is scrolled', async () => {
        api.searchLibrary.mockImplementation(({ page }) => ok({
            results: page === 1
                ? [libraryHit({ id: 70, name: 'Browse One' })]
                : [libraryHit({ id: 71, name: 'Browse Two' })],
            total_results: 2,
        }));
        await openModal();
        expect(await screen.findByText('Browse One')).toBeInTheDocument();
        expect(screen.queryByText('Browse Two')).not.toBeInTheDocument();

        scrollToBottom();
        expect(await screen.findByText('Browse Two')).toBeInTheDocument();
        // Page 1 is kept, not replaced.
        expect(rowNames()).toEqual(['Browse One', 'Browse Two']);
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalledWith(
            expect.objectContaining({ q: BROWSE_Q, page: 2 }),
        ));
    });

    it('returns to the browse list when the typed query is cleared', async () => {
        api.searchLibrary.mockImplementation(({ q }) => ok({
            results: q === BROWSE_Q
                ? [libraryHit({ id: 70, name: 'Browse One' })]
                : [libraryHit({ id: 80, name: 'Searched Hit' })],
            total_results: 1,
        }));
        const input = await openModal();
        expect(await screen.findByText('Browse One')).toBeInTheDocument();

        type(input, 'toxic');
        expect(await screen.findByText('Searched Hit')).toBeInTheDocument();
        expect(screen.queryByText('Browse One')).not.toBeInTheDocument();

        type(input, '');
        expect(await screen.findByText('Browse One')).toBeInTheDocument();
        expect(screen.queryByText('Searched Hit')).not.toBeInTheDocument();
    });

    it('hides a browsed rule that is already attached to this rule set', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ source_rule_id: 70 })] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(screen.getByText('Add an Existing Rule'));
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        expect(await screen.findByText(/No rules to add yet/)).toBeInTheDocument();
        expect(screen.queryByText('Toxic Speech')).not.toBeInTheDocument();
    });

    it('dedupes a browsed rule against the bookmark row for the same rule', async () => {
        api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [
            { rule_id: 70, name: 'Toxic Bookmark', predicate: 'X', public_id: 'pub-70' },
        ] } });
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        await openModal();
        await waitFor(() => expect(api.searchLibrary).toHaveBeenCalled());
        expect(await screen.findByText('Toxic Bookmark')).toBeInTheDocument();
        expect(screen.queryByText('Toxic Speech')).not.toBeInTheDocument();
        expect(screen.getByText('BOOKMARK')).toBeInTheDocument();
        expect(screen.queryByText('LIBRARY')).not.toBeInTheDocument();
    });

    it('attaches a browsed rule through the normal add flow', async () => {
        api.searchLibrary.mockResolvedValue({ data: { results: [libraryHit()], total_results: 1 } });
        await openModal();
        fireEvent.click(await screen.findByText('Toxic Speech'));
        fireEvent.click(screen.getByText('Add to Rule Set'));
        await waitFor(() => expect(api.addRuleToClassifier).toHaveBeenCalledWith('5', '70'));
    });

    it('does not browse while the picker is closed', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        await new Promise((r) => setTimeout(r, 400));
        expect(api.searchLibrary).not.toHaveBeenCalled();
    });
});

describe('RulesManager — delete rule', () => {
    beforeEach(() => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
    });

    it('deletes a rule after confirm and removes it from the list', async () => {
        const { container } = renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(container.querySelector('.delete-icon'));
        await waitFor(() => expect(mockConfirm).toHaveBeenCalled());
        await waitFor(() => expect(api.deleteRuleSetup).toHaveBeenCalledWith(1));
        await waitFor(() => expect(screen.queryByText('Rule Alpha')).not.toBeInTheDocument());
    });

    it('does not delete when the confirm is cancelled', async () => {
        mockConfirm.mockResolvedValue(false);
        const { container } = renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(container.querySelector('.delete-icon'));
        await waitFor(() => expect(mockConfirm).toHaveBeenCalled());
        expect(api.deleteRuleSetup).not.toHaveBeenCalled();
        expect(screen.getByText('Rule Alpha')).toBeInTheDocument();
    });
});

describe('RulesManager — export & test-set entry points', () => {
    it('offers Export on a draft rule instead of the removed Publish button', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ is_local_draft: true })] } });
        renderPage();
        await screen.findByText('Rule Alpha');
        expect(screen.getByRole('button', { name: /export rule/i })).toBeEnabled();
        expect(screen.queryByRole('button', { name: /publish/i })).toBeNull();
    });

    it('opens the in-place logic editor (groups + condition) and saves via saveEditedRule', async () => {
        api.getClassifierRules.mockResolvedValue({
            data: {
                rules: [ruleFixture({
                    logic: {
                        groups: { required: [{ ce_id: 11, name: 'CE One' }, { ce_id: 12, name: 'CE Two' }] },
                        condition: 'all of required',
                        predicate: 'CE One AND CE Two',
                    },
                })],
            },
        });
        renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(screen.getByRole('button', { name: "Edit this rule's groups and condition in place" }));
        // Modal shows the group editor prefilled with the rule's condition.
        const conditionInput = await screen.findByLabelText('Firing condition');
        expect(conditionInput).toHaveValue('all of required');
        // The condition is generated now — choose "any 1" instead of typing.
        expect(conditionInput).toHaveAttribute('readonly');
        fireEvent.change(screen.getByLabelText('How much of required must match'),
            { target: { value: '1' } });
        fireEvent.click(screen.getByRole('button', { name: /Save logic/ }));
        await waitFor(() => expect(api.saveEditedRule).toHaveBeenCalledTimes(1));
        expect(api.saveEditedRule).toHaveBeenCalledWith(1, {
            groups: { required: [11, 12] },
            condition: '1 of required',
            new_name: null,
        });
        // Duplicate probe ran against the same shape, excluding this setup.
        expect(api.checkRuleDuplicate).toHaveBeenCalledWith({
            groups: { required: [11, 12] },
            condition: '1 of required',
            classifier_id: 5,
            exclude_setup_id: 1,
        });
    });

    it('surfaces the backend 409 detail when saving edited logic fails', async () => {
        api.getClassifierRules.mockResolvedValue({
            data: {
                rules: [ruleFixture({
                    logic: {
                        groups: { required: [{ ce_id: 11, name: 'CE One' }] },
                        condition: 'all of required',
                        predicate: 'CE One',
                    },
                })],
            },
        });
        api.saveEditedRule.mockRejectedValue({ response: { data: { detail: 'name already taken' } } });
        renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(screen.getByRole('button', { name: "Edit this rule's groups and condition in place" }));
        await screen.findByLabelText('Firing condition');
        fireEvent.click(screen.getByRole('button', { name: /Save logic/ }));
        await waitFor(() => expect(mockAlert).toHaveBeenCalledWith(expect.objectContaining({
            message: 'name already taken',
        })));
    });

    it('navigates to the rule page for test-set generation using source_rule_id', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ source_rule_id: 321 })] } });
        renderPage();
        await screen.findByText('Rule Alpha');
        fireEvent.click(screen.getByRole('button', { name: "Open this rule's page" }));
        expect(mockNavigate).toHaveBeenCalledWith('/rules/321');
    });
});

describe('RulesManager — navigation & action bar', () => {
    it('navigates via the breadcrumb crumbs (Hub, Rule Sets)', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getByText('Hub'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
        fireEvent.click(screen.getByText('Rule Sets'));
        expect(mockNavigate).toHaveBeenCalledWith('/guardrails');
    });

    it('"Create a New Rule" card opens the Create chooser', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Create a New Rule')[0]);
        expect(await screen.findByText('What do you want to create?')).toBeInTheDocument();
    });

    it('shows Evaluate/Monitor + Download for an active (trained) classifier', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ custom_name: 'Same' })] } });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [1], trained_rule_names: ['Same'] },
        });
        api.getTrainingStatus.mockResolvedValue({ data: { status: 'active', is_training: false } });
        renderPage();
        expect(await screen.findByRole('button', { name: /Evaluate/ })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Monitor/ })).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: /Evaluate/ }));
        expect(mockNavigate).toHaveBeenCalledWith('/classifiers/5/evaluate');
        fireEvent.click(screen.getByRole('button', { name: /Monitor/ }));
        expect(mockNavigate).toHaveBeenCalledWith('/classifiers/5/monitor');
        // Download button present for active classifiers.
        expect(screen.getByRole('button', { name: /Download/ })).toBeInTheDocument();
    });

    it('triggers downloadClassifier on the download button', async () => {
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture({ custom_name: 'Same' })] } });
        api.getClassifierDetails.mockResolvedValue({
            data: { model_id: 1, model_name: 'M', name: 'C', trained_rule_setup_ids: [1], trained_rule_names: ['Same'] },
        });
        api.getTrainingStatus.mockResolvedValue({ data: { status: 'active', is_training: false } });
        renderPage();
        const dl = await screen.findByRole('button', { name: /Download/ });
        fireEvent.click(dl);
        await waitFor(() => expect(api.downloadClassifier).toHaveBeenCalledWith('5', 'C'));
    });

    it('hides Evaluate/Monitor/Download for an untrained classifier', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        expect(screen.queryByRole('button', { name: /Evaluate/ })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Download/ })).not.toBeInTheDocument();
    });
});

describe('RulesManager — library refresh event', () => {
    it('refetches rules + bookmarks when gavel:libraryChanged fires', async () => {
        renderPage();
        await waitFor(() => expect(api.getClassifierRules).toHaveBeenCalledTimes(1));
        window.dispatchEvent(new Event('gavel:libraryChanged'));
        await waitFor(() => expect(api.getClassifierRules).toHaveBeenCalledTimes(2));
    });
});

describe('RulesManager — post-training chain banner', () => {
    // Training's own banner is gated on status === 'training', so once the run
    // finished the page went silent while calibration ran for minutes. The same
    // banner now carries the chain stage; the Calibration/Evaluation tabs are
    // untouched — this only says which stage is running right now.
    it('shows "Calibrating" with its live detail after training completes', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'active', is_training: false, is_trained: true,
                training_phase: null, training_phase_detail: null,
                post_training_phase: 'Calibrating',
                post_training_phase_detail: 'Loading calibration datasets…',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Calibrating')).toBeInTheDocument();
        expect(screen.getByText(/Loading calibration datasets/)).toBeInTheDocument();
    });

    it('shows "Evaluating" for the second stage', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'active', is_training: false, is_trained: true,
                training_phase: null, training_phase_detail: null,
                post_training_phase: 'Evaluating',
                post_training_phase_detail: 'Scoring rule 1 of 3…',
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        expect(await screen.findByText('Evaluating')).toBeInTheDocument();
        expect(screen.getByText(/Scoring rule 1 of 3/)).toBeInTheDocument();
    });

    it('shows no banner once the chain is done', async () => {
        api.getTrainingStatus.mockResolvedValue({
            data: {
                status: 'active', is_training: false, is_trained: true,
                training_phase: null, training_phase_detail: null,
                post_training_phase: null, post_training_phase_detail: null,
            },
        });
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();
        await screen.findByText(ruleFixture().custom_name || ruleFixture().name);
        expect(screen.queryByText('Calibrating')).not.toBeInTheDocument();
        expect(screen.queryByText('Evaluating')).not.toBeInTheDocument();
    });
});

describe('RulesManager — empty states wait for the fetch (#11)', () => {
    // An empty array meant both "not fetched yet" and "fetched, nothing there",
    // so a slow backend rendered "No Rules Defined" and "No models yet" as if
    // they were answers — next to a "Loading..." breadcrumb saying otherwise.
    it('shows a loading state, not "No Rules Defined", while the fetch is in flight', async () => {
        let release;
        api.getClassifierRules.mockReturnValue(new Promise((resolve) => {
            release = () => resolve({ data: { rules: [] } });
        }));
        renderPage();

        expect(await screen.findByText(/Loading rules/i)).toBeInTheDocument();
        expect(screen.queryByText('No Rules Defined')).not.toBeInTheDocument();

        // Only once the response resolves EMPTY does the empty state appear.
        await act(async () => { release(); });
        expect(await screen.findByText('No Rules Defined')).toBeInTheDocument();
        expect(screen.queryByText(/Loading rules/i)).not.toBeInTheDocument();
    });

    it('still reaches the empty state when the request fails', async () => {
        // A rejected fetch must not leave the page stuck on "Loading rules…".
        api.getClassifierRules.mockRejectedValue(new Error('boom'));
        renderPage();
        expect(await screen.findByText('No Rules Defined')).toBeInTheDocument();
    });

    it('does not claim "No models yet" before the models call resolves', async () => {
        let release;
        api.getUserModels.mockReturnValue(new Promise((resolve) => {
            release = () => resolve({ data: { models: [] } });
        }));
        api.getClassifierRules.mockResolvedValue({ data: { rules: [ruleFixture()] } });
        renderPage();

        await screen.findByText(ruleFixture().custom_name || ruleFixture().name);
        expect(screen.queryByText(/No models yet — add one/i)).not.toBeInTheDocument();
        await act(async () => { release(); });
    });
});
