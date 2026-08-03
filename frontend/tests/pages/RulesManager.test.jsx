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
    downloadClassifier: vi.fn(() => Promise.resolve()),
    listLocalDrafts: vi.fn(() => ok({ rules: [] })),
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
    api.getRuleBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.getCEBookmarks.mockResolvedValue({ data: { bookmarks: [] } });
    api.listLocalDrafts.mockResolvedValue({ data: { rules: [] } });
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

    it('shows an empty-list message when no bookmarks or drafts exist', async () => {
        renderPage();
        await screen.findByText('No Rules Defined');
        fireEvent.click(screen.getAllByText('Add an Existing Rule')[0]);
        expect(await screen.findByText(/No rules in your Library yet/)).toBeInTheDocument();
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
        expect(await screen.findByText(/No rules in your Library yet/)).toBeInTheDocument();
        expect(screen.queryByText('Already Attached')).not.toBeInTheDocument();
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
