// Behavior tests for the Evaluation page (Evaluation.jsx).
//
// Evaluation drives the calibrate -> evaluate -> results flow for a single
// classifier. On mount it fans out four parallel API calls (details, results,
// test datasets, calibration-data status) plus a follow-up thresholds fetch,
// then renders two tabs:
//   * Calibration — per-CE readiness chips, Run Calibration, threshold table,
//     calibration-curve plot.
//   * Evaluation  — single-test-set-per-rule run button, weighted averages,
//     per-use-case TPR/FPR/Accuracy/F1, ROC/PR AUC table, ROC plot.
//
// These tests mock the api module, the router hooks (useParams/useNavigate),
// and stub the Sidebar (Layout renders it; it has its own fetches). The real
// TutorialProvider wraps the render so useTutorialContent has a context.
// window.alert is stubbed so interaction guards can be asserted.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TutorialProvider } from '../../src/contexts/TutorialContext';

// ---- router hooks ----
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual,
        useNavigate: () => mockNavigate,
        useParams: () => ({ classifierId: '77' }),
    };
});

// ---- API mock: every export the page touches, benign defaults ----
vi.mock('../../src/api', () => {
    const ok = (data = {}) => Promise.resolve({ data });
    return {
        default: { get: vi.fn(() => ok()), post: vi.fn(() => ok()) },
        getClassifierDetails: vi.fn(() => ok({ name: 'Test Classifier', model_name: 'Llama-Guard' })),
        getEvaluationResults: vi.fn(() => ok({ calibration: null, evaluation: null })),
        startCalibration: vi.fn(() => ok({})),
        startEvaluation: vi.fn(() => ok({})),
        getCalibratedThresholds: vi.fn(() => ok({ thresholds: null })),
        listClassifierTestDatasets: vi.fn(() => ok({ datasets: [] })),
        getCalibrationDataStatus: vi.fn(() => ok({ all_ready: true, ces: [] })),
    };
});

// ---- Sidebar stub (Layout renders it; irrelevant to this page) ----
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

import Evaluation from '../../src/pages/Evaluation';
import * as api from '../../src/api';

const renderPage = () =>
    render(
        <TutorialProvider>
            <MemoryRouter initialEntries={['/classifiers/77/evaluation']}>
                <Evaluation />
            </MemoryRouter>
        </TutorialProvider>,
    );

// Tabs and the Run buttons share words ("Calibration"/"Evaluation"), so the
// tab buttons are matched by their exact accessible name. The inline page-help
// card also has a collapsible toggle whose accessible name is the page title
// ("Evaluation"); it carries aria-expanded, so we exclude it to land on the tab.
const getTabButton = (name) =>
    screen.getAllByRole('button', { name }).find((b) => !b.hasAttribute('aria-expanded'));
const clickTab = (name) => fireEvent.click(getTabButton(name));

// --- fixtures ---
const ok = (data = {}) => Promise.resolve({ data });

const dataset = (over = {}) => ({
    dataset_id: 1,
    scenario_name: 'Scenario A',
    dataset_type: 'positive',
    status: 'ready',
    created_at: '2026-01-01T00:00:00Z',
    ...over,
});

// A complete scenario = one positive + one negative half, both ready.
const completeScenario = (name, posId, negId) => [
    dataset({ dataset_id: posId, scenario_name: name, dataset_type: 'positive' }),
    dataset({ dataset_id: negId, scenario_name: name, dataset_type: 'negative' }),
];

const evalMetrics = (over = {}) => ({
    eval_type: 'evaluation',
    created_at: '2026-02-02T10:00:00Z',
    metrics: {
        weighted_averages: { Precision: 0.92, Recall: 0.81, F1: 0.86 },
        metrics: [
            { Usecase: 'use-case-1', TPR: 0.9, FPR: 0.05, Accuracy: 0.88, F1: 0.87, Support_Pos: 10, Support_Neg: 12 },
        ],
        auc: [
            { Usecase: 'auc-case-1', ROC_AUC: 0.95, PR_AUC: 0.93, num_pos: 10, num_neg: 12 },
        ],
        split_counts: { positive: 10, negative: 12 },
    },
    ...over,
});

const calibrationResult = (over = {}) => ({
    eval_type: 'calibration',
    created_at: '2026-02-01T10:00:00Z',
    thresholds: { 'CE-A': {} },
    ...over,
});

beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'alert').mockImplementation(() => {});
    // Reset to benign defaults each test.
    api.getClassifierDetails.mockResolvedValue({ data: { name: 'Test Classifier', model_name: 'Llama-Guard' } });
    api.getEvaluationResults.mockResolvedValue({ data: { calibration: null, evaluation: null } });
    api.listClassifierTestDatasets.mockResolvedValue({ data: { datasets: [] } });
    api.getCalibrationDataStatus.mockResolvedValue({ data: { all_ready: true, ces: [] } });
    api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: null } });
    api.startCalibration.mockResolvedValue({ data: {} });
    api.startEvaluation.mockResolvedValue({ data: {} });
});

afterEach(() => {
    vi.restoreAllMocks();
});

describe('Evaluation — mount & loading', () => {
    it('shows the loading state before data resolves, then clears it', async () => {
        let resolve;
        api.getClassifierDetails.mockReturnValue(new Promise((r) => { resolve = r; }));
        renderPage();
        expect(screen.getByText('Loading...')).toBeInTheDocument();
        resolve({ data: { name: 'Test Classifier' } });
        await waitFor(() => expect(screen.queryByText('Loading...')).not.toBeInTheDocument());
    });

    it('fetches all four sources for the classifierId param on mount', async () => {
        renderPage();
        await waitFor(() => expect(api.getClassifierDetails).toHaveBeenCalledWith('77'));
        expect(api.getEvaluationResults).toHaveBeenCalledWith('77');
        expect(api.listClassifierTestDatasets).toHaveBeenCalledWith('77');
        expect(api.getCalibrationDataStatus).toHaveBeenCalledWith('77');
    });

    it('renders the header with the classifier name and the two tabs', async () => {
        renderPage();
        expect(await screen.findByText('Evaluate: Test Classifier')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Calibration' })).toBeInTheDocument();
        expect(getTabButton('Evaluation')).toBeInTheDocument();
    });

    it('keeps rendering when the mount fetch rejects (error path clears loading)', async () => {
        api.getClassifierDetails.mockRejectedValue(new Error('boom'));
        renderPage();
        // Loading clears in the finally block even though details failed.
        await waitFor(() => expect(screen.queryByText('Loading...')).not.toBeInTheDocument());
        // Calibration tab default content still renders.
        expect(screen.getByText('Threshold Calibration')).toBeInTheDocument();
    });
});

describe('Evaluation — header navigation', () => {
    it('rule set breadcrumb navigates to the classifier rules page', async () => {
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        // The middle crumb is now the rule set's own name (from the details fetch).
        fireEvent.click(screen.getByText('Test Classifier'));
        expect(mockNavigate).toHaveBeenCalledWith('/classifiers/77/rules');
    });
});

describe('Evaluation — Calibration tab', () => {
    it('renders per-CE readiness chips (Ready / Missing)', async () => {
        api.getCalibrationDataStatus.mockResolvedValue({
            data: {
                all_ready: false,
                ces: [
                    { ce_id: 1, name: 'CE-One', has_calibration: true },
                    { ce_id: 2, name: 'CE-Two', has_calibration: false },
                ],
            },
        });
        renderPage();
        expect(await screen.findByText('CE-One: Ready')).toBeInTheDocument();
        expect(screen.getByText('CE-Two: Missing')).toBeInTheDocument();
        // The "some CEs missing" warning appears because all_ready is false.
        expect(screen.getByText(/Some CEs are missing calibration data/)).toBeInTheDocument();
    });

    it('disables Run Calibration when calibration data is not all ready', async () => {
        api.getCalibrationDataStatus.mockResolvedValue({
            data: { all_ready: false, ces: [{ ce_id: 1, name: 'CE-One', has_calibration: false }] },
        });
        renderPage();
        const btn = await screen.findByText('Run Calibration');
        expect(btn.closest('button')).toBeDisabled();
    });

    it('runs calibration and shows the running state when all_ready', async () => {
        renderPage();
        const btn = await screen.findByText('Run Calibration');
        fireEvent.click(btn.closest('button'));
        // force:false — this is a first calibration, not a manual recalibrate.
        await waitFor(() => expect(api.startCalibration).toHaveBeenCalledWith('77', { force: false }));
        // Button label flips to "Calibrating..." while the run is in flight.
        expect(await screen.findByText('Calibrating...')).toBeInTheDocument();
    });

    it('alerts and clears the running state when starting calibration fails', async () => {
        api.startCalibration.mockRejectedValue({ response: { data: { detail: 'no calib data' } } });
        renderPage();
        const btn = await screen.findByText('Run Calibration');
        fireEvent.click(btn.closest('button'));
        await waitFor(() => expect(window.alert).toHaveBeenCalledWith('no calib data'));
        // Label returns to "Run Calibration" after the failure.
        expect(await screen.findByText('Run Calibration')).toBeInTheDocument();
    });

    it('renders the calibrated-thresholds table with FPR/TPR/Youden metrics', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: calibrationResult(), evaluation: null },
        });
        api.getCalibratedThresholds.mockResolvedValue({
            data: {
                thresholds: {
                    'CE-Alpha': { threshold: 0.512, patience: 3, youden_j: 0.8421, tpr_at_optimal: 0.91, fpr_at_optimal: 0.07 },
                },
            },
        });
        renderPage();
        expect(await screen.findByText('Calibrated Thresholds')).toBeInTheDocument();
        expect(screen.getByText('CE-Alpha')).toBeInTheDocument();
        expect(screen.getByText('0.512')).toBeInTheDocument();
        expect(screen.getByText('0.8421')).toBeInTheDocument();
        expect(screen.getByText('0.910')).toBeInTheDocument(); // TPR
        expect(screen.getByText('0.070')).toBeInTheDocument(); // FPR
    });

    it('renders the calibration-curve plot when a mosaic image is present', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: calibrationResult({ plots: { mosaic: 'AAAA' } }), evaluation: null },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { 'CE-A': { threshold: 0.5, patience: 1 } } } });
        renderPage();
        const img = await screen.findByAltText('Calibration curves');
        expect(img).toHaveAttribute('src', 'data:image/png;base64,AAAA');
    });

    it('shows the error badge and message on a calibration_error result', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: { eval_type: 'calibration_error', metrics: { error: 'calibration blew up' }, created_at: '2026-02-01T00:00:00Z' },
                evaluation: null,
            },
        });
        renderPage();
        expect(await screen.findByText('calibration blew up')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
    });
});

describe('Evaluation — Evaluation tab: single test set per rule', () => {
    it('disables Run Evaluation and warns when no test sets are ready', async () => {
        // Default mock returns no ready datasets. Calibrated, so the ONLY
        // blocker we're testing is the missing test sets.
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { topicA: { threshold: 0.5, patience: 1 } } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText(/No test sets ready yet/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeDisabled();
    });

    it('enables Run Evaluation once the rules have a ready test set (no picker)', async () => {
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { topicA: { threshold: 0.5, patience: 1 } } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const runBtn = await screen.findByRole('button', { name: 'Run Evaluation' });
        expect(runBtn).not.toBeDisabled();
        // The old per-scenario picker is gone.
        expect(screen.queryByText(/Test scenarios/i)).not.toBeInTheDocument();
        expect(screen.queryByText('Scenario A')).not.toBeInTheDocument();
    });

    it('locks evaluation until the classifier is calibrated (no thresholds yet)', async () => {
        // Test sets are ready but there are no calibrated thresholds — eval must
        // be blocked with a "calibrate first" message.
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: null } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const runBtn = await screen.findByRole('button', { name: /Calibrate first/i });
        expect(runBtn).toBeDisabled();
        expect(screen.getByText(/Not calibrated yet/i)).toBeInTheDocument();
    });

    it('keeps evaluation locked while calibration is still running (even with ready test sets)', async () => {
        // Test sets are ready, so the only thing blocking eval is the
        // in-progress calibration — eval must not be clickable until it finishes.
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: { eval_type: 'calibration_running', created_at: '2026-02-01T00:00:00Z' },
                evaluation: null,
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        // Button is disabled and relabeled; a warning explains why.
        const runBtn = await screen.findByRole('button', { name: /Calibration running/i });
        expect(runBtn).toBeDisabled();
        expect(screen.getByText(/evaluation unlocks when it finishes/i)).toBeInTheDocument();
    });
});

describe('Evaluation — running an evaluation', () => {
    it('starts evaluation with NO dataset ids (backend auto-uses each rule\'s test set)', async () => {
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { topicA: { threshold: 0.5, patience: 1 } } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const runBtn = await screen.findByRole('button', { name: 'Run Evaluation' });
        fireEvent.click(runBtn);
        await waitFor(() => expect(api.startEvaluation).toHaveBeenCalledTimes(1));
        const [id, payload] = api.startEvaluation.mock.calls[0];
        expect(id).toBe('77');
        // No selection is sent — the backend resolves every active rule's set.
        expect(payload.test_dataset_ids).toBeUndefined();
        expect(await screen.findByText('Evaluating...')).toBeInTheDocument();
    });

    it('alerts and clears running when starting the evaluation fails', async () => {
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { topicA: { threshold: 0.5, patience: 1 } } } });
        api.startEvaluation.mockRejectedValue({ response: { data: { detail: 'eval failed' } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const runBtn = await screen.findByRole('button', { name: 'Run Evaluation' });
        fireEvent.click(runBtn);
        await waitFor(() => expect(window.alert).toHaveBeenCalledWith('eval failed'));
        expect(await screen.findByRole('button', { name: 'Run Evaluation' })).toBeInTheDocument();
    });
});

describe('Evaluation — Evaluation tab: metrics display', () => {
    beforeEach(() => {
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: null, evaluation: evalMetrics() },
        });
    });

    it('renders weighted averages as percentages', async () => {
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('Weighted Averages')).toBeInTheDocument();
        expect(screen.getByText('92.0%')).toBeInTheDocument(); // Precision
        expect(screen.getByText('81.0%')).toBeInTheDocument(); // Recall
    });

    it('renders the per-use-case metrics table (TPR/FPR/Accuracy/F1)', async () => {
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('Per Use-Case Metrics')).toBeInTheDocument();
        expect(screen.getByText('use-case-1')).toBeInTheDocument();
        expect(screen.getByText('0.900')).toBeInTheDocument(); // TPR
        expect(screen.getByText('0.050')).toBeInTheDocument(); // FPR
    });

    it('renders the AUC (ROC/PR) scores table', async () => {
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('AUC Scores')).toBeInTheDocument();
        expect(screen.getByText('0.950')).toBeInTheDocument(); // ROC AUC
        expect(screen.getByText('0.930')).toBeInTheDocument(); // PR AUC
    });

    it('hides dead neutral rows with zero positive AND negative support', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: null,
                evaluation: evalMetrics({
                    metrics: {
                        weighted_averages: { F1: 0.5 },
                        metrics: [
                            { Usecase: 'real', TPR: 0.8, FPR: 0.1, Accuracy: 0.8, F1: 0.8, Support_Pos: 5, Support_Neg: 5 },
                            { Usecase: 'conversational', TPR: 0, FPR: 0, Accuracy: 0, F1: 0, Support_Pos: 0, Support_Neg: 0 },
                        ],
                        auc: [],
                    },
                }),
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('real')).toBeInTheDocument();
        // The empty neutral pseudo-use-case row is dropped.
        expect(screen.queryByText('conversational')).not.toBeInTheDocument();
    });

    it('puts neutral pseudo-use-cases in their own Neutral Use Cases table, not Per Use-Case', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: null,
                evaluation: evalMetrics({
                    metrics: {
                        weighted_averages: { F1: 0.5 },
                        metrics: [
                            { Usecase: 'real_rule', TPR: 0.8, FPR: 0.1, Accuracy: 0.8, F1: 0.8, Support_Pos: 5, Support_Neg: 5 },
                            { Usecase: 'conversational', TPR: 0, FPR: 0.83, Accuracy: 0.17, F1: 0, Support_Pos: 0, Support_Neg: 500 },
                            { Usecase: 'instructive', TPR: 0, FPR: 0.996, Accuracy: 0.004, F1: 0, Support_Pos: 0, Support_Neg: 500 },
                        ],
                        auc: [
                            { Usecase: 'real_rule', ROC_AUC: 0.9, PR_AUC: 0.9, num_pos: 5, num_neg: 5 },
                            { Usecase: 'conversational', ROC_AUC: null, PR_AUC: null, num_pos: 0, num_neg: 500 },
                        ],
                    },
                }),
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');

        // The dedicated neutral table renders the two pseudo-use-cases.
        expect(await screen.findByText('Neutral Use Cases')).toBeInTheDocument();
        expect(screen.getByText('conversational')).toBeInTheDocument();
        expect(screen.getByText('instructive')).toBeInTheDocument();
        // Support shown as P/N.
        expect(screen.getAllByText('0/500').length).toBe(2);

        // The Per Use-Case table shows the real rule but NOT the neutral rows.
        const perUseCase = screen.getByText('Per Use-Case Metrics').closest('div');
        expect(within(perUseCase).getByText('real_rule')).toBeInTheDocument();
        expect(within(perUseCase).queryByText('conversational')).not.toBeInTheDocument();

        // AUC table excludes the neutral row (no positives → no AUC).
        const aucTable = screen.getByText('AUC Scores').closest('div');
        expect(within(aucTable).queryByText('conversational')).not.toBeInTheDocument();
    });

    it('renders the ROC plot when roc_all_usecases image is present', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: null,
                evaluation: evalMetrics({ plots: { roc_all_usecases: 'ZZZZ' } }),
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const img = await screen.findByAltText('ROC curves');
        expect(img).toHaveAttribute('src', 'data:image/png;base64,ZZZZ');
    });

    it('shows the evaluation_error badge and message', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: null,
                evaluation: { eval_type: 'evaluation_error', metrics: { error: 'eval crashed' }, created_at: '2026-02-02T00:00:00Z' },
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('eval crashed')).toBeInTheDocument();
    });

    it('shows the calibrated-thresholds hint when thresholds are loaded (not yet evaluated)', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: calibrationResult(), evaluation: null },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { 'CE-A': { threshold: 0.5, patience: 1 } } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText(/Using calibrated thresholds/)).toBeInTheDocument();
    });

    it('locks evaluation once a successful evaluation exists (retrain to re-run)', async () => {
        api.listClassifierTestDatasets.mockResolvedValue({
            data: { datasets: completeScenario('Scenario A', 1, 2) },
        });
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: calibrationResult(), evaluation: evalMetrics() },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        const btn = await screen.findByRole('button', { name: /Evaluated/ });
        expect(btn).toBeDisabled();
        expect(screen.getByText(/retrain to re-evaluate/i)).toBeInTheDocument();
    });
});

describe('Evaluation — once-per-training lock', () => {
    // Calibration is NOT locked once it has run: training auto-calibrates, so the
    // button's remaining job is the data-changed case (regenerated CE calibration
    // sets), which needs force:true. Evaluation stays locked — see above.
    it('offers Recalibrate once a successful calibration exists', async () => {
        api.getCalibrationDataStatus.mockResolvedValue({ data: { all_ready: true, ces: [{ ce_id: 1, name: 'CE-A', has_calibration: true }] } });
        api.getEvaluationResults.mockResolvedValue({
            data: { calibration: calibrationResult(), evaluation: null },
        });
        api.getCalibratedThresholds.mockResolvedValue({ data: { thresholds: { 'CE-A': { threshold: 0.5, patience: 1 } } } });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        const btn = await screen.findByRole('button', { name: /Recalibrate/ });
        expect(btn).not.toBeDisabled();
        expect(screen.getByText(/retraining recalibrates automatically/i)).toBeInTheDocument();

        fireEvent.click(btn);
        await waitFor(() => expect(api.startCalibration).toHaveBeenCalledWith('77', { force: true }));
    });
});

describe('Evaluation — resume polling on refresh', () => {
    it('resumes the running state when a calibration_running row is present', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: { eval_type: 'calibration_running', created_at: '2026-02-01T00:00:00Z' },
                evaluation: null,
            },
        });
        renderPage();
        // The Run Calibration button reads "Calibrating..." because the page
        // detected an in-flight run on mount.
        expect(await screen.findByText('Calibrating...')).toBeInTheDocument();
    });

    it('resumes the evaluating state when an evaluation_running row is present', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: null,
                evaluation: { eval_type: 'evaluation_running', created_at: '2026-02-02T00:00:00Z' },
            },
        });
        renderPage();
        await screen.findByText('Evaluate: Test Classifier');
        clickTab('Evaluation');
        expect(await screen.findByText('Evaluating...')).toBeInTheDocument();
    });

    it('shows the live phase published on the calibration_running row', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: {
                    eval_type: 'calibration_running',
                    created_at: '2026-02-01T00:00:00Z',
                    metrics: { phase: 'Running on the cluster GPU…' },
                },
                evaluation: null,
            },
        });
        renderPage();
        // The phase line surfaces the current stage so the user isn't left
        // guessing why it's taking a while.
        expect(await screen.findByText('Running on the cluster GPU…')).toBeInTheDocument();
    });

    it('falls back to a generic phase line when no phase is published yet', async () => {
        api.getEvaluationResults.mockResolvedValue({
            data: {
                calibration: { eval_type: 'calibration_running', created_at: '2026-02-01T00:00:00Z' },
                evaluation: null,
            },
        });
        renderPage();
        expect(await screen.findByText('Starting calibration…')).toBeInTheDocument();
    });
});
