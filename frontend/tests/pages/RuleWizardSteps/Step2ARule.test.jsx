// Behavior tests for Step2ARule — the rule-generation step of the wizard.
//
// The component is self-contained: it reads scenario.description from the
// run's step-1 data, calls generateGavelPipeline, then links + persists the
// proposal through onPatchStep. We mock '../../api' so nothing hits the
// network and drive every branch (no-scenario gate, missing user, generate
// success, generate error, link failure, discard, and proposal rendering
// variants).

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// --- Mock the API module. Every export Step2ARule touches resolves to a
// benign default; individual tests override via mockResolvedValue etc.
vi.mock('../../../src/api', () => ({
    generateGavelPipeline: vi.fn(() => Promise.resolve({ data: {} })),
    updatePipelineLinks: vi.fn(() => Promise.resolve({ data: {} })),
    discardPipelineResources: vi.fn(() => Promise.resolve({ data: {} })),
}));

import Step2ARule from '../../../src/pages/RuleWizardSteps/Step2ARule';
import {
    generateGavelPipeline,
    updatePipelineLinks,
    discardPipelineResources,
} from '../../../src/api';

const setUser = (user = { user_id: 7, email: 'a@b.c' }) => {
    if (user === null) sessionStorage.setItem('user', 'null');
    else sessionStorage.setItem('user', JSON.stringify(user));
};

// A run whose step 1 produced a scenario description.
const runWithScenario = (overrides = {}) => {
    const { steps: extraSteps, ...rest } = overrides;
    return {
        run_id: 99,
        steps: {
            '1': { status: 'completed', data: { description: 'A worker scenario' } },
            ...(extraSteps || {}),
        },
        ...rest,
    };
};

// A complete proposal payload as returned by generateGavelPipeline
// (v2 shape: named CE groups + a firing condition over them).
const fullProposal = {
    rule_id: 55,
    name: 'My Rule',
    predicate: 'A AND B',
    categories: ['Safety', 'Harm'],
    groups: {
        required: ['ce_nec'],
        option_1: ['ce_f1', 'ce_f2'],
        supporting: ['ce_suf'],
    },
    condition: 'all of required and 1 of option_1',
    new_ces: [
        { ce_id: 1, ce_name: 'New CE One', definition: 'def one' },
        { ce_id: 2, ce_name: 'New CE Two', definition: 'def two' },
    ],
};

const renderStep = (props = {}) =>
    render(
        <Step2ARule
            run={runWithScenario()}
            classifierId={null}
            onPatchStep={vi.fn(() => Promise.resolve())}
            setRun={vi.fn()}
            {...props}
        />,
    );

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    setUser();
    generateGavelPipeline.mockResolvedValue({ data: {} });
    updatePipelineLinks.mockResolvedValue({ data: {} });
    discardPipelineResources.mockResolvedValue({ data: {} });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
});

describe('Step2ARule — gate when no scenario', () => {
    it('renders the "no scenario yet" banner when step 1 has no description', () => {
        const run = { run_id: 1, steps: { '1': { data: {} } } };
        renderStep({ run });
        expect(
            screen.getByText(/Step 1 hasn't produced a scenario yet/i),
        ).toBeInTheDocument();
        // The generate UI must NOT appear.
        expect(screen.queryByText(/Generate Rule/i)).not.toBeInTheDocument();
    });

    it('treats a whitespace-only description as no scenario', () => {
        const run = { run_id: 1, steps: { '1': { data: { description: '   ' } } } };
        renderStep({ run });
        expect(
            screen.getByText(/Step 1 hasn't produced a scenario yet/i),
        ).toBeInTheDocument();
    });

    it('handles a completely undefined run without throwing', () => {
        renderStep({ run: undefined });
        expect(
            screen.getByText(/Step 1 hasn't produced a scenario yet/i),
        ).toBeInTheDocument();
    });
});

describe('Step2ARule — initial render with scenario', () => {
    it('shows the scenario text and the Generate button', () => {
        renderStep();
        expect(screen.getByText('Scenario from step 1')).toBeInTheDocument();
        expect(screen.getByText('A worker scenario')).toBeInTheDocument();
        expect(screen.getByText(/Generate Rule/i)).toBeInTheDocument();
    });

    it('rehydrates an existing proposal from persisted step state', () => {
        const run = runWithScenario({
            steps: {
                '2A': { status: 'completed', data: { proposal: fullProposal } },
            },
        });
        renderStep({ run });
        // Proposal section is shown, generate button hidden.
        expect(screen.getByText('My Rule')).toBeInTheDocument();
        expect(screen.queryByText(/Generate Rule/i)).not.toBeInTheDocument();
        expect(
            screen.getByText(/Rule proposal generated/i),
        ).toBeInTheDocument();
    });
});

describe('Step2ARule — generate flow', () => {
    it('blocks generation and shows an error when user context is missing', async () => {
        setUser(null);
        renderStep();
        fireEvent.click(screen.getByText(/Generate Rule/i));
        expect(
            await screen.findByText(/Missing user context/i),
        ).toBeInTheDocument();
        expect(generateGavelPipeline).not.toHaveBeenCalled();
    });

    it('handles malformed user JSON in localStorage as missing user', async () => {
        sessionStorage.setItem('user', '{not valid json');
        renderStep();
        fireEvent.click(screen.getByText(/Generate Rule/i));
        expect(
            await screen.findByText(/Missing user context/i),
        ).toBeInTheDocument();
        expect(generateGavelPipeline).not.toHaveBeenCalled();
    });

    it('calls generateGavelPipeline with description, user_id and null classifier (Pipeline A)', async () => {
        generateGavelPipeline.mockResolvedValue({ data: fullProposal });
        const onPatchStep = vi.fn(() => Promise.resolve());
        const setRun = vi.fn();
        renderStep({ onPatchStep, setRun });

        fireEvent.click(screen.getByText(/Generate Rule/i));

        await waitFor(() =>
            expect(generateGavelPipeline).toHaveBeenCalledWith(
                'A worker scenario',
                7,
                null,
            ),
        );
        // startStep then completeStep both flow through onPatchStep.
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith(
                '2A',
                expect.objectContaining({ status: 'in_progress' }),
            ),
        );
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith(
                '2A',
                expect.objectContaining({
                    status: 'completed',
                    data: expect.objectContaining({
                        rule_id: 55,
                        ce_ids: [1, 2],
                    }),
                }),
            ),
        );
    });

    it('parses a string classifierId into an int for generateGavelPipeline', async () => {
        generateGavelPipeline.mockResolvedValue({ data: fullProposal });
        renderStep({ classifierId: '42' });
        fireEvent.click(screen.getByText(/Generate Rule/i));
        await waitFor(() =>
            expect(generateGavelPipeline).toHaveBeenCalledWith(
                'A worker scenario',
                7,
                42,
            ),
        );
    });

    it('links the rule onto the run and pushes the updated run via setRun', async () => {
        generateGavelPipeline.mockResolvedValue({ data: fullProposal });
        updatePipelineLinks.mockResolvedValue({ data: { run_id: 99, linked: true } });
        const setRun = vi.fn();
        renderStep({ setRun });

        fireEvent.click(screen.getByText(/Generate Rule/i));

        await waitFor(() =>
            expect(updatePipelineLinks).toHaveBeenCalledWith(99, { ruleId: 55 }),
        );
        await waitFor(() =>
            expect(setRun).toHaveBeenCalledWith({ run_id: 99, linked: true }),
        );
    });

    it('does not call setRun when link response has no data', async () => {
        generateGavelPipeline.mockResolvedValue({ data: fullProposal });
        updatePipelineLinks.mockResolvedValue({});
        const setRun = vi.fn();
        renderStep({ setRun });
        fireEvent.click(screen.getByText(/Generate Rule/i));
        await waitFor(() => expect(updatePipelineLinks).toHaveBeenCalled());
        expect(setRun).not.toHaveBeenCalled();
    });

    it('swallows a link failure but still completes the step and renders the proposal', async () => {
        generateGavelPipeline.mockResolvedValue({ data: fullProposal });
        updatePipelineLinks.mockRejectedValue(new Error('link boom'));
        const onPatchStep = vi.fn(() => Promise.resolve());
        renderStep({ onPatchStep });

        fireEvent.click(screen.getByText(/Generate Rule/i));

        // Proposal still appears despite the link error.
        expect(await screen.findByText('My Rule')).toBeInTheDocument();
        expect(console.warn).toHaveBeenCalled();
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith(
                '2A',
                expect.objectContaining({ status: 'completed' }),
            ),
        );
    });

    it('passes ruleId null when the proposal has no rule_id', async () => {
        generateGavelPipeline.mockResolvedValue({
            data: { ...fullProposal, rule_id: undefined },
        });
        renderStep();
        fireEvent.click(screen.getByText(/Generate Rule/i));
        await waitFor(() =>
            expect(updatePipelineLinks).toHaveBeenCalledWith(99, { ruleId: null }),
        );
    });

    it('defaults ce_ids/new_ces to empty arrays when proposal omits new_ces', async () => {
        generateGavelPipeline.mockResolvedValue({
            data: { rule_id: 5, name: 'R', predicate: 'P' },
        });
        const onPatchStep = vi.fn(() => Promise.resolve());
        renderStep({ onPatchStep });
        fireEvent.click(screen.getByText(/Generate Rule/i));
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith(
                '2A',
                expect.objectContaining({
                    status: 'completed',
                    data: expect.objectContaining({ ce_ids: [], new_ces: [] }),
                }),
            ),
        );
    });
});

describe('Step2ARule — generate error handling', () => {
    it('shows the API detail message and marks the step errored', async () => {
        generateGavelPipeline.mockRejectedValue({
            response: { data: { detail: 'model refused' } },
        });
        const onPatchStep = vi.fn(() => Promise.resolve());
        renderStep({ onPatchStep });

        fireEvent.click(screen.getByText(/Generate Rule/i));

        expect(await screen.findByText('model refused')).toBeInTheDocument();
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith(
                '2A',
                expect.objectContaining({
                    status: 'error',
                    data: expect.objectContaining({ error: 'model refused' }),
                }),
            ),
        );
        // No proposal rendered.
        expect(screen.queryByText(/Rule proposal generated/i)).not.toBeInTheDocument();
    });

    it('falls back to err.message when no response detail is present', async () => {
        generateGavelPipeline.mockRejectedValue(new Error('network down'));
        renderStep();
        fireEvent.click(screen.getByText(/Generate Rule/i));
        expect(await screen.findByText('network down')).toBeInTheDocument();
    });

    it('re-enables the Generate button after an error (generating reset in finally)', async () => {
        generateGavelPipeline.mockRejectedValue(new Error('oops'));
        renderStep();
        const btn = screen.getByText(/Generate Rule/i).closest('button');
        fireEvent.click(btn);
        await screen.findByText('oops');
        await waitFor(() => expect(btn).not.toBeDisabled());
    });
});

describe('Step2ARule — proposal rendering variants', () => {
    it('renders name, groups, condition, categories, and new CEs', () => {
        const run = runWithScenario({
            steps: { '2A': { status: 'completed', data: { proposal: fullProposal } } },
        });
        renderStep({ run });

        expect(screen.getByText('My Rule')).toBeInTheDocument();
        expect(screen.getByText('Safety')).toBeInTheDocument();
        expect(screen.getByText('Harm')).toBeInTheDocument();
        // CE groups by name + members.
        expect(screen.getByText('required:')).toBeInTheDocument();
        expect(screen.getByText(/ce_nec/)).toBeInTheDocument();
        expect(screen.getByText('option_1:')).toBeInTheDocument();
        expect(screen.getByText(/ce_f1, ce_f2/)).toBeInTheDocument();
        expect(screen.getByText('supporting:')).toBeInTheDocument();
        // Condition verbatim.
        expect(screen.getByText('all of required and 1 of option_1')).toBeInTheDocument();
        // New CEs.
        expect(screen.getByText('New Cognitive Elements (2)')).toBeInTheDocument();
        expect(screen.getByText('New CE One')).toBeInTheDocument();
        expect(screen.getByText('def two')).toBeInTheDocument();
    });

    it('falls back to the predicate when the proposal has no groups', () => {
        const minimal = {
            rule_id: 9,
            name: 'Minimal',
            predicate: 'X',
            new_ces: [],
        };
        const run = runWithScenario({
            steps: { '2A': { status: 'completed', data: { proposal: minimal } } },
        });
        renderStep({ run });

        expect(screen.getByText('Minimal')).toBeInTheDocument();
        expect(screen.getByText('X')).toBeInTheDocument();
        expect(screen.queryByText('CE Groups')).not.toBeInTheDocument();
        // Empty new_ces => no "New Cognitive Elements" header.
        expect(screen.queryByText(/New Cognitive Elements/)).not.toBeInTheDocument();
    });
});

describe('Step2ARule — discard flow', () => {
    const renderWithProposal = (extraProps = {}) => {
        const run = runWithScenario({
            steps: { '2A': { status: 'completed', data: { proposal: fullProposal } } },
        });
        const onPatchStep = vi.fn(() => Promise.resolve());
        const utils = renderStep({ run, onPatchStep, ...extraProps });
        return { onPatchStep, ...utils };
    };

    it('discards the new CEs + rule and resets the step to pending', async () => {
        const { onPatchStep } = renderWithProposal();

        fireEvent.click(screen.getByText(/Discard \+ Re-generate/i));

        await waitFor(() =>
            expect(discardPipelineResources).toHaveBeenCalledWith([1, 2], 55),
        );
        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith('2A', {
                status: 'pending',
                data: {},
            }),
        );
        // Back to the generate UI.
        expect(await screen.findByText(/Generate Rule/i)).toBeInTheDocument();
        expect(screen.queryByText('My Rule')).not.toBeInTheDocument();
    });

    it('still resets the step even if discardPipelineResources rejects', async () => {
        discardPipelineResources.mockRejectedValue(new Error('cleanup failed'));
        const { onPatchStep } = renderWithProposal();

        fireEvent.click(screen.getByText(/Discard \+ Re-generate/i));

        await waitFor(() =>
            expect(onPatchStep).toHaveBeenCalledWith('2A', {
                status: 'pending',
                data: {},
            }),
        );
        expect(await screen.findByText(/Generate Rule/i)).toBeInTheDocument();
    });

    it('passes null ruleId and empty ce list to discard when proposal lacks them', async () => {
        const run = runWithScenario({
            steps: {
                '2A': {
                    status: 'completed',
                    data: { proposal: { name: 'NoIds', predicate: 'P' } },
                },
            },
        });
        renderStep({ run });
        fireEvent.click(screen.getByText(/Discard \+ Re-generate/i));
        await waitFor(() =>
            expect(discardPipelineResources).toHaveBeenCalledWith([], null),
        );
    });
});
