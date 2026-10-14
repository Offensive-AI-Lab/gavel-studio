// Step 2A shows the rule proposal: one rule (the original shape) or several.
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';

const api = vi.hoisted(() => ({
    generateGavelPipeline: vi.fn(),
    updatePipelineLinks: vi.fn(),
    discardPipelineResources: vi.fn(),
}));
vi.mock('../../src/api', () => api);
vi.mock('../../src/components/InlineHelp/InlineHelp', () => ({ default: () => null }));

import Step2ARule, { proposalRules } from '../../src/pages/RuleWizardSteps/Step2ARule';

const ONE = {
    success: true, rule_id: 500, name: 'single_rule', description: 'detects x',
    groups: { ctx: ['a'], act: ['b'] }, condition: 'all of ctx and 1 of act', predicate: 'a AND b',
    categories: [], new_ces: [{ ce_id: 1, ce_name: 'b', definition: 'def b' }],
    rules: [{ rule_id: 500, name: 'single_rule', description: 'detects x',
        groups: { ctx: ['a'], act: ['b'] }, condition: 'all of ctx and 1 of act', categories: [] }],
};
const TWO = {
    ...ONE, rule_id: 601, name: 'doctor_rule',
    new_ces: [{ ce_id: 1, ce_name: 'b', definition: 'def b' }, { ce_id: 2, ce_name: 'c', definition: 'def c' }],
    rules: [
        { rule_id: 601, name: 'doctor_rule', description: 'doctor misuse', groups: { g1: ['a', 'b'] }, condition: 'all of g1', categories: [] },
        { rule_id: 602, name: 'bank_rule', description: 'bank misuse', groups: { g2: ['a', 'c'] }, condition: 'all of g2', categories: [] },
    ],
};

const run = { run_id: 9, steps: { '1': { data: { description: 'scenario text' } } } };
const patch = vi.fn(async () => ({}));

function mount(onAdvance = vi.fn()) {
    return render(<Step2ARule run={run} classifierId={null} onPatchStep={patch} setRun={vi.fn()} onAdvance={onAdvance} />);
}

beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.setItem('user', JSON.stringify({ user_id: 1 }));
    api.updatePipelineLinks.mockResolvedValue({ data: null });
    api.discardPipelineResources.mockResolvedValue({});
});

describe('proposalRules', () => {
    it('returns the rules list when the response has one', () => {
        expect(proposalRules(TWO).map(r => r.name)).toEqual(['doctor_rule', 'bank_rule']);
    });
    it('turns an older single-rule response (no rules list) into one rule', () => {
        const old = { rule_id: 7, name: 'old', groups: {}, condition: '', predicate: 'p' };
        expect(proposalRules(old)).toEqual([expect.objectContaining({ rule_id: 7, name: 'old', predicate: 'p' })]);
    });
    it('handles no proposal', () => {
        expect(proposalRules(null)).toEqual([]);
    });
});

describe('Step2ARule', () => {
    it('shows one rule exactly as before for a single-rule answer', async () => {
        api.generateGavelPipeline.mockResolvedValue({ data: ONE });
        mount();
        fireEvent.click(screen.getByText(/Generate Rule/));
        await screen.findByText('single_rule');
        expect(screen.queryByText(/Rule 1 of/)).toBeNull();
        expect(screen.getByText(/Rule proposal generated/)).toBeTruthy();
    });

    it('shows every rule when the answer holds several', async () => {
        api.generateGavelPipeline.mockResolvedValue({ data: TWO });
        mount();
        fireEvent.click(screen.getByText(/Generate Rule/));
        await screen.findByText('doctor_rule');
        expect(screen.getByText('bank_rule')).toBeTruthy();
        expect(screen.getByText('Rule 1 of 2')).toBeTruthy();
        expect(screen.getByText('Rule 2 of 2')).toBeTruthy();
        expect(screen.getByText(/2 rules proposed/)).toBeTruthy();
        // the shared new CEs are listed once
        expect(screen.getByText('def b')).toBeTruthy();
        expect(screen.getByText('def c')).toBeTruthy();
    });

    it('links the first rule to the run and remembers every rule id', async () => {
        api.generateGavelPipeline.mockResolvedValue({ data: TWO });
        mount();
        fireEvent.click(screen.getByText(/Generate Rule/));
        await screen.findByText('bank_rule');
        expect(api.updatePipelineLinks).toHaveBeenCalledWith(9, { ruleId: 601 });
        const completed = patch.mock.calls.find(c => c[1]?.status === 'completed');
        expect(completed[1].data.rule_ids).toEqual([601, 602]);
        expect(completed[1].data.rule_id).toBe(601);
    });

    it('Discard removes every rule and CE of the proposal', async () => {
        api.generateGavelPipeline.mockResolvedValue({ data: TWO });
        mount();
        fireEvent.click(screen.getByText(/Generate Rule/));
        await screen.findByText('bank_rule');
        fireEvent.click(screen.getByText(/Discard/));
        await waitFor(() => expect(api.discardPipelineResources).toHaveBeenCalledWith([1, 2], 601, [601, 602]));
    });
});
