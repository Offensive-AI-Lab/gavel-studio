// Direct tests for the background "build everything after approval" jobs.
//
// These used to be covered indirectly through the page wizards' onFinish (now
// removed — the flows are modals). buildRuleInBackground / buildCeInBackground
// are the real logic both the modals and (previously) the pages delegate to, so
// testing them directly is the durable coverage.
//
// Strategy:
//   * mock '../../src/api' so the chained calls are observable + controllable
//   * keep the REAL runInTray but make sleep() instant so the poll loop flies
//   * pass a fake tray ({ start -> task with update/success/error }); the job
//     runs detached, so assertions wait via waitFor

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { waitFor } from '@testing-library/react';

vi.mock('../../src/hooks/runInTray', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, sleep: () => Promise.resolve() };
});

const api = vi.hoisted(() => ({
    generateCeTraining:    vi.fn(),
    generateCeCalibration: vi.fn(),
    embedResources:        vi.fn(),
    completePipelineRun:   vi.fn(),
    getRuleDefaultsStatus: vi.fn(),
}));
vi.mock('../../src/api', () => api);

import { buildRuleInBackground, buildCeInBackground } from '../../src/hooks/pipelineBuild';

let trayTask, tray;
beforeEach(() => {
    vi.clearAllMocks();
    trayTask = { update: vi.fn(), success: vi.fn(), error: vi.fn() };
    tray = { start: vi.fn(() => trayTask) };
    api.generateCeTraining.mockResolvedValue({ data: {} });
    api.generateCeCalibration.mockResolvedValue({ data: {} });
    api.embedResources.mockResolvedValue({ data: {} });
    api.completePipelineRun.mockResolvedValue({ data: {} });
    api.getRuleDefaultsStatus.mockResolvedValue({ data: { state: 'ready' } });
});

describe('buildRuleInBackground', () => {
    it('trains + calibrates each new CE (deferred), embeds, waits for the set, completes', async () => {
        const run = {
            run_id: 12, rule_id: 500,
            steps: {
                '1':  { data: { description: 'catch jailbreaks' } },
                '2A': { data: { new_ces: [
                    { ce_id: 11, ce_name: 'a', definition: 'd' },
                    { ce_id: 22, ce_name: 'b', definition: 'd' },
                ] } },
            },
        };
        expect(buildRuleInBackground(tray, { run, userId: 7 })).toBe(true);

        await waitFor(() => expect(api.generateCeTraining).toHaveBeenCalledTimes(2));
        // New CEs MUST be deferred so they don't show half-built.
        expect(api.generateCeTraining).toHaveBeenCalledWith(expect.objectContaining({ defer_ready: true }));
        await waitFor(() => expect(api.generateCeCalibration).toHaveBeenCalledTimes(2));
        await waitFor(() => expect(api.embedResources).toHaveBeenCalledWith(
            expect.objectContaining({ ruleId: 500, ceIds: [11, 22], userId: 7, classifierId: null, scenario: 'catch jailbreaks' }),
        ));
        await waitFor(() => expect(api.completePipelineRun).toHaveBeenCalledWith(12));
    });

    it('falls back to step-2A rule_id and existing ce_ids', async () => {
        const run = { run_id: 13, steps: { '2A': { data: { rule_id: 777, ce_ids: [9] } } } };
        expect(buildRuleInBackground(tray, { run, userId: 1 })).toBe(true);
        await waitFor(() => expect(api.embedResources).toHaveBeenCalledWith(
            expect.objectContaining({ ruleId: 777, ceIds: [9], scenario: null }),
        ));
    });

    it('returns false and starts no job when there is no ruleId', () => {
        expect(buildRuleInBackground(tray, { run: { run_id: 1, steps: { '2A': { data: {} } } } })).toBe(false);
        expect(tray.start).not.toHaveBeenCalled();
    });

    it('routes a build failure to the tray error chip', async () => {
        api.embedResources.mockRejectedValue(new Error('embed boom'));
        const run = { run_id: 18, rule_id: 3, steps: { '2A': { data: { ce_ids: [1] } } } };
        buildRuleInBackground(tray, { run, userId: 1 });
        await waitFor(() => expect(trayTask.error).toHaveBeenCalled());
        expect(api.completePipelineRun).not.toHaveBeenCalled();
    });
});

describe('buildCeInBackground', () => {
    const ceRun = { run_id: 88, steps: { '1': { data: { ce_data: { name: 'MyCE', definition: 'd', type: 'ACTION' } } } } };

    it('creates the CE (deferred), calibrates, embeds, completes', async () => {
        api.generateCeTraining.mockResolvedValue({ data: { ce_id: 555 } });
        expect(buildCeInBackground(tray, { run: ceRun, userId: 42 })).toBe(true);

        await waitFor(() => expect(api.generateCeTraining).toHaveBeenCalledWith(
            expect.objectContaining({ ce_name: 'MyCE', defer_ready: true }),
        ));
        await waitFor(() => expect(api.generateCeCalibration).toHaveBeenCalledWith(555, expect.any(Number)));
        await waitFor(() => expect(api.embedResources).toHaveBeenCalledWith(
            { ruleId: null, ceIds: [555], userId: 42, classifierId: null },
        ));
        await waitFor(() => expect(api.completePipelineRun).toHaveBeenCalledWith(88));
    });

    it('returns false and starts no job when the CE proposal is missing', () => {
        expect(buildCeInBackground(tray, { run: { run_id: 1, steps: {} }, userId: 1 })).toBe(false);
        expect(tray.start).not.toHaveBeenCalled();
    });

    it('errors to the tray (no embed/complete) when training returns no ce_id', async () => {
        api.generateCeTraining.mockResolvedValue({ data: {} }); // missing ce_id
        buildCeInBackground(tray, { run: ceRun, userId: 1 });
        await waitFor(() => expect(trayTask.error).toHaveBeenCalled());
        expect(api.embedResources).not.toHaveBeenCalled();
        expect(api.completePipelineRun).not.toHaveBeenCalled();
    });
});
