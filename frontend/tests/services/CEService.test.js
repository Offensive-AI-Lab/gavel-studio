// Behavior tests for CEService.js — the Cognitive Element add/remove flows.
//
// CEService is a pure logic module (no React component). It drives the UI via
// SweetAlert2 (`Swal.fire`) and the backend via `../api` (`api.post`,
// `api.delete`, and the named `getUserCEs` export). It also renders a
// PipelineModal to a static HTML string for the error/info popups.
//
// We mock Swal so each test can script the sequence of dialog answers, and we
// mock `../api` so nothing hits the network. Every branch is exercised:
//   - handleAddCEFlow: Select Existing (true) / Create New (false) / Cancel,
//     empty vs non-empty available list, link success/failure, create
//     success/failure, the `res.data.ces` vs `res.data` fallback, and the
//     "already in rule" filter. Quick-adds pass the backend's optional
//     `group` name (default "additional"); the backend owns the rule's
//     groups/condition and re-derives the predicate itself.
//   - handleRemoveCEFlow: delete success and delete failure (membership-only
//     local update — no client-side predicate rebuild).

import { describe, it, expect, vi, beforeEach } from 'vitest';
import Swal from 'sweetalert2';
import api, { getUserCEs } from '../../src/api';
import { handleAddCEFlow, handleRemoveCEFlow } from '../../src/services/CEService';

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn() },
}));

vi.mock('../../src/api', () => ({
    default: {
        post: vi.fn(() => Promise.resolve({ data: {} })),
        delete: vi.fn(() => Promise.resolve({ data: {} })),
    },
    getUserCEs: vi.fn(() => Promise.resolve({ data: { ces: [] } })),
}));

// Helper: queue up the answers Swal.fire should return, in call order.
const queueSwal = (...returns) => {
    Swal.fire.mockReset();
    returns.forEach((r) => Swal.fire.mockResolvedValueOnce(r));
    // Any extra calls (e.g. showPipelineMessage at the end) resolve benignly.
    Swal.fire.mockResolvedValue({ value: undefined });
};

// A fresh rules fixture for each test (mutated in place by the service).
const makeRules = () => [
    {
        setup_id: 'setup-1',
        predicate: 'IF TRUE THEN BLOCK',
        active_ces: [{ ce_id: 'ce-existing', name: 'Existing CE' }],
    },
];

beforeEach(() => {
    vi.clearAllMocks();
    api.post.mockResolvedValue({ data: {} });
    api.delete.mockResolvedValue({ data: {} });
    getUserCEs.mockResolvedValue({ data: { ces: [] } });
});

describe('handleAddCEFlow — top-level dialog branches', () => {
    it('opens the Add Cognitive Element dialog with the expected config', async () => {
        queueSwal({ value: 'cancelled' }); // not true, not false
        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        expect(Swal.fire).toHaveBeenCalledTimes(1);
        const cfg = Swal.fire.mock.calls[0][0];
        expect(cfg.title).toBe('Add Cognitive Element');
        expect(cfg.showDenyButton).toBe(true);
        expect(cfg.confirmButtonText).toBe('Select Existing');
        expect(cfg.denyButtonText).toBe('Create New (Hand)');
        // Neither path taken -> no api calls, no state change.
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });

    it('does nothing on cancel (value undefined)', async () => {
        queueSwal({ value: undefined });
        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);
        expect(getUserCEs).not.toHaveBeenCalled();
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });
});

describe('handleAddCEFlow — Select Existing (result === true)', () => {
    it('fetches user CEs and links the chosen one, updating UI + predicate', async () => {
        getUserCEs.mockResolvedValueOnce({
            data: { ces: [
                { ce_id: 'ce-existing', name: 'Existing CE' }, // filtered out
                { ce_id: 'ce-new', name: 'Brand New CE' },
            ] },
        });
        // 1: top dialog -> true ; 2: select dialog -> chosen ce id
        queueSwal({ value: true }, { value: 'ce-new' });

        const rules = makeRules();
        const updateState = vi.fn();
        await handleAddCEFlow('user-1', rules, 0, updateState);

        expect(getUserCEs).toHaveBeenCalledWith('user-1');

        // Select dialog only offered the non-existing CE.
        const selectCfg = Swal.fire.mock.calls[1][0];
        expect(selectCfg.input).toBe('select');
        expect(selectCfg.inputOptions).toEqual({ 'ce-new': 'Brand New CE' });

        // Linked via api.post to the right setup, into the default
        // quick-add group.
        expect(api.post).toHaveBeenCalledWith('/rules/setup/setup-1/link-ce', { ce_id: 'ce-new', group: 'additional' });

        // Local UI updated: CE appended to the membership list. The
        // predicate is server-derived now — no client-side rebuild.
        expect(updateState).toHaveBeenCalledTimes(1);
        const newRules = updateState.mock.calls[0][0];
        expect(newRules[0].active_ces).toEqual([
            { ce_id: 'ce-existing', name: 'Existing CE' },
            { name: 'Brand New CE', ce_id: 'ce-new' },
        ]);
        expect(newRules[0].predicate).toBe('IF TRUE THEN BLOCK');
    });

    it('falls back to res.data when res.data.ces is absent', async () => {
        // No `.ces` wrapper — service should use res.data directly.
        getUserCEs.mockResolvedValueOnce({
            data: [{ ce_id: 'ce-bare', name: 'Bare CE' }],
        });
        queueSwal({ value: true }, { value: 'ce-bare' });

        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        const selectCfg = Swal.fire.mock.calls[1][0];
        expect(selectCfg.inputOptions).toEqual({ 'ce-bare': 'Bare CE' });
        expect(api.post).toHaveBeenCalledWith('/rules/setup/setup-1/link-ce', { ce_id: 'ce-bare', group: 'additional' });
        expect(updateState).toHaveBeenCalledTimes(1);
    });

    it('handles a bare empty array in res.data (no .ces wrapper)', async () => {
        getUserCEs.mockResolvedValueOnce({ data: [] });
        queueSwal({ value: true });

        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        // availableCes is empty -> "Nothing To Add" pipeline message, no select.
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
        // Second Swal.fire was the info popup.
        expect(Swal.fire).toHaveBeenCalledTimes(2);
        expect(Swal.fire.mock.calls[1][0].html).toContain('Nothing To Add');
    });

    it('shows "Nothing To Add" when every CE is already in the rule', async () => {
        getUserCEs.mockResolvedValueOnce({
            data: { ces: [{ ce_id: 'ce-existing', name: 'Existing CE' }] },
        });
        queueSwal({ value: true });

        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        expect(Swal.fire).toHaveBeenCalledTimes(2);
        expect(Swal.fire.mock.calls[1][0].html).toContain('Nothing To Add');
        expect(Swal.fire.mock.calls[1][0].html).toContain('already in this rule');
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });

    it('does nothing when no CE is selected from the dropdown', async () => {
        getUserCEs.mockResolvedValueOnce({
            data: { ces: [{ ce_id: 'ce-new', name: 'Brand New CE' }] },
        });
        queueSwal({ value: true }, { value: undefined }); // select cancelled

        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });

    it('shows a Link Failed popup when api.post rejects', async () => {
        getUserCEs.mockResolvedValueOnce({
            data: { ces: [{ ce_id: 'ce-new', name: 'Brand New CE' }] },
        });
        api.post.mockRejectedValueOnce(new Error('boom'));
        queueSwal({ value: true }, { value: 'ce-new' });

        const updateState = vi.fn();
        await handleAddCEFlow('user-1', makeRules(), 0, updateState);

        expect(api.post).toHaveBeenCalledTimes(1);
        expect(updateState).not.toHaveBeenCalled();
        // Final Swal call is the error popup.
        const lastCall = Swal.fire.mock.calls[Swal.fire.mock.calls.length - 1][0];
        expect(lastCall.html).toContain('Link Failed');
        expect(lastCall.confirmButtonColor).toBe('#ef4444');
    });
});

describe('handleAddCEFlow — Create New (result === false)', () => {
    it('creates a CE, posts to create-ce, and updates the UI', async () => {
        api.post.mockResolvedValueOnce({ data: { ce_id: 'ce-created' } });
        // 1: top dialog -> false ; 2: text dialog -> name
        queueSwal({ value: false }, { value: 'My Fresh CE' });

        const rules = makeRules();
        const updateState = vi.fn();
        await handleAddCEFlow('user-9', rules, 0, updateState);

        // Create text dialog config.
        const textCfg = Swal.fire.mock.calls[1][0];
        expect(textCfg.title).toBe('Create CE');
        expect(textCfg.input).toBe('text');

        expect(api.post).toHaveBeenCalledWith('/rules/setup/setup-1/create-ce', {
            name: 'My Fresh CE',
            user_id: 'user-9',
            group: 'additional',
        });

        expect(updateState).toHaveBeenCalledTimes(1);
        const newRules = updateState.mock.calls[0][0];
        expect(newRules[0].active_ces).toContainEqual({ name: 'My Fresh CE', ce_id: 'ce-created' });
    });

    it('does nothing when the create dialog is dismissed without a name', async () => {
        queueSwal({ value: false }, { value: undefined });
        const updateState = vi.fn();
        await handleAddCEFlow('user-9', makeRules(), 0, updateState);
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });

    it('does nothing for an empty-string name (falsy)', async () => {
        queueSwal({ value: false }, { value: '' });
        const updateState = vi.fn();
        await handleAddCEFlow('user-9', makeRules(), 0, updateState);
        expect(api.post).not.toHaveBeenCalled();
        expect(updateState).not.toHaveBeenCalled();
    });

    it('shows a Creation Failed popup when create api.post rejects', async () => {
        api.post.mockRejectedValueOnce(new Error('nope'));
        queueSwal({ value: false }, { value: 'Doomed CE' });

        const updateState = vi.fn();
        await handleAddCEFlow('user-9', makeRules(), 0, updateState);

        expect(api.post).toHaveBeenCalledTimes(1);
        expect(updateState).not.toHaveBeenCalled();
        const lastCall = Swal.fire.mock.calls[Swal.fire.mock.calls.length - 1][0];
        expect(lastCall.html).toContain('Creation Failed');
        expect(lastCall.confirmButtonColor).toBe('#ef4444');
    });
});

describe('handleRemoveCEFlow', () => {
    it('deletes the CE and updates the local membership list', async () => {
        const rules = [
            {
                setup_id: 'setup-1',
                predicate: 'A AND B',
                active_ces: [
                    { ce_id: 'ce-a', name: 'A' },
                    { ce_id: 'ce-b', name: 'B' },
                ],
            },
        ];
        const updateState = vi.fn();
        await handleRemoveCEFlow('setup-1', 'ce-a', 'A', rules, 0, updateState);

        expect(api.delete).toHaveBeenCalledWith('/rules/setup/setup-1/ce/ce-a');
        expect(updateState).toHaveBeenCalledTimes(1);
        const newRules = updateState.mock.calls[0][0];
        expect(newRules[0].active_ces).toEqual([{ ce_id: 'ce-b', name: 'B' }]);
        // Predicate is server-derived; the local copy is left untouched
        // until the next fetch.
        expect(newRules[0].predicate).toBe('A AND B');
    });

    it('shows a Remove Failed popup and does not update state on delete error', async () => {
        api.delete.mockRejectedValueOnce(new Error('fail'));
        const rules = makeRules();
        const updateState = vi.fn();
        await handleRemoveCEFlow('setup-1', 'ce-existing', 'Existing CE', rules, 0, updateState);

        expect(updateState).not.toHaveBeenCalled();
        expect(Swal.fire).toHaveBeenCalledTimes(1);
        const cfg = Swal.fire.mock.calls[0][0];
        expect(cfg.html).toContain('Remove Failed');
        expect(cfg.confirmButtonColor).toBe('#ef4444');
    });
});
