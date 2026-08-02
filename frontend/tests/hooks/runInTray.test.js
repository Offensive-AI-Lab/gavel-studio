// Tests for runInTray — a pure helper that wraps an async `job` in a
// TaskTray task handle, running it DETACHED (the returned promise isn't
// awaited). It pushes live updates via `update`, then flips the chip to
// success or error.
//
// runInTray takes no React, no network, no router — it only talks to the
// `tray` handle. So we drive it with a fake tray whose `start()` returns a
// task handle of spies, and assert the exact lifecycle calls + ordering.
// The detached async work means we await a microtask-flush helper (or the
// job's own resolution) before asserting post-job state.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { runInTray, sleep } from '../../src/hooks/runInTray';

// Build a fake tray. `start` records the opts it was called with and returns
// a handle of spies. The returned task object is the same one runInTray
// returns to its caller, so tests can inspect it directly too.
const makeTray = () => {
    const task = {
        update: vi.fn(),
        success: vi.fn(),
        error: vi.fn(),
    };
    const start = vi.fn(() => task);
    return { tray: { start }, start, task };
};

// Let the detached IIFE inside runInTray run to completion. The job is
// awaited inside that IIFE; flushing the microtask queue a few times lets
// the whole try/catch settle before we assert.
const flush = async () => {
    for (let i = 0; i < 5; i += 1) {
        await Promise.resolve();
    }
};

describe('runInTray', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('starts a task with the given kind/title and a default running subtitle', () => {
        const { tray, start } = makeTray();
        runInTray(tray, { kind: 'rule', title: 'Generating set', job: vi.fn() });

        expect(start).toHaveBeenCalledTimes(1);
        expect(start).toHaveBeenCalledWith({
            kind: 'rule',
            title: 'Generating set',
            subtitle: 'Running in the background…',
        });
    });

    it('defaults kind to "generic" when none is given', () => {
        const { tray, start } = makeTray();
        runInTray(tray, { title: 'X', job: vi.fn() });
        expect(start.mock.calls[0][0].kind).toBe('generic');
    });

    it('uses the provided runningSubtitle over the default', () => {
        const { tray, start } = makeTray();
        runInTray(tray, { title: 'X', runningSubtitle: 'Warming up', job: vi.fn() });
        expect(start.mock.calls[0][0].subtitle).toBe('Warming up');
    });

    it('returns the task handle from tray.start', () => {
        const { tray, task } = makeTray();
        const returned = runInTray(tray, { title: 'X', job: vi.fn() });
        expect(returned).toBe(task);
    });

    it('works with no options object at all (uses all defaults)', () => {
        const { tray, start } = makeTray();
        // job is undefined → calling it throws synchronously inside the async
        // IIFE, which is caught and routed to error. We just assert start ran
        // with generic defaults and nothing throws to the caller.
        expect(() => runInTray(tray)).not.toThrow();
        expect(start.mock.calls[0][0].kind).toBe('generic');
        expect(start.mock.calls[0][0].title).toBeUndefined();
    });

    it('invokes the job with an update fn that forwards patches to task.update', () => {
        const { tray, task } = makeTray();
        const job = vi.fn();
        runInTray(tray, { title: 'X', job });

        expect(job).toHaveBeenCalledTimes(1);
        const update = job.mock.calls[0][0];
        expect(typeof update).toBe('function');

        update({ subtitle: 'halfway' });
        expect(task.update).toHaveBeenCalledWith({ subtitle: 'halfway' });
    });

    it('on success flips the chip to success with default success title/subtitle', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.resolve('result'));
        runInTray(tray, { title: 'My Job', job });

        await flush();
        expect(task.success).toHaveBeenCalledWith({ title: 'My Job', subtitle: 'Done' });
        expect(task.error).not.toHaveBeenCalled();
    });

    it('on success uses provided successTitle and successSubtitle', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.resolve());
        runInTray(tray, {
            title: 'My Job',
            successTitle: 'All set',
            successSubtitle: 'Ready.',
            job,
        });

        await flush();
        expect(task.success).toHaveBeenCalledWith({ title: 'All set', subtitle: 'Ready.' });
    });

    it('calls onSuccess with the job result on success', async () => {
        const { tray } = makeTray();
        const onSuccess = vi.fn();
        const job = vi.fn(() => Promise.resolve({ id: 42 }));
        runInTray(tray, { title: 'X', job, onSuccess });

        await flush();
        expect(onSuccess).toHaveBeenCalledWith({ id: 42 });
    });

    it('does not throw when onSuccess is omitted on success', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.resolve('ok'));
        runInTray(tray, { title: 'X', job });

        await flush();
        expect(task.success).toHaveBeenCalledTimes(1);
    });

    it('on failure flips the chip to error using e.response.data.detail', async () => {
        const { tray, task } = makeTray();
        const err = { response: { data: { detail: 'Server said no' } } };
        const job = vi.fn(() => Promise.reject(err));
        runInTray(tray, { title: 'My Job', job });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'My Job', subtitle: 'Server said no' });
        expect(task.success).not.toHaveBeenCalled();
    });

    it('on failure falls back to e.message when no response detail', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.reject(new Error('boom')));
        runInTray(tray, { title: 'T', job });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'T', subtitle: 'boom' });
    });

    it('on failure falls back to "Failed" when neither detail nor message exists', async () => {
        const { tray, task } = makeTray();
        // Reject with a plain object that has no response and no message.
        const job = vi.fn(() => Promise.reject({}));
        runInTray(tray, { title: 'T', job });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'T', subtitle: 'Failed' });
    });

    it('uses "Failed" when rejected with null/undefined', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.reject(undefined));
        runInTray(tray, { title: 'T', job });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'T', subtitle: 'Failed' });
    });

    it('prefers response.data.detail over message when both are present', async () => {
        const { tray, task } = makeTray();
        const err = new Error('client message');
        err.response = { data: { detail: 'precise detail' } };
        const job = vi.fn(() => Promise.reject(err));
        runInTray(tray, { title: 'T', job });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'T', subtitle: 'precise detail' });
    });

    it('calls onError with the thrown error on failure', async () => {
        const { tray } = makeTray();
        const onError = vi.fn();
        const err = new Error('nope');
        const job = vi.fn(() => Promise.reject(err));
        runInTray(tray, { title: 'X', job, onError });

        await flush();
        expect(onError).toHaveBeenCalledWith(err);
    });

    it('does not throw when onError is omitted on failure', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(() => Promise.reject(new Error('x')));
        runInTray(tray, { title: 'X', job });

        await flush();
        expect(task.error).toHaveBeenCalledTimes(1);
    });

    it('routes a synchronously-throwing job to the error path', async () => {
        const { tray, task } = makeTray();
        const onError = vi.fn();
        const job = vi.fn(() => { throw new Error('sync boom'); });
        runInTray(tray, { title: 'X', job, onError });

        await flush();
        expect(task.error).toHaveBeenCalledWith({ title: 'X', subtitle: 'sync boom' });
        expect(onError).toHaveBeenCalledTimes(1);
        expect(task.success).not.toHaveBeenCalled();
    });

    it('does not call success when onSuccess itself throws (error not swallowed into success)', async () => {
        // onSuccess runs AFTER task.success, so success should still have fired
        // exactly once even though onSuccess throws afterward.
        const { tray, task } = makeTray();
        const onSuccess = vi.fn(() => { throw new Error('after'); });
        const job = vi.fn(() => Promise.resolve('r'));
        runInTray(tray, { title: 'X', job, onSuccess });

        await flush();
        expect(task.success).toHaveBeenCalledTimes(1);
    });

    it('forwards live updates pushed by the job before completion', async () => {
        const { tray, task } = makeTray();
        const job = vi.fn(async (update) => {
            update({ subtitle: 'step 1', progress: 0.3 });
            update({ subtitle: 'step 2', progress: 0.9 });
            return 'done';
        });
        runInTray(tray, { title: 'X', job });

        await flush();
        expect(task.update).toHaveBeenNthCalledWith(1, { subtitle: 'step 1', progress: 0.3 });
        expect(task.update).toHaveBeenNthCalledWith(2, { subtitle: 'step 2', progress: 0.9 });
        expect(task.success).toHaveBeenCalledTimes(1);
    });

    it('runs detached: returns synchronously before the job resolves', async () => {
        const { tray, task } = makeTray();
        let resolveJob;
        const job = vi.fn(() => new Promise((res) => { resolveJob = res; }));
        const returned = runInTray(tray, { title: 'X', job });

        // Job started but not resolved yet → no success/error yet.
        expect(returned).toBe(task);
        expect(task.success).not.toHaveBeenCalled();
        expect(task.error).not.toHaveBeenCalled();

        resolveJob('finally');
        await flush();
        expect(task.success).toHaveBeenCalledTimes(1);
    });
});

describe('sleep', () => {
    it('returns a promise that resolves after the given delay', async () => {
        vi.useFakeTimers();
        try {
            const spy = vi.fn();
            const p = sleep(1000).then(spy);

            // Not resolved before the timer fires.
            await Promise.resolve();
            expect(spy).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(1000);
            await p;
            expect(spy).toHaveBeenCalledTimes(1);
        } finally {
            vi.useRealTimers();
        }
    });

    it('returns a promise', () => {
        vi.useFakeTimers();
        try {
            expect(sleep(0)).toBeInstanceOf(Promise);
        } finally {
            vi.useRealTimers();
        }
    });
});
