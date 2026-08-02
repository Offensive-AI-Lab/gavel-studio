// TaskTrayContext drives every long-running pipeline chip in the app.
// The Promise machinery in `prompt()` and the auto-dismiss on success are
// the parts most likely to break silently — they're invisible until a user
// is staring at a stuck chip wondering why nothing happens. The tests below
// pin down the public contract:
//
//   * start() returns a handle that drives the lifecycle
//   * prompt() resolves with the open() return value, ONLY after a click
//   * needs-input dispatches expand the panel — unless autoExpand=false
//   * success auto-dismisses; error sticks
//   * useTaskTray throws when used outside the provider

import React, { useEffect, useRef } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { act, render, renderHook, screen } from '@testing-library/react';
import { TaskTrayProvider, useTaskTray } from '../../src/contexts/TaskTrayContext';

const wrapper = ({ children }) => <TaskTrayProvider>{children}</TaskTrayProvider>;

describe('TaskTrayContext', () => {
    describe('useTaskTray hook', () => {
        it('throws if used outside the provider', () => {
            // We want a loud error so a developer who forgets to wrap the
            // app sees it on first load, not a confusing null deref later.
            expect(() => renderHook(() => useTaskTray())).toThrow(/inside <TaskTrayProvider>/);
        });

        it('returns a working tray inside the provider', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            expect(result.current).toBeTruthy();
            expect(typeof result.current.start).toBe('function');
            expect(result.current.tasks).toEqual([]);
        });
    });

    describe('start()', () => {
        it('appends a running task with caller-supplied title/subtitle', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            act(() => {
                result.current.start({ kind: 'rule', title: 'Generating', subtitle: 'thinking' });
            });
            expect(result.current.tasks).toHaveLength(1);
            const t = result.current.tasks[0];
            expect(t.kind).toBe('rule');
            expect(t.title).toBe('Generating');
            expect(t.subtitle).toBe('thinking');
            expect(t.status).toBe('running');
        });

        it('uses sensible defaults when called with no args', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            act(() => { result.current.start(); });
            const t = result.current.tasks[0];
            expect(t.kind).toBe('generic');
            expect(t.status).toBe('running');
            expect(t.title).toBeTruthy();
        });

        it('returns a handle whose update() patches the task', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start({ title: 'A' }); });
            act(() => { handle.update({ title: 'B', subtitle: 'C' }); });
            expect(result.current.tasks[0].title).toBe('B');
            expect(result.current.tasks[0].subtitle).toBe('C');
        });

        it('returns a handle whose close() removes the task', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start(); });
            expect(result.current.tasks).toHaveLength(1);
            act(() => { handle.close(); });
            expect(result.current.tasks).toHaveLength(0);
        });
    });

    describe('prompt()', () => {
        it('moves the chip to needs-input and stays pending until clicked', async () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start({ title: 'A' }); });

            const open = vi.fn(() => Promise.resolve('open-result'));
            let promptPromise;
            act(() => {
                promptPromise = handle.prompt({ title: 'click me', subtitle: 'sub', open });
            });

            // Without a click on the chip, prompt must not resolve yet.
            expect(result.current.tasks[0].status).toBe('needs-input');
            expect(result.current.tasks[0].title).toBe('click me');
            expect(open).not.toHaveBeenCalled();

            // The user clicking the chip is the trigger that fires open().
            await act(async () => {
                result.current.open(result.current.tasks[0].id);
                await promptPromise.then((v) => {
                    expect(v).toBe('open-result');
                });
            });
            expect(open).toHaveBeenCalledTimes(1);
        });

        it('autoExpands the panel by default; not when autoExpand=false', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start({ title: 'A' }); });
            expect(result.current.expanded).toBe(false);

            // First call: default autoExpand=true → panel pops open.
            act(() => { handle.prompt({ open: () => Promise.resolve() }); });
            expect(result.current.expanded).toBe(true);

            // Reset and call again with autoExpand=false → must not expand.
            // Park the chip back via re-prompting with autoExpand=false.
            act(() => { result.current.setExpanded(false); });
            act(() => { handle.prompt({ open: () => Promise.resolve(), autoExpand: false }); });
            expect(result.current.expanded).toBe(false);
        });

        it('drops the chip back to running while open() is in flight', async () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start(); });

            let resolveOpen;
            const open = () => new Promise((r) => { resolveOpen = r; });
            let promptPromise;
            act(() => { promptPromise = handle.prompt({ open }); });
            expect(result.current.tasks[0].status).toBe('needs-input');

            await act(async () => {
                result.current.open(result.current.tasks[0].id);
            });
            // While open() hasn't resolved, the chip should be back to
            // 'running' so the visual state matches reality.
            expect(result.current.tasks[0].status).toBe('running');

            await act(async () => {
                resolveOpen('done');
                await promptPromise;
            });
        });

        it('rejects the prompt promise when open() throws', async () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start(); });
            const err = new Error('boom');
            const open = vi.fn(() => Promise.reject(err));
            let promptPromise;
            act(() => { promptPromise = handle.prompt({ open }); });

            await act(async () => {
                result.current.open(result.current.tasks[0].id);
                await expect(promptPromise).rejects.toThrow('boom');
            });
        });
    });

    describe('success() / error()', () => {
        it('success auto-dismisses after the timeout', () => {
            vi.useFakeTimers();
            try {
                const { result } = renderHook(() => useTaskTray(), { wrapper });
                let handle;
                act(() => { handle = result.current.start(); });
                act(() => { handle.success({ title: 'Saved!' }); });
                expect(result.current.tasks[0].status).toBe('success');
                expect(result.current.tasks).toHaveLength(1);

                act(() => { vi.advanceTimersByTime(10_000); });
                expect(result.current.tasks).toHaveLength(0);
            } finally {
                vi.useRealTimers();
            }
        });

        it('error sticks until the user dismisses it', () => {
            vi.useFakeTimers();
            try {
                const { result } = renderHook(() => useTaskTray(), { wrapper });
                let handle;
                act(() => { handle = result.current.start(); });
                act(() => { handle.error({ title: 'oops' }); });
                act(() => { vi.advanceTimersByTime(60_000); });
                // Errors must NOT auto-dismiss — the user has to see them.
                expect(result.current.tasks).toHaveLength(1);
                expect(result.current.tasks[0].status).toBe('error');

                act(() => { result.current.dismiss(result.current.tasks[0].id); });
                expect(result.current.tasks).toHaveLength(0);
            } finally {
                vi.useRealTimers();
            }
        });

        it('open() on a success/error task fires the onOpen callback if set', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start(); });
            const onOpen = vi.fn();
            act(() => { handle.success({ title: 'done', onOpen }); });
            act(() => { result.current.open(result.current.tasks[0].id); });
            expect(onOpen).toHaveBeenCalledTimes(1);
        });

        it('open() on a task with no handler is a quiet no-op', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            let handle;
            act(() => { handle = result.current.start(); });
            // Plain running chip with no openHandler — clicking does nothing.
            expect(() => {
                act(() => { result.current.open(result.current.tasks[0].id); });
            }).not.toThrow();
        });
    });

    describe('toggleExpanded / setExpanded', () => {
        it('toggleExpanded flips the panel state', () => {
            const { result } = renderHook(() => useTaskTray(), { wrapper });
            expect(result.current.expanded).toBe(false);
            act(() => { result.current.toggleExpanded(); });
            expect(result.current.expanded).toBe(true);
            act(() => { result.current.toggleExpanded(); });
            expect(result.current.expanded).toBe(false);
        });
    });
});
