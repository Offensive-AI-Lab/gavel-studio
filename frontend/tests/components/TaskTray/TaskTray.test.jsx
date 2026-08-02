// TaskTray is the visible top-right pill + expanded panel of chips. It
// reads from useTaskTray() and renders different UI per task status. We
// test the parts that affect what the user sees:
//
//   * the tray hides itself when there are zero tasks
//   * the pill summary chooses the "loudest" status across all chips
//     (needs-input > error > running > success)
//   * needs-input chips are clickable; running ones are not
//   * dismiss button only shows on success/error chips
//   * click-outside collapses the expanded panel
//
// We use the real TaskTrayProvider so the inter-component wiring is
// exercised, and drive task creation via a test-only helper component
// that calls useTaskTray().start().

import React, { useEffect } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { TaskTrayProvider, useTaskTray } from '../../../src/contexts/TaskTrayContext';
import TaskTray from '../../../src/components/TaskTray/TaskTray';


// Helper that pushes pre-cooked task states into the tray. `setup` runs once
// on mount and is given the tray API.
const TrayDriver = ({ setup }) => {
    const tray = useTaskTray();
    useEffect(() => {
        if (setup) setup(tray);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return null;
};

const renderTray = (setup) => render(
    <TaskTrayProvider>
        <TaskTray />
        <TrayDriver setup={setup} />
    </TaskTrayProvider>,
);


describe('TaskTray', () => {
    it('renders nothing when there are no tasks', () => {
        const { container } = renderTray(() => {});
        // The tray returns null on empty list — the region landmark should
        // not be in the DOM.
        expect(container).toBeEmptyDOMElement();
    });

    it('shows the pill with the task count once a task is started', () => {
        renderTray((tray) => { tray.start({ title: 'A' }); });
        // Pill always shows the total count.
        expect(screen.getByRole('region', { name: /Background tasks/ })).toBeInTheDocument();
        const pillBtn = screen.getByRole('button', { name: /Background tasks/ });
        // Default state is collapsed → aria-expanded=false.
        expect(pillBtn).toHaveAttribute('aria-expanded', 'false');
    });

    it('expands the panel when the pill is clicked', () => {
        renderTray((tray) => { tray.start({ title: 'A' }); });
        const pillBtn = screen.getByRole('button', { name: /Background tasks/ });
        fireEvent.click(pillBtn);
        expect(pillBtn).toHaveAttribute('aria-expanded', 'true');
        // The chip body becomes visible.
        expect(screen.getByText('A')).toBeInTheDocument();
    });

    it('summary text reflects needs-input over running', () => {
        // running + needs-input → "needs you" wins because it's the loudest.
        renderTray((tray) => {
            tray.start({ title: 'still running' });
            const handle = tray.start({ title: 'waiting' });
            handle.prompt({ title: 'waiting', open: () => Promise.resolve(), autoExpand: false });
        });
        expect(screen.getByText(/needs? you/i)).toBeInTheDocument();
    });

    it('summary falls back to "running" when only running tasks exist', () => {
        renderTray((tray) => { tray.start({ title: 'A' }); tray.start({ title: 'B' }); });
        expect(screen.getByText(/2 running/i)).toBeInTheDocument();
    });

    it('summary shows "failed" when at least one task is in error', () => {
        renderTray((tray) => {
            const a = tray.start({ title: 'ok' });
            const b = tray.start({ title: 'bad' });
            b.error({ title: 'boom' });
        });
        expect(screen.getByText(/1 failed/i)).toBeInTheDocument();
    });

    it('summary shows "done" when only successes remain', () => {
        // Need to disable the auto-dismiss timer so the success chip stays.
        vi.useFakeTimers();
        try {
            renderTray((tray) => {
                const h = tray.start({ title: 'a' });
                h.success({ title: 'saved' });
            });
            // Don't advance timers — the chip is still there.
            expect(screen.getByText(/1 done/i)).toBeInTheDocument();
        } finally {
            vi.useRealTimers();
        }
    });

    it('needs-input chip is clickable; running chip is not', () => {
        // Render expanded so chips are visible.
        renderTray((tray) => {
            const h = tray.start({ title: 'click me' });
            // prompt() patches the task — pass the title back so it's
            // preserved (otherwise `title: title ?? undefined` clobbers it).
            h.prompt({ title: 'click me', open: () => Promise.resolve(), autoExpand: true });
            tray.start({ title: 'just running' });
        });
        // After prompt(), tray is expanded, the chip with status needs-input
        // gets role=button, the running chip does not.
        const buttons = screen.getAllByRole('button');
        const labels = buttons.map((b) => b.textContent || '');
        expect(labels.some((l) => l.includes('click me'))).toBe(true);
        // The running chip's title is rendered but its container is NOT a button.
        const runningTitle = screen.getByText('just running');
        const runningChip = runningTitle.closest('.task-chip');
        expect(runningChip).not.toHaveAttribute('role', 'button');
    });

    it('Enter on a needs-input chip fires the open handler', () => {
        const open = vi.fn(() => Promise.resolve());
        renderTray((tray) => {
            const h = tray.start({ title: 'press enter' });
            h.prompt({ title: 'press enter', open, autoExpand: true });
        });
        const chipTitle = screen.getByText('press enter');
        const chip = chipTitle.closest('.task-chip');
        // Keyboard activation is critical for accessibility.
        fireEvent.keyDown(chip, { key: 'Enter' });
        expect(open).toHaveBeenCalled();
    });

    it('shows a Dismiss button on success / error chips and removes the task on click', () => {
        renderTray((tray) => {
            const h = tray.start({ title: 'will fail' });
            h.error({ title: 'oh no' });
            tray.toggleExpanded();
        });
        const dismiss = screen.getByLabelText('Dismiss task');
        fireEvent.click(dismiss);
        // Chip removed → tray returns to empty → entire tray DOM gone.
        expect(screen.queryByText('oh no')).not.toBeInTheDocument();
    });

    it('does not show a Dismiss button on running chips', () => {
        renderTray((tray) => { tray.start({ title: 'A' }); tray.toggleExpanded(); });
        expect(screen.queryByLabelText('Dismiss task')).not.toBeInTheDocument();
    });

    it('click-outside collapses the expanded panel', () => {
        renderTray((tray) => { tray.start({ title: 'A' }); tray.toggleExpanded(); });
        const pillBtn = screen.getByRole('button', { name: /Background tasks/ });
        expect(pillBtn).toHaveAttribute('aria-expanded', 'true');

        // Simulate a click on the document body, OUTSIDE the tray.
        act(() => {
            document.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
        });

        expect(pillBtn).toHaveAttribute('aria-expanded', 'false');
    });

    it('pill summary singularizes correctly for one task', () => {
        renderTray((tray) => {
            const h = tray.start({ title: 'lonely' });
            h.prompt({ title: 'lonely', open: () => Promise.resolve(), autoExpand: false });
        });
        // Singular: "1 needs you" (no trailing s).
        expect(screen.getByText(/1 needs you/i)).toBeInTheDocument();
    });
});
