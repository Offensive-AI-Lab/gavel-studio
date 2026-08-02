// Resilience: what happens to an in-flight background task when the user
// navigates away or closes the tab mid-task.
//
// runInTray runs the job DETACHED and talks only to the App-level TaskTray
// (not component state). So unmounting the component that *started* the job
// must NOT cancel it, must NOT throw, and must NOT produce a React
// "setState on an unmounted component" warning — the job keeps running and
// its result (success OR failure) lands on the tray chip regardless of where
// the user went.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { runInTray } from '../../src/hooks/runInTray';

const makeTray = () => {
    const task = { update: vi.fn(), success: vi.fn(), error: vi.fn() };
    return { tray: { start: vi.fn(() => task) }, task };
};

// Minimal component that kicks off a detached tray job on click.
function Starter({ tray, job, onSuccess, onError }) {
    return (
        <button onClick={() => runInTray(tray, { title: 'bg', job, onSuccess, onError })}>
            go
        </button>
    );
}

describe('runInTray — component unmount / navigate-away mid-task', () => {
    let errSpy;
    beforeEach(() => {
        // Catch any React warning (e.g. update-on-unmounted) that the detached
        // task might trigger; we assert it never fires.
        errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    });
    afterEach(() => {
        errSpy.mockRestore();
    });

    it('a started job survives the component unmounting and still completes', async () => {
        const { tray, task } = makeTray();
        let resolveJob;
        const job = vi.fn(() => new Promise((r) => { resolveJob = r; }));

        const { unmount } = render(<Starter tray={tray} job={job} />);
        fireEvent.click(screen.getByText('go'));
        expect(job).toHaveBeenCalledTimes(1);

        // User leaves the page / closes the tab before the job finishes.
        unmount();

        // The job resolves afterwards — detached from the unmounted component.
        resolveJob('ok');
        await waitFor(() => expect(task.success).toHaveBeenCalledTimes(1));
        expect(task.error).not.toHaveBeenCalled();
    });

    it('a job that FAILS after unmount routes to the tray error chip (no crash)', async () => {
        const { tray, task } = makeTray();
        let rejectJob;
        const job = vi.fn(() => new Promise((_, rej) => { rejectJob = rej; }));

        const { unmount } = render(<Starter tray={tray} job={job} />);
        fireEvent.click(screen.getByText('go'));
        unmount();

        // Network dies after the user already navigated away.
        rejectJob(new Error('network lost'));
        await waitFor(() => expect(task.error).toHaveBeenCalledTimes(1));
        expect(task.error.mock.calls[0][0].subtitle).toBe('network lost');
        expect(task.success).not.toHaveBeenCalled();
    });

    it('does not emit a React update-on-unmounted warning', async () => {
        const { tray, task } = makeTray();
        let resolveJob;
        const job = () => new Promise((r) => { resolveJob = r; });
        // onSuccess deliberately does nothing that touches React state.
        const { unmount } = render(<Starter tray={tray} job={job} onSuccess={() => {}} />);
        fireEvent.click(screen.getByText('go'));
        unmount();
        resolveJob('done');
        await waitFor(() => expect(task.success).toHaveBeenCalled());

        const warned = errSpy.mock.calls.some(([msg]) =>
            typeof msg === 'string' && /unmounted|not wrapped in act/i.test(msg),
        );
        expect(warned).toBe(false);
    });
});
