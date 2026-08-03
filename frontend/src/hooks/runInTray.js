// Run an async job as a background TaskTray task — non-blocking.
//
// The job runs DETACHED: the returned promise is not awaited by the caller, so
// the user can navigate away from the wizard/page while the job keeps going.
// The TaskTray lives at App level (outside the Router), so its chip continues
// to show progress and the final success/error no matter where the user goes.
//
//   runInTray(tray, {
//     kind: 'rule',
//     title: 'Generating test & calibration set',
//     job: async (update) => { await kickoff(); await pollUntilReady(update); },
//     successSubtitle: 'Ready.',
//   });
//
// `job` receives an `update(patch)` fn to push live subtitle/progress to the
// chip. Throwing inside `job` flips the chip to error with the message.
//
// `errorOnOpen(e)` (optional) makes the red error chip CLICKABLE: it becomes
// the chip's onOpen handler, called with the thrown error. Use it to show the
// full failure reason (the chip subtitle is one ellipsized line) and/or offer
// a retry. Without it the error chip is display-only, as before.
export function runInTray(tray, { kind = 'generic', title, runningSubtitle, successTitle, successSubtitle, job, onSuccess, onError, errorOnOpen } = {}) {
    const task = tray.start({ kind, title, subtitle: runningSubtitle || 'Running in the background…' });
    (async () => {
        try {
            const result = await job(((patch) => task.update(patch)));
            task.success({ title: successTitle || title, subtitle: successSubtitle || 'Done' });
            onSuccess?.(result);
        } catch (e) {
            const msg = e?.response?.data?.detail || e?.message || 'Failed';
            const patch = { title, subtitle: msg };
            if (errorOnOpen) patch.onOpen = () => errorOnOpen(e);
            task.error(patch);
            onError?.(e);
        }
    })();
    return task;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
