import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

// TaskTray context — a small in-memory queue of long-running pipeline tasks
// (AI rule generation, CE training, publish, etc.) rendered as chips in the
// top-right corner so the user can keep working while one or more pipelines
// run in the background.
//
// A task moves through a small state machine:
//   running       → blue spinner; nothing for the user to do
//   needs-input   → amber pulse; clicking opens a callback (proposal review,
//                   name conflict, etc.) — the pipeline is suspended on a
//                   promise that resolves when the callback finishes
//   success       → green; auto-dismisses after a few seconds
//   error         → red; sticks until the user closes it
//
// The pipeline drives a task with a small handle returned from `start`:
//   const task = tray.start({ kind: 'rule', title: 'Generating rule' });
//   task.update({ subtitle: 'Reviewing CEs...' });
//   const decision = await task.prompt({
//       title: 'Click to review proposal',
//       open: () => showProposalReviewModal(aiProposal),  // returns a promise
//   });
//   task.success({ subtitle: 'Saved as draft' });
//
// The tray only handles UI orchestration. The actual modals (Swal dialogs,
// custom panels) are still owned by the pipeline that called `prompt`.

const TaskTrayContext = createContext(null);

let _idCounter = 0;
const nextId = () => `task-${++_idCounter}-${Date.now()}`;

// Successes auto-dismiss after this many ms so they don't pile up.
const SUCCESS_AUTO_DISMISS_MS = 5000;

export const TaskTrayProvider = ({ children }) => {
    const [tasks, setTasks] = useState([]);
    // The tray collapses to a single pill by default. Expanded shows the
    // full chip list. Auto-expanded once whenever any task transitions to
    // 'needs-input' so the user sees the amber prompt without having to
    // hunt for the pill — after that, the user is in control.
    const [expanded, setExpanded] = useState(false);
    // Hold pending input resolvers keyed by task id. When the user clicks a
    // 'needs-input' chip we resolve the awaiting promise with whatever the
    // open() callback returned.
    const pendingResolvers = useRef(new Map());

    const _patch = useCallback((id, patch) => {
        setTasks((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    }, []);

    const _remove = useCallback((id) => {
        setTasks((prev) => prev.filter((t) => t.id !== id));
        pendingResolvers.current.delete(id);
    }, []);

    // Public API ------------------------------------------------------------

    const start = useCallback((opts = {}) => {
        const id = opts.id || nextId();
        const task = {
            id,
            kind: opts.kind || 'generic',           // 'rule' | 'ce' | 'publish' | 'generic'
            title: opts.title || 'Working…',
            subtitle: opts.subtitle || '',
            status: 'running',
            createdAt: Date.now(),
            // openHandler is set when the task moves to 'needs-input'
            openHandler: null,
        };
        setTasks((prev) => [...prev, task]);

        // Return a small handle so callers can drive the lifecycle without
        // touching the context directly.
        return {
            id,
            update: (patch) => _patch(id, patch),
            // Move task to needs-input. Returns a promise that resolves with
            // whatever `open()` returns. The user must click the chip first;
            // we then invoke open() and pass its result through.
            //
            // `autoExpand` defaults to true: the FIRST time a task moves
            // into needs-input we expand the tray so the user notices.
            // Callers that re-prompt the same task (e.g., the proposal
            // review's park-for-later loop after the user dismissed via X)
            // should pass `autoExpand: false` — the user just closed the
            // modal, popping the panel back open is the opposite of what
            // they asked for.
            prompt: ({ title, subtitle, open, autoExpand = true }) =>
                new Promise((resolve, reject) => {
                    const handler = async () => {
                        // While the modal is open, drop the chip back to
                        // 'running' so the visual state matches reality.
                        _patch(id, { status: 'running', openHandler: null });
                        try {
                            const result = await open();
                            resolve(result);
                        } catch (err) {
                            reject(err);
                        } finally {
                            pendingResolvers.current.delete(id);
                        }
                    };
                    pendingResolvers.current.set(id, handler);
                    _patch(id, {
                        status: 'needs-input',
                        title: title ?? undefined,
                        subtitle: subtitle ?? undefined,
                        openHandler: handler,
                    });
                    if (autoExpand) {
                        setExpanded(true);
                    }
                }),
            success: ({ title, subtitle, onOpen } = {}) => {
                _patch(id, {
                    status: 'success',
                    title: title ?? undefined,
                    subtitle: subtitle ?? undefined,
                    openHandler: onOpen ?? null,
                });
                setTimeout(() => _remove(id), SUCCESS_AUTO_DISMISS_MS);
            },
            error: ({ title, subtitle, onOpen } = {}) => {
                _patch(id, {
                    status: 'error',
                    title: title ?? 'Something went wrong',
                    subtitle: subtitle ?? '',
                    openHandler: onOpen ?? null,
                });
            },
            close: () => _remove(id),
        };
    }, [_patch, _remove]);

    const open = useCallback((id) => {
        const handler = pendingResolvers.current.get(id);
        const task = tasks.find((t) => t.id === id);
        if (handler) {
            // 'needs-input' click — runs the registered open() callback.
            handler();
        } else if (task?.openHandler) {
            // 'success' / 'error' click — runs whatever onOpen the caller
            // set, often a navigate-to-detail action.
            try { task.openHandler(); } catch (_) {}
        }
    }, [tasks]);

    const dismiss = useCallback((id) => {
        _remove(id);
    }, [_remove]);

    const value = useMemo(() => ({
        tasks,
        start,
        open,
        dismiss,
        expanded,
        setExpanded,
        toggleExpanded: () => setExpanded((v) => !v),
    }), [tasks, start, open, dismiss, expanded]);

    return (
        <TaskTrayContext.Provider value={value}>
            {children}
        </TaskTrayContext.Provider>
    );
};

export const useTaskTray = () => {
    const ctx = useContext(TaskTrayContext);
    if (!ctx) throw new Error('useTaskTray must be used inside <TaskTrayProvider>');
    return ctx;
};
