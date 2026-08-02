// Helpers shared by every step component in the wizard.

// Pull the persisted state for a given step (or the default if absent).
// Step components always treat the run as read-only and patch through
// `onPatchStep` — we never mutate `run.steps[X].data` in place.
export function getStepState(run, stepId) {
    return (run?.steps && run.steps[stepId]) || { status: 'pending', data: {} };
}

// Mark a step started. Components call this at the top of any
// LLM-triggering action so the sidebar shows the "in_progress" pill.
export function startStep(onPatchStep, stepId, data) {
    return onPatchStep(stepId, { status: 'in_progress', data });
}

// Mark a step completed with its final payload.
export function completeStep(onPatchStep, stepId, data) {
    return onPatchStep(stepId, { status: 'completed', data });
}

// Mark a step errored. The sidebar shows a red badge; the user can
// retry without leaving the step.
export function errorStep(onPatchStep, stepId, errorMessage, data = {}) {
    return onPatchStep(stepId, {
        status: 'error',
        data: { ...data, error: errorMessage },
    });
}


// Shared visual primitives — these are styled inline because the
// step components are tightly coupled to the wizard's frame styles
// in RuleWizard.jsx, and pulling a stylesheet for ~80 lines of
// border + radius declarations would be more weight than value.

export const card = {
    background: 'rgba(15, 23, 42, 0.50)',
    border: '1px solid rgba(148, 163, 184, 0.14)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 14,
    boxShadow: '0 8px 24px -16px rgba(2, 6, 23, 0.8)',
};

export const muted = { color: '#94a3b8' };

export const primaryBtn = {
    padding: '8px 16px',
    borderRadius: 10,
    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
    color: '#fff', fontWeight: 600,
    border: 'none', cursor: 'pointer',
};

export const secondaryBtn = {
    padding: '8px 16px',
    borderRadius: 10,
    background: 'rgba(148, 163, 184, 0.15)',
    color: '#cbd5e1', fontWeight: 600,
    border: '1px solid rgba(148, 163, 184, 0.30)',
    cursor: 'pointer',
};

export const fieldStyle = {
    width: '100%',
    padding: '10px 12px',
    background: 'rgba(15, 23, 42, 0.55)',
    border: '1px solid rgba(148, 163, 184, 0.25)',
    borderRadius: 8,
    color: '#e2e8f0',
    fontFamily: 'inherit',
    fontSize: 14,
};

export const successBanner = {
    padding: '10px 14px',
    background: 'rgba(16, 185, 129, 0.10)',
    border: '1px solid rgba(16, 185, 129, 0.35)',
    color: '#6ee7b7',
    borderRadius: 8,
    fontSize: 13,
    display: 'flex', alignItems: 'center', gap: 8,
};

export const errorBanner = {
    padding: '10px 14px',
    background: 'rgba(239, 68, 68, 0.10)',
    border: '1px solid rgba(239, 68, 68, 0.35)',
    color: '#fca5a5',
    borderRadius: 8,
    fontSize: 13,
    display: 'flex', alignItems: 'center', gap: 8,
};
