// Turn anything the backend can put in `detail` into text we can render.
//
// FastAPI's `detail` is not always a string: the missing-key contract error
// sends `{code, message}`, and a request-validation failure sends a list of
// `{loc, msg, type}` entries. React 19 throws "Objects are not valid as a
// React child" the moment one of those reaches JSX, and string concatenation
// turns it into "[object Object]". Everything that reads `detail` goes
// through here so neither can happen.

const asText = (value) => (typeof value === 'string' && value.trim() ? value.trim() : null);

const fromDetail = (detail) => {
    const direct = asText(detail);
    if (direct) return direct;
    if (Array.isArray(detail)) {
        // Pydantic validation errors: one entry per bad field.
        const parts = detail
            .map((entry) => asText(entry) || asText(entry?.msg) || asText(entry?.message))
            .filter(Boolean);
        return parts.length ? parts.join('; ') : null;
    }
    if (detail && typeof detail === 'object') {
        return asText(detail.message) || asText(detail.msg) || asText(detail.detail);
    }
    return null;
};

// The backend's reason for the failure, as a string — or null when it gave
// none. Use it where the call site already has its own fallback wording.
export const detailText = (source) => {
    if (typeof source === 'string') return asText(source);
    if (!source || typeof source !== 'object') return null;
    const body = source.response?.data ?? source.data ?? source;
    return fromDetail(body?.detail) ?? fromDetail(source.detail);
};

// The best message we can show for a failure: the backend's reason, then the
// exception's own message, then the caller's fallback.
export const errorText = (source, fallback = 'Something went wrong. Try again.') =>
    detailText(source) || asText(source?.message) || fallback;

// True when a stopped run was stopped BY THE USER. A cancelled run is written
// off exactly like a run the server lost — status 'error' (or
// 'needs_retraining' when a previous model survived), phase columns cleared,
// reason in `training_log` — so `training_log` is the only thing that tells
// them apart. A cancellation is not a failure, so the banners that read this
// say what happened in plain words instead of showing failure copy.
export const wasTrainingCancelled = (trainingLog) =>
    typeof trainingLog === 'string' && trainingLog.toLowerCase().includes('cancel');
