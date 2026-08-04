// Imperative opener for the shared "set the OpenAI key" modal.
//
// The modal is mounted once, globally, from App.jsx and subscribes here.
// Anything that discovers the key is missing calls promptForOpenAiKey():
// the axios interceptor in api.js does it for the 503 contract error, and
// the surfaces that never throw (200 + success:false, polled status text,
// task-tray chips) can call it directly.
//
// This module imports nothing on purpose — api.js and the modal both depend
// on it, so it has to stay at the bottom of the dependency graph.

export const OPENAI_KEY_MISSING = 'OPENAI_KEY_MISSING';

const listeners = new Set();

export const subscribeOpenAiKeyPrompt = (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
};

// options.onSaved — called once the key is saved, so the surface that failed
// can retry itself.
export const promptForOpenAiKey = (options = {}) => {
    listeners.forEach((listener) => listener(options));
};

// --- "a key was just saved" ---------------------------------------------
//
// A separate channel from the prompt above, because they answer different
// questions. `promptForOpenAiKey({ onSaved })` belongs to the ONE call that
// failed and wants to retry itself. This one is a broadcast: every surface
// that is showing a "no key yet" note can re-check and clear it, without
// anybody holding a reference to anybody else.

const savedListeners = new Set();

export const subscribeOpenAiKeySaved = (listener) => {
    savedListeners.add(listener);
    return () => savedListeners.delete(listener);
};

export const notifyOpenAiKeySaved = () => {
    savedListeners.forEach((listener) => listener());
};

// Does this carry the backend's missing-key marker? Accepts an axios error,
// an axios response, or a plain response body — never match on error prose.
export const isOpenAiKeyMissing = (source) => {
    if (!source || typeof source !== 'object') return false;
    const body = source.response?.data ?? source.data ?? source;
    if (!body || typeof body !== 'object') return false;
    return body.detail?.code === OPENAI_KEY_MISSING || body.code === OPENAI_KEY_MISSING;
};

// The human sentence the backend sent with the marker, if there is one.
export const openAiKeyMissingMessage = (source) => {
    const body = source?.response?.data ?? source?.data ?? source;
    const detail = body?.detail;
    return (typeof detail?.message === 'string' && detail.message) || null;
};
