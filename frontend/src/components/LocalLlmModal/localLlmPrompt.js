// Imperative opener for the shared "set a local LLM" modal — the
// alternative to OpenAiKeyModal's key prompt (components/OpenAiKeyModal/
// openAiKeyPrompt.js), which this file deliberately mirrors.
//
// The modal is mounted once, globally, from App.jsx and subscribes here.
// Any surface offering "Set local LLM" next to its "Set API key" button
// calls promptForLocalLlm(): either one unblocks the same backend gate
// (utils/llm_access.py), so both buttons live on the same "AI features need
// a key or a local model" banner and either can satisfy it.
//
// This module imports nothing on purpose, same reasoning as
// openAiKeyPrompt.js: it has to stay at the bottom of the dependency graph.

const listeners = new Set();

export const subscribeLocalLlmPrompt = (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
};

// options.onSaved — called once the model is saved, so the surface that
// failed can retry itself.
export const promptForLocalLlm = (options = {}) => {
    listeners.forEach((listener) => listener(options));
};

// --- "a local model was just saved" ---------------------------------------
//
// Mirrors subscribeOpenAiKeySaved: every surface showing a "needs a key or a
// local model" note can re-check and clear it, without holding a reference
// to anybody else.

const savedListeners = new Set();

export const subscribeLocalLlmSaved = (listener) => {
    savedListeners.add(listener);
    return () => savedListeners.delete(listener);
};

export const notifyLocalLlmSaved = () => {
    savedListeners.forEach((listener) => listener());
};
