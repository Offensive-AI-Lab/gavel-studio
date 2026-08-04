// Contract tests for the missing-OpenAI-key plumbing in src/api.js:
//   * the two settings calls hit the documented endpoints,
//   * the response interceptor opens the shared modal ONLY on the exact
//     contract error (503 + detail.code === 'OPENAI_KEY_MISSING'),
//   * and it always re-rejects, so every existing per-page catch still runs.
// The axios instance is mocked; no network.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const mocks = vi.hoisted(() => {
    const instance = {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
        interceptors: { response: { use: vi.fn() } },
    };
    return { instance };
});

vi.mock('axios', () => ({
    default: { create: vi.fn(() => mocks.instance) },
}));

import * as api from '../src/api';
import { subscribeOpenAiKeyPrompt } from '../src/components/OpenAiKeyModal/openAiKeyPrompt';

// Captured at import time: api.js registers the interceptor as a side effect
// of being loaded, and the per-test mock resets must not lose it.
const [onFulfilled, onRejected] = mocks.instance.interceptors.response.use.mock.calls[0];

const keyMissingError = (status = 503) => ({
    response: { status, data: { detail: { code: 'OPENAI_KEY_MISSING', message: 'No OpenAI key is set.' } } },
});

let prompted;
let unsubscribe;

beforeEach(() => {
    mocks.instance.get.mockReset().mockResolvedValue({ data: {} });
    mocks.instance.put.mockReset().mockResolvedValue({ data: {} });
    prompted = vi.fn();
    unsubscribe = subscribeOpenAiKeyPrompt(prompted);
});

afterEach(() => {
    unsubscribe();
});

describe('api.js — OpenAI key settings calls', () => {
    it('getOpenAiKeyStatus reads GET /settings/openai-key', async () => {
        await api.getOpenAiKeyStatus();
        expect(mocks.instance.get).toHaveBeenCalledWith('/settings/openai-key');
    });

    it('saveOpenAiKey puts {api_key} to /settings/openai-key', async () => {
        await api.saveOpenAiKey('sk-test');
        expect(mocks.instance.put).toHaveBeenCalledWith('/settings/openai-key', { api_key: 'sk-test' });
    });
});

describe('api.js — missing-key response interceptor', () => {
    it('registers a pass-through success handler', () => {
        const response = { status: 200, data: { ok: true } };
        expect(onFulfilled(response)).toBe(response);
    });

    it('opens the key modal on 503 + OPENAI_KEY_MISSING', async () => {
        const error = keyMissingError();
        await expect(onRejected(error)).rejects.toBe(error);
        expect(prompted).toHaveBeenCalledTimes(1);
    });

    it('ignores the marker on any other status', async () => {
        const error = keyMissingError(500);
        await expect(onRejected(error)).rejects.toBe(error);
        expect(prompted).not.toHaveBeenCalled();
    });

    it('ignores a 503 without the marker (never string-matches the prose)', async () => {
        const error = { response: { status: 503, data: { detail: 'OpenAI key missing' } } };
        await expect(onRejected(error)).rejects.toBe(error);
        expect(prompted).not.toHaveBeenCalled();
    });

    it('stays quiet for a call that fired on its own (skipKeyPrompt)', async () => {
        const error = { ...keyMissingError(), config: { skipKeyPrompt: true } };
        await expect(onRejected(error)).rejects.toBe(error);
        expect(prompted).not.toHaveBeenCalled();
    });

    it('deriveScenario marks itself as self-firing', async () => {
        mocks.instance.post.mockReset().mockResolvedValue({ data: {} });
        await api.deriveScenario(7);
        expect(mocks.instance.post).toHaveBeenCalledWith(
            '/ai/derive-scenario', { rule_id: 7 }, { skipKeyPrompt: true },
        );
    });

    it('startScenarioChat marks itself self-firing only when the step opens it', async () => {
        mocks.instance.post.mockReset().mockResolvedValue({ data: {} });
        // The wizard's mount bootstrap — a note, never a modal.
        await api.startScenarioChat({ skipKeyPrompt: true });
        expect(mocks.instance.post).toHaveBeenLastCalledWith(
            '/ai/scenario-chat/start', undefined, { skipKeyPrompt: true },
        );
        // Restart is a click — it keeps the modal.
        await api.startScenarioChat();
        expect(mocks.instance.post).toHaveBeenLastCalledWith(
            '/ai/scenario-chat/start', undefined, undefined,
        );
    });

    it('re-rejects network errors that carry no response at all', async () => {
        const error = new Error('Network Error');
        await expect(onRejected(error)).rejects.toBe(error);
        expect(prompted).not.toHaveBeenCalled();
    });
});
