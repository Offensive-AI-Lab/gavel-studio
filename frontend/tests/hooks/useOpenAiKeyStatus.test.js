// Tests for useOpenAiKeyStatus — the upfront "is an OpenAI key set?" check.
//
// The contract this file pins down:
//   * it asks GET /settings/openai-key once on mount and reports the answer,
//   * a status call that FAILS reads as configured (a backend blip must never
//     tell the user their key is missing),
//   * it never opens the key modal — prompting stays user-triggered,
//   * and it re-checks when the modal broadcasts a successful save, so a note
//     put up by this hook clears itself.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';

vi.mock('../../src/api', () => ({ getOpenAiKeyStatus: vi.fn() }));

import useOpenAiKeyStatus from '../../src/hooks/useOpenAiKeyStatus';
import { getOpenAiKeyStatus } from '../../src/api';
import {
    notifyOpenAiKeySaved,
    subscribeOpenAiKeyPrompt,
} from '../../src/components/OpenAiKeyModal/openAiKeyPrompt';

const status = (configured) => ({ data: { configured } });

let prompted;
let unsubscribe;

beforeEach(() => {
    getOpenAiKeyStatus.mockReset().mockResolvedValue(status(true));
    prompted = vi.fn();
    unsubscribe = subscribeOpenAiKeyPrompt(prompted);
});

afterEach(() => { unsubscribe(); });

describe('useOpenAiKeyStatus', () => {
    it('reports an unconfigured key after the check lands', async () => {
        getOpenAiKeyStatus.mockResolvedValue(status(false));
        const { result } = renderHook(() => useOpenAiKeyStatus());

        // Optimistic until the answer is in — nothing red flashes on mount.
        expect(result.current.checked).toBe(false);
        expect(result.current.configured).toBe(true);

        await waitFor(() => expect(result.current.checked).toBe(true));
        expect(result.current.configured).toBe(false);
        expect(getOpenAiKeyStatus).toHaveBeenCalledTimes(1);
    });

    it('reports a configured key', async () => {
        getOpenAiKeyStatus.mockResolvedValue(status(true));
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.checked).toBe(true));
        expect(result.current.configured).toBe(true);
    });

    it('treats a failed status call as configured', async () => {
        getOpenAiKeyStatus.mockRejectedValue(new Error('Network Error'));
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.checked).toBe(true));
        expect(result.current.configured).toBe(true);
    });

    it('treats a body without the field as configured', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: {} });
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.checked).toBe(true));
        expect(result.current.configured).toBe(true);
    });

    it('never asks for the key itself', async () => {
        getOpenAiKeyStatus.mockResolvedValue(status(false));
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.configured).toBe(false));
        expect(prompted).not.toHaveBeenCalled();
    });

    it('re-checks when a key is saved, so the answer flips without a reload', async () => {
        getOpenAiKeyStatus.mockResolvedValue(status(false));
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.configured).toBe(false));

        getOpenAiKeyStatus.mockResolvedValue(status(true));
        await act(async () => { notifyOpenAiKeySaved(); });

        await waitFor(() => expect(result.current.configured).toBe(true));
        expect(getOpenAiKeyStatus).toHaveBeenCalledTimes(2);
    });

    it('unsubscribes on unmount — a later save does not re-check', async () => {
        const { result, unmount } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.checked).toBe(true));
        unmount();

        await act(async () => { notifyOpenAiKeySaved(); });
        expect(getOpenAiKeyStatus).toHaveBeenCalledTimes(1);
    });

    it('refresh() asks again on demand', async () => {
        const { result } = renderHook(() => useOpenAiKeyStatus());
        await waitFor(() => expect(result.current.checked).toBe(true));

        getOpenAiKeyStatus.mockResolvedValue(status(false));
        await act(async () => { await result.current.refresh(); });

        expect(result.current.configured).toBe(false);
        expect(getOpenAiKeyStatus).toHaveBeenCalledTimes(2);
    });
});
