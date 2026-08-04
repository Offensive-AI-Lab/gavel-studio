// Tests for Step1CEChat — the conversational CE step. It drives generateCe
// with the running concept + clarification history, renders replies as chat
// bubbles, shows the proposed CE for review, and "Approve & Build" hands off
// to the wizard's onAdvance.
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

const { generateCe, getOpenAiKeyStatus } = vi.hoisted(() => ({
    generateCe: vi.fn(),
    getOpenAiKeyStatus: vi.fn(),
}));
vi.mock('../../../src/api', () => ({ generateCe, getOpenAiKeyStatus }));

// The shared key-modal opener — stubbed so we can assert when it is asked to
// open and drive its onSaved callback without mounting the modal.
vi.mock('../../../src/components/OpenAiKeyModal/openAiKeyPrompt', async (importOriginal) => {
    const actual = await importOriginal();
    return { ...actual, promptForOpenAiKey: vi.fn() };
});

import Step1CEChat from '../../../src/pages/CEWizardSteps/Step1CEChat';
import {
    notifyOpenAiKeySaved,
    promptForOpenAiKey,
} from '../../../src/components/OpenAiKeyModal/openAiKeyPrompt';

// The backend's contract error for a missing/refused key.
const KEY_MESSAGE = 'This feature needs an OpenAI key. Add yours to continue.';
// The step's own wording, shown upfront when no key is set at all.
const KEY_MISSING_TEXT = 'This step needs an OpenAI key. Add yours to continue.';
const keyMissingError = () => ({
    response: { status: 503, data: { detail: { code: 'OPENAI_KEY_MISSING', message: KEY_MESSAGE } } },
});

const renderChat = (overrides = {}) => {
    const onPatchStep = overrides.onPatchStep || vi.fn(() => Promise.resolve({}));
    const onAdvance = overrides.onAdvance || vi.fn(() => Promise.resolve());
    const run = overrides.run || { steps: {} };
    const utils = render(<Step1CEChat run={run} onPatchStep={onPatchStep} onAdvance={onAdvance} />);
    return { onPatchStep, onAdvance, ...utils };
};

const typeAndSend = (text) => {
    fireEvent.change(screen.getByPlaceholderText(/Describe the concept|Answer the question/i), { target: { value: text } });
    fireEvent.click(screen.getByRole('button', { name: /Send/i }));
};

beforeEach(() => {
    vi.clearAllMocks();
    getOpenAiKeyStatus.mockResolvedValue({ data: { configured: true } });
});

describe('Step1CEChat', () => {
    it('renders the greeting and an input', () => {
        renderChat();
        expect(screen.getByText(/Describe the Cognitive Element you want to capture/i)).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/Describe the concept/i)).toBeInTheDocument();
    });

    it('auto-focuses the input on load and after a reply lands', async () => {
        generateCe.mockResolvedValueOnce({ data: { needs_clarification: true, clarification_question: 'ACTION or CONTEXT?' } });
        renderChat();
        const input = screen.getByPlaceholderText(/Describe the concept/i);
        await waitFor(() => expect(input).toHaveFocus());
        typeAndSend('kill people');
        await screen.findByText('ACTION or CONTEXT?');
        // Refocused automatically so the user can answer without clicking.
        await waitFor(() => expect(input).toHaveFocus());
    });

    it('asks a clarifying question as a chat bubble', async () => {
        generateCe.mockResolvedValueOnce({ data: { needs_clarification: true, clarification_question: 'ACTION or CONTEXT?' } });
        renderChat();
        typeAndSend('kill people');
        // The user's message + the assistant's clarification both render.
        expect(await screen.findByText('kill people')).toBeInTheDocument();
        expect(await screen.findByText('ACTION or CONTEXT?')).toBeInTheDocument();
        // First call: concept as description, empty history.
        expect(generateCe).toHaveBeenCalledWith('kill people', null, []);
    });

    it('sends the clarification answer with accumulated history, then shows the proposed CE', async () => {
        generateCe
            .mockResolvedValueOnce({ data: { needs_clarification: true, clarification_question: 'ACTION or CONTEXT?' } })
            .mockResolvedValueOnce({ data: { ce_data: { name: 'incites_violence', type: 'ACTION', definition: 'Encourages harm.' } } });
        const { onPatchStep } = renderChat();

        typeAndSend('kill people');
        await screen.findByText('ACTION or CONTEXT?');
        typeAndSend('the action');

        // Second call carries the Q&A history.
        await waitFor(() => expect(generateCe).toHaveBeenLastCalledWith('kill people', null, [
            { question: 'ACTION or CONTEXT?', answer: 'the action' },
        ]));
        // The proposed CE renders + the Approve button appears.
        expect(await screen.findByText('incites_violence')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Approve & Build/i })).toBeInTheDocument();
        // The proposal was persisted as the completed step-1 data.
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', expect.objectContaining({
            status: 'completed',
            data: expect.objectContaining({ ce_data: expect.objectContaining({ name: 'incites_violence' }) }),
        })));
    });

    it('Approve & Build calls onAdvance', async () => {
        generateCe.mockResolvedValueOnce({ data: { ce_data: { name: 'x', type: 'ACTION', definition: 'd' } } });
        const { onAdvance } = renderChat();
        typeAndSend('some concept');
        const approve = await screen.findByRole('button', { name: /Approve & Build/i });
        fireEvent.click(approve);
        await waitFor(() => expect(onAdvance).toHaveBeenCalled());
    });

    it('offers "Set API key" on the 503 key error and replays the turn once it is saved', async () => {
        generateCe.mockRejectedValueOnce(keyMissingError());
        renderChat();
        typeAndSend('some concept');

        expect(await screen.findByText(KEY_MESSAGE)).toBeInTheDocument();
        const btn = screen.getByRole('button', { name: /Set API key/i });
        // The axios interceptor already opened the modal for this one — the
        // banner must not open a second copy on its own.
        expect(promptForOpenAiKey).not.toHaveBeenCalled();

        generateCe.mockResolvedValueOnce({ data: { ce_data: { name: 'after_key', type: 'ACTION', definition: 'd' } } });
        fireEvent.click(btn);
        expect(promptForOpenAiKey).toHaveBeenCalledTimes(1);

        promptForOpenAiKey.mock.calls[0][0].onSaved();
        expect(await screen.findByText('after_key')).toBeInTheDocument();
        // Same turn, same arguments.
        expect(generateCe).toHaveBeenLastCalledWith('some concept', null, []);
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
    });

    it('opens the key modal itself when a 200 body carries the marker', async () => {
        // No axios error → the interceptor never sees this one.
        generateCe.mockResolvedValueOnce({ data: { success: false, detail: { code: 'OPENAI_KEY_MISSING', message: KEY_MESSAGE } } });
        renderChat();
        typeAndSend('some concept');

        expect(await screen.findByText(KEY_MESSAGE)).toBeInTheDocument();
        await waitFor(() => expect(promptForOpenAiKey).toHaveBeenCalledTimes(1));
        expect(screen.getByRole('button', { name: /Set API key/i })).toBeInTheDocument();
    });

    it('shows an ordinary failure with no key action', async () => {
        generateCe.mockRejectedValueOnce({ response: { status: 500, data: { detail: 'generator exploded' } } });
        renderChat();
        typeAndSend('some concept');

        expect(await screen.findByText('generator exploded')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
        expect(promptForOpenAiKey).not.toHaveBeenCalled();
    });

    it('shows the refusal reason when the concept is already covered', async () => {
        generateCe.mockResolvedValueOnce({ data: { refuse: true, refuse_reason: 'Already covered by an existing CE.' } });
        renderChat();
        typeAndSend('duplicate concept');
        expect(await screen.findByText('Already covered by an existing CE.')).toBeInTheDocument();
        // No proposal → input stays available.
        expect(screen.getByPlaceholderText(/Answer the question|Describe the concept/i)).toBeInTheDocument();
    });
});

describe('Step1CEChat — missing OpenAI key', () => {
    it('shows the note + Set API key as soon as the step opens, and opens nothing', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        renderChat();

        expect(await screen.findByText(KEY_MISSING_TEXT)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Set API key/i })).toBeInTheDocument();
        // The modal is action-triggered: opening the wizard is not an action.
        expect(promptForOpenAiKey).not.toHaveBeenCalled();
        expect(generateCe).not.toHaveBeenCalled();
    });

    it('leaves the chat usable — the note never blocks typing', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        renderChat();
        await screen.findByText(KEY_MISSING_TEXT);

        const input = screen.getByPlaceholderText(/Describe the concept/i);
        expect(input).not.toBeDisabled();
        fireEvent.change(input, { target: { value: 'still typable' } });
        expect(input).toHaveValue('still typable');
        expect(screen.getByRole('button', { name: /Send/i })).not.toBeDisabled();
    });

    it('shows nothing when a key is configured', async () => {
        renderChat();
        await waitFor(() => expect(getOpenAiKeyStatus).toHaveBeenCalled());
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
    });

    it('stays quiet when the status call itself fails', async () => {
        getOpenAiKeyStatus.mockRejectedValue(new Error('backend down'));
        renderChat();
        await waitFor(() => expect(getOpenAiKeyStatus).toHaveBeenCalled());
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
    });

    it('asks for the key when the user presses Set API key', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        renderChat();
        await screen.findByText(KEY_MISSING_TEXT);

        fireEvent.click(screen.getByRole('button', { name: /Set API key/i }));
        expect(promptForOpenAiKey).toHaveBeenCalledTimes(1);
    });

    it('clears the note by itself once a key is saved', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        renderChat();
        await screen.findByText(KEY_MISSING_TEXT);

        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: true } });
        await act(async () => { notifyOpenAiKeySaved(); });

        await waitFor(() => expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull());
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
    });

    it('a send with no key still reports the backend message over the upfront note', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        generateCe.mockRejectedValueOnce({
            response: { status: 503, data: { detail: { code: 'OPENAI_KEY_MISSING', message: KEY_MESSAGE } } },
        });
        renderChat();
        await screen.findByText(KEY_MISSING_TEXT);

        typeAndSend('some concept');
        expect(await screen.findByText(KEY_MESSAGE)).toBeInTheDocument();
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
        expect(screen.getByRole('button', { name: /Set API key/i })).toBeInTheDocument();
    });
});
