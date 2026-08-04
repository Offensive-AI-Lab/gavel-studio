// Behavior tests for Step1Scenario — the scenario-ideation chat step of the
// rule wizard. It's a self-contained component: props `run` + `onPatchStep`,
// and the only network surface is startScenarioChat / sendScenarioChatMessage
// from ../../api. We mock that module so nothing hits the network.

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// --- Mock ../../api. Only the scenario chat + key-status exports are used by
// this file, but we keep the mock minimal + benign.
const startScenarioChat = vi.fn();
const sendScenarioChatMessage = vi.fn();
const getOpenAiKeyStatus = vi.fn();
vi.mock('../../../src/api', () => ({
    startScenarioChat: (...a) => startScenarioChat(...a),
    sendScenarioChatMessage: (...a) => sendScenarioChatMessage(...a),
    getOpenAiKeyStatus: (...a) => getOpenAiKeyStatus(...a),
}));

import Step1Scenario from '../../../src/pages/RuleWizardSteps/Step1Scenario';
import {
    notifyOpenAiKeySaved,
    subscribeOpenAiKeyPrompt,
} from '../../../src/components/OpenAiKeyModal/openAiKeyPrompt';

// The step's own wording for a missing key (KEY_MISSING_TEXT in the component).
const KEY_MISSING_TEXT = 'This step needs an OpenAI key. Add yours to continue.';

// The real prompt channel — every test can assert whether the modal was asked
// for, without mocking the module the component imports.
let prompted;
let unsubscribePrompt;

// Default benign responses; individual tests override as needed.
beforeEach(() => {
    vi.clearAllMocks();
    startScenarioChat.mockResolvedValue({ data: { session_id: 'sess-1', message: 'Hi, describe your scenario' } });
    sendScenarioChatMessage.mockResolvedValue({ data: { message: 'Tell me more', is_final: false } });
    getOpenAiKeyStatus.mockResolvedValue({ data: { configured: true } });
    prompted = vi.fn();
    unsubscribePrompt = subscribeOpenAiKeyPrompt(prompted);
});

afterEach(() => { unsubscribePrompt(); });

// Helper: render with a run object + a captured onPatchStep spy.
function setup(run = { steps: {} }, onPatchStep = vi.fn(() => Promise.resolve()), onAdvance = vi.fn(() => Promise.resolve())) {
    const utils = render(<Step1Scenario run={run} onPatchStep={onPatchStep} onAdvance={onAdvance} />);
    return { onPatchStep, onAdvance, ...utils };
}

describe('Step1Scenario — bootstrap', () => {
    it('starts a chat on mount when no session exists and renders the assistant greeting', async () => {
        const { onPatchStep } = setup();
        await waitFor(() => expect(startScenarioChat).toHaveBeenCalledTimes(1));
        expect(await screen.findByText('Hi, describe your scenario')).toBeInTheDocument();
        // startStep -> onPatchStep('1', { status: 'in_progress', data: {...} })
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', expect.objectContaining({
            status: 'in_progress',
            data: expect.objectContaining({
                session_id: 'sess-1',
                messages: [{ role: 'assistant', content: 'Hi, describe your scenario' }],
            }),
        })));
    });

    it('renders the Restart button and the chat header', async () => {
        setup();
        expect(await screen.findByText('Scenario Chat')).toBeInTheDocument();
        expect(screen.getByText('Restart')).toBeInTheDocument();
    });

    it('does NOT start a chat when a session already exists in step data', async () => {
        const run = { steps: { 1: { status: 'in_progress', data: { session_id: 'existing', messages: [{ role: 'assistant', content: 'restored' }] } } } };
        setup(run);
        expect(await screen.findByText('restored')).toBeInTheDocument();
        expect(startScenarioChat).not.toHaveBeenCalled();
    });

    it('does NOT start a chat when the step is already completed/finalized', async () => {
        const run = { steps: { 1: { status: 'completed', data: { description: 'final desc', name: 'final_name' } } } };
        setup(run);
        // finalized panel renders without a session bootstrap
        expect(await screen.findByText(/Scenario finalized/i)).toBeInTheDocument();
        expect(startScenarioChat).not.toHaveBeenCalled();
    });

    it('shows an error banner and does not crash when startScenarioChat rejects', async () => {
        startScenarioChat.mockRejectedValueOnce(new Error('boom'));
        setup();
        // The failure is surfaced to the user, not just the console.
        expect(await screen.findByText('boom')).toBeInTheDocument();
        // Header still present; Restart remains available as the retry path.
        expect(screen.getByText('Scenario Chat')).toBeInTheDocument();
        expect(screen.getByText('Restart')).toBeInTheDocument();
    });

    it('shows the backend detail when startScenarioChat rejects with an axios-style error', async () => {
        startScenarioChat.mockRejectedValueOnce({
            response: { data: { detail: 'LLM error: AuthenticationError - no api key' } },
        });
        setup();
        expect(await screen.findByText('LLM error: AuthenticationError - no api key')).toBeInTheDocument();
    });

    it('renders the missing-key detail object as text when the bootstrap fails', async () => {
        startScenarioChat.mockRejectedValueOnce({
            response: {
                status: 503,
                data: { detail: { code: 'OPENAI_KEY_MISSING', message: 'No API key is set.' } },
            },
        });
        setup();
        expect(await screen.findByText('No API key is set.')).toBeInTheDocument();
        expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument();
        expect(screen.getByText(/Set API key/i)).toBeInTheDocument();
    });

    it('Restart clears a bootstrap error and starts a fresh session', async () => {
        startScenarioChat.mockRejectedValueOnce(new Error('boom'));
        setup();
        await screen.findByText('boom');
        // Second attempt (via Restart) succeeds — banner goes away.
        fireEvent.click(screen.getByText('Restart'));
        expect(await screen.findByText('Hi, describe your scenario')).toBeInTheDocument();
        expect(screen.queryByText('boom')).not.toBeInTheDocument();
    });
});

describe('Step1Scenario — sending messages', () => {
    it('disables the input and send button until a session is established', async () => {
        // Keep startScenarioChat pending so no session yet.
        startScenarioChat.mockReturnValueOnce(new Promise(() => {}));
        setup();
        const input = await screen.findByPlaceholderText(/Describe the misuse/i);
        expect(input).toBeDisabled();
    });

    it('send button is disabled when input is empty/whitespace and enabled with text', async () => {
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        const sendBtn = screen.getByRole('button', { name: /Send/i });
        expect(sendBtn).toBeDisabled();
        fireEvent.change(input, { target: { value: '   ' } });
        expect(sendBtn).toBeDisabled();
        fireEvent.change(input, { target: { value: 'a real message' } });
        expect(sendBtn).not.toBeDisabled();
    });

    it('sends a message via the Send button, shows user + assistant bubbles, clears input', async () => {
        const user = userEvent.setup();
        const { onPatchStep } = setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        await user.type(input, 'catch medical advice');
        await user.click(screen.getByRole('button', { name: /Send/i }));

        await waitFor(() => expect(sendScenarioChatMessage).toHaveBeenCalledWith('sess-1', 'catch medical advice'));
        expect(await screen.findByText('catch medical advice')).toBeInTheDocument();
        expect(await screen.findByText('Tell me more')).toBeInTheDocument();
        expect(input).toHaveValue('');
        // non-final -> in_progress patch
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', expect.objectContaining({ status: 'in_progress' })));
    });

    it('auto-focuses the input after a reply lands so the user can keep typing', async () => {
        const user = userEvent.setup();
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        // Focus the input on bootstrap (session ready, not finalized).
        await waitFor(() => expect(input).toHaveFocus());
        await user.type(input, 'catch medical advice');
        // Clicking the button blurs the input; after the reply it should be
        // refocused automatically (no manual click needed to type again).
        await user.click(screen.getByRole('button', { name: /Send/i }));
        await screen.findByText('Tell me more');
        await waitFor(() => expect(input).toHaveFocus());
    });

    it('sends a message by pressing Enter', async () => {
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'enter key send' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await waitFor(() => expect(sendScenarioChatMessage).toHaveBeenCalledWith('sess-1', 'enter key send'));
    });

    it('does not send on a non-Enter key', async () => {
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'no send' } });
        fireEvent.keyDown(input, { key: 'a' });
        expect(sendScenarioChatMessage).not.toHaveBeenCalled();
    });

    it('does not send when input trims to empty', async () => {
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: '    ' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(sendScenarioChatMessage).not.toHaveBeenCalled();
    });

    it('shows an error banner and re-enables sending when sendScenarioChatMessage rejects', async () => {
        sendScenarioChatMessage.mockRejectedValueOnce(new Error('net'));
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'will fail' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        // The failure is visible to the user.
        expect(await screen.findByText('net')).toBeInTheDocument();
        // user bubble still shown
        expect(screen.getByText('will fail')).toBeInTheDocument();
        // input re-enabled (sending reset to false) — the failure is non-blocking
        await waitFor(() => expect(input).not.toBeDisabled());
    });

    it('shows the backend detail in the banner when the send fails with an axios-style error', async () => {
        sendScenarioChatMessage.mockRejectedValueOnce({
            response: { data: { detail: 'LLM error: AuthenticationError - The api_key client option must be set' } },
        });
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'will fail' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('LLM error: AuthenticationError - The api_key client option must be set')).toBeInTheDocument();
    });

    it('clears the error banner when the next send attempt starts', async () => {
        sendScenarioChatMessage.mockRejectedValueOnce({
            response: { data: { detail: 'LLM error: transient' } },
        });
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'first try' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('LLM error: transient');
        // Retry with the default (successful) mock — the banner goes away.
        fireEvent.change(input, { target: { value: 'second try' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('Tell me more')).toBeInTheDocument();
        expect(screen.queryByText('LLM error: transient')).not.toBeInTheDocument();
    });

    it('shows a Thinking… indicator while a send is in flight and hides it after', async () => {
        let resolveReply;
        sendScenarioChatMessage.mockReturnValueOnce(new Promise((res) => { resolveReply = res; }));
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'slow one' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(await screen.findByText('Thinking…')).toBeInTheDocument();
        resolveReply({ data: { message: 'finally', is_final: false } });
        expect(await screen.findByText('finally')).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByText('Thinking…')).not.toBeInTheDocument());
    });
});

describe('Step1Scenario — finalization', () => {
    it('shows the finalized panel and uses the AI-proposed scenario_name', async () => {
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: {
                message: 'Great, finalized!',
                is_final: true,
                scenario_description: 'Model Gives Medical Advice! Without, any disclaimer text here.',
                scenario_name: 'unqualified_medical_advice',
            },
        });
        const { onPatchStep } = setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'finalize please' } });
        fireEvent.keyDown(input, { key: 'Enter' });

        expect(await screen.findByText(/Scenario finalized/i)).toBeInTheDocument();
        // Name comes from the AI-proposed scenario_name, not a slug of the text.
        const nameInput = screen.getByPlaceholderText(/medical_advice_without_disclaimer/i);
        expect(nameInput).toHaveValue('unqualified_medical_advice');
        // description textarea populated
        const desc = screen.getByText('Description').parentElement.querySelector('textarea');
        expect(desc).toHaveValue('Model Gives Medical Advice! Without, any disclaimer text here.');
        // completed patch issued
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', expect.objectContaining({ status: 'completed' })));
        // chat input no longer rendered once finalized
        expect(screen.queryByPlaceholderText(/Describe the misuse/i)).not.toBeInTheDocument();
    });

    it('auto-advances to the next step once the scenario is finalized', async () => {
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: { message: 'done', is_final: true, scenario_description: 'desc', scenario_name: 'n' },
        });
        const { onAdvance } = setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await waitFor(() => expect(onAdvance).toHaveBeenCalled());
    });

    it('does NOT auto-advance while the chat is still in progress', async () => {
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: { message: 'a follow-up question?', is_final: false, scenario_description: null },
        });
        const { onAdvance } = setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('a follow-up question?');
        expect(onAdvance).not.toHaveBeenCalled();
    });

    it('falls back to a content-word slug when no scenario_name is given', async () => {
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: {
                message: 'done',
                is_final: true,
                scenario_description: 'Model Gives Medical Advice! Without, any disclaimer.',
            },
        });
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText(/Scenario finalized/i);
        // Fallback: first 4 content words, lowercased, punctuation stripped.
        const nameInput = screen.getByPlaceholderText(/medical_advice_without_disclaimer/i);
        expect(nameInput).toHaveValue('model_gives_medical_advice');
    });

    it('does not overwrite an existing name when finalizing', async () => {
        const run = { steps: { 1: { status: 'in_progress', data: { session_id: 'sess-1', messages: [{ role: 'assistant', content: 'hi' }], name: 'kept_name' } } } };
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: { message: 'done', is_final: true, scenario_description: 'Some New Description' },
        });
        setup(run);
        await screen.findByText('hi');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText(/Scenario finalized/i);
        const nameInput = screen.getByPlaceholderText(/medical_advice_without_disclaimer/i);
        expect(nameInput).toHaveValue('kept_name');
    });

    it('does not finalize when is_final is true but description is missing', async () => {
        sendScenarioChatMessage.mockResolvedValueOnce({
            data: { message: 'almost', is_final: true, scenario_description: '' },
        });
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'go' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await screen.findByText('almost');
        // still in chat mode, no finalized banner
        expect(screen.queryByText(/Scenario finalized/i)).not.toBeInTheDocument();
        expect(screen.getByPlaceholderText(/Describe the misuse/i)).toBeInTheDocument();
    });

    it('renders the finalized panel directly when the run starts completed', async () => {
        const run = { steps: { 1: { status: 'completed', data: { description: 'restored desc', name: 'restored_name' } } } };
        setup(run);
        const nameInput = await screen.findByPlaceholderText(/medical_advice_without_disclaimer/i);
        expect(nameInput).toHaveValue('restored_name');
        const desc = screen.getByText('Description').parentElement.querySelector('textarea');
        expect(desc).toHaveValue('restored desc');
    });

    it('lets the user edit name + description and persists overrides via Save edits', async () => {
        const run = { steps: { 1: { status: 'completed', data: { session_id: 'sx', messages: [{ role: 'assistant', content: 'm' }], description: 'd', name: 'n' } } } };
        const { onPatchStep } = setup(run);
        const nameInput = await screen.findByPlaceholderText(/medical_advice_without_disclaimer/i);
        fireEvent.change(nameInput, { target: { value: 'edited_name' } });
        const desc = screen.getByText('Description').parentElement.querySelector('textarea');
        fireEvent.change(desc, { target: { value: 'edited description' } });
        fireEvent.click(screen.getByRole('button', { name: /Save edits/i }));
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', {
            status: 'completed',
            data: {
                session_id: 'sx',
                messages: [{ role: 'assistant', content: 'm' }],
                description: 'edited description',
                name: 'edited_name',
            },
        }));
    });
});

describe('Step1Scenario — restart', () => {
    it('restarts the chat, resetting state and starting a fresh session', async () => {
        const run = { steps: { 1: { status: 'completed', data: { session_id: 'old', description: 'old desc', name: 'old_name' } } } };
        // Restart calls startScenarioChat again.
        startScenarioChat.mockResolvedValue({ data: { session_id: 'fresh', message: 'fresh greeting' } });
        const { onPatchStep } = setup(run);
        // finalized panel up first
        await screen.findByText(/Scenario finalized/i);
        fireEvent.click(screen.getByText('Restart'));
        expect(await screen.findByText('fresh greeting')).toBeInTheDocument();
        // back in chat mode
        expect(screen.getByPlaceholderText(/Describe the misuse/i)).toBeInTheDocument();
        await waitFor(() => expect(onPatchStep).toHaveBeenCalledWith('1', expect.objectContaining({
            status: 'in_progress',
            data: expect.objectContaining({ session_id: 'fresh', description: '', name: '' }),
        })));
    });
});

describe('Step1Scenario — missing OpenAI key', () => {
    it('shows the note + Set API key as soon as the step opens, and opens nothing', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        setup();

        expect(await screen.findByText(KEY_MISSING_TEXT)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Set API key/i })).toBeInTheDocument();
        // The modal is action-triggered: opening the wizard is not an action.
        expect(prompted).not.toHaveBeenCalled();
    });

    it('leaves the chat usable — the note never blocks typing', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        setup();
        await screen.findByText(KEY_MISSING_TEXT);

        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        expect(input).not.toBeDisabled();
        fireEvent.change(input, { target: { value: 'still typable' } });
        expect(input).toHaveValue('still typable');
        expect(screen.getByRole('button', { name: /Send/i })).not.toBeDisabled();
    });

    it('shows nothing when a key is configured', async () => {
        setup();
        await screen.findByText('Hi, describe your scenario');
        await waitFor(() => expect(getOpenAiKeyStatus).toHaveBeenCalled());
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
    });

    it('stays quiet when the status call itself fails', async () => {
        getOpenAiKeyStatus.mockRejectedValue(new Error('backend down'));
        setup();
        await screen.findByText('Hi, describe your scenario');
        await waitFor(() => expect(getOpenAiKeyStatus).toHaveBeenCalled());
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
    });

    it('does not nag on a finalized step', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        const run = { steps: { 1: { status: 'completed', data: { description: 'd', name: 'n' } } } };
        setup(run);
        await screen.findByText(/Scenario finalized/i);
        await waitFor(() => expect(getOpenAiKeyStatus).toHaveBeenCalled());
        expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull();
    });

    it('the mount bootstrap fires on its own, so it never throws the modal at the user', async () => {
        setup();
        await waitFor(() => expect(startScenarioChat).toHaveBeenCalledWith({ skipKeyPrompt: true }));
        // Restart IS a user action — it keeps the modal.
        fireEvent.click(screen.getByText('Restart'));
        await waitFor(() => expect(startScenarioChat).toHaveBeenLastCalledWith());
    });

    it('asks for the key when the user presses Set API key', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        setup();
        await screen.findByText(KEY_MISSING_TEXT);

        fireEvent.click(screen.getByRole('button', { name: /Set API key/i }));
        expect(prompted).toHaveBeenCalledTimes(1);
    });

    it('clears the note by itself once a key is saved', async () => {
        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: false } });
        setup();
        await screen.findByText(KEY_MISSING_TEXT);

        getOpenAiKeyStatus.mockResolvedValue({ data: { configured: true } });
        await act(async () => { notifyOpenAiKeySaved(); });

        await waitFor(() => expect(screen.queryByText(KEY_MISSING_TEXT)).toBeNull());
        expect(screen.queryByRole('button', { name: /Set API key/i })).toBeNull();
    });

    it('a failed send still shows the backend message and offers the key', async () => {
        sendScenarioChatMessage.mockRejectedValueOnce({
            response: {
                status: 503,
                data: { detail: { code: 'OPENAI_KEY_MISSING', message: 'No API key is set.' } },
            },
        });
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'will fail' } });
        fireEvent.keyDown(input, { key: 'Enter' });

        // The backend's own sentence wins over the generic note.
        expect(await screen.findByText('No API key is set.')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: /Set API key/i }));
        expect(prompted).toHaveBeenCalledTimes(1);
    });

    it('re-opens the chat once a key is added after a failed bootstrap', async () => {
        startScenarioChat.mockRejectedValueOnce({
            response: {
                status: 503,
                data: { detail: { code: 'OPENAI_KEY_MISSING', message: 'No API key is set.' } },
            },
        });
        setup();
        await screen.findByText('No API key is set.');

        fireEvent.click(screen.getByRole('button', { name: /Set API key/i }));
        expect(prompted).toHaveBeenCalledTimes(1);
        // Saving the key replays exactly what failed — the bootstrap.
        await act(async () => { prompted.mock.calls[0][0].onSaved(); });
        expect(await screen.findByText('Hi, describe your scenario')).toBeInTheDocument();
        expect(screen.queryByText('No API key is set.')).toBeNull();
    });
});

describe('Step1Scenario — message rendering', () => {
    it('renders both user and assistant bubbles from restored messages', async () => {
        const run = { steps: { 1: { status: 'in_progress', data: { session_id: 's', messages: [
            { role: 'assistant', content: 'assistant line' },
            { role: 'user', content: 'user line' },
        ] } } } };
        setup(run);
        expect(await screen.findByText('assistant line')).toBeInTheDocument();
        expect(screen.getByText('user line')).toBeInTheDocument();
        expect(startScenarioChat).not.toHaveBeenCalled();
    });
});
