// Behavior tests for Step1Scenario — the scenario-ideation chat step of the
// rule wizard. It's a self-contained component: props `run` + `onPatchStep`,
// and the only network surface is startScenarioChat / sendScenarioChatMessage
// from ../../api. We mock that module so nothing hits the network.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// --- Mock ../../api. Only the two scenario chat exports are used by this
// file, but we keep the mock minimal + benign.
const startScenarioChat = vi.fn();
const sendScenarioChatMessage = vi.fn();
vi.mock('../../../src/api', () => ({
    startScenarioChat: (...a) => startScenarioChat(...a),
    sendScenarioChatMessage: (...a) => sendScenarioChatMessage(...a),
}));

import Step1Scenario from '../../../src/pages/RuleWizardSteps/Step1Scenario';

// Default benign responses; individual tests override as needed.
beforeEach(() => {
    vi.clearAllMocks();
    startScenarioChat.mockResolvedValue({ data: { session_id: 'sess-1', message: 'Hi, describe your scenario' } });
    sendScenarioChatMessage.mockResolvedValue({ data: { message: 'Tell me more', is_final: false } });
});

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

    it('logs an error and does not crash when startScenarioChat rejects', async () => {
        const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        startScenarioChat.mockRejectedValueOnce(new Error('boom'));
        setup();
        await waitFor(() => expect(errSpy).toHaveBeenCalledWith('Start chat failed:', expect.any(Error)));
        // Header still present.
        expect(screen.getByText('Scenario Chat')).toBeInTheDocument();
        errSpy.mockRestore();
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

    it('logs an error and re-enables sending when sendScenarioChatMessage rejects', async () => {
        const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        sendScenarioChatMessage.mockRejectedValueOnce(new Error('net'));
        setup();
        await screen.findByText('Hi, describe your scenario');
        const input = screen.getByPlaceholderText(/Describe the misuse/i);
        fireEvent.change(input, { target: { value: 'will fail' } });
        fireEvent.keyDown(input, { key: 'Enter' });
        await waitFor(() => expect(errSpy).toHaveBeenCalledWith('Chat send failed:', expect.any(Error)));
        // user bubble still shown
        expect(screen.getByText('will fail')).toBeInTheDocument();
        // input re-enabled (sending reset to false)
        await waitFor(() => expect(input).not.toBeDisabled());
        errSpy.mockRestore();
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
