// Tests for Step1CEChat — the conversational CE step. It drives generateCe
// with the running concept + clarification history, renders replies as chat
// bubbles, shows the proposed CE for review, and "Approve & Build" hands off
// to the wizard's onAdvance.
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const { generateCe } = vi.hoisted(() => ({ generateCe: vi.fn() }));
vi.mock('../../../src/api', () => ({ generateCe }));

import Step1CEChat from '../../../src/pages/CEWizardSteps/Step1CEChat';

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

beforeEach(() => { vi.clearAllMocks(); });

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

    it('shows the refusal reason when the concept is already covered', async () => {
        generateCe.mockResolvedValueOnce({ data: { refuse: true, refuse_reason: 'Already covered by an existing CE.' } });
        renderChat();
        typeAndSend('duplicate concept');
        expect(await screen.findByText('Already covered by an existing CE.')).toBeInTheDocument();
        // No proposal → input stays available.
        expect(screen.getByPlaceholderText(/Answer the question|Describe the concept/i)).toBeInTheDocument();
    });
});
