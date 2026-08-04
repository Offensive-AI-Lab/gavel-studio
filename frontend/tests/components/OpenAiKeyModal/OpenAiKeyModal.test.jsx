// Behavior tests for the shared OpenAiKeyModal.
//
// The modal takes no props: it is mounted once globally and opened by
// promptForOpenAiKey(). We mock src/api so nothing hits the network, and drive
// the four paths: closed by default, opens on the prompt, saves (and hands
// control back via onSaved), and reports a key the backend refuses.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';

vi.mock('../../../src/api', () => ({
    saveOpenAiKey: vi.fn(),
}));

import OpenAiKeyModal from '../../../src/components/OpenAiKeyModal/OpenAiKeyModal';
import {
    promptForOpenAiKey,
    subscribeOpenAiKeySaved,
} from '../../../src/components/OpenAiKeyModal/openAiKeyPrompt';
import { saveOpenAiKey } from '../../../src/api';

const openModal = async (options) => {
    await act(async () => { promptForOpenAiKey(options); });
    return waitFor(() => screen.getByLabelText('OpenAI API Key'));
};

describe('OpenAiKeyModal', () => {
    beforeEach(() => {
        saveOpenAiKey.mockReset().mockResolvedValue({ data: { configured: true } });
    });

    it('renders nothing until something asks for the key', () => {
        render(<OpenAiKeyModal />);
        expect(screen.queryByLabelText('OpenAI API Key')).toBeNull();
    });

    it('opens on promptForOpenAiKey with a masked, non-autofilled input', async () => {
        render(<OpenAiKeyModal />);
        const input = await openModal();
        expect(input).toHaveAttribute('type', 'password');
        expect(input).toHaveAttribute('autocomplete', 'off');
        expect(screen.getByText('Add your OpenAI key')).toBeInTheDocument();
    });

    it('saves the trimmed key, calls onSaved and closes', async () => {
        const onSaved = vi.fn();
        render(<OpenAiKeyModal />);
        const input = await openModal({ onSaved });

        fireEvent.change(input, { target: { value: '  sk-test-key  ' } });
        await act(async () => { fireEvent.click(screen.getByText('Save key')); });

        expect(saveOpenAiKey).toHaveBeenCalledWith('sk-test-key');
        expect(onSaved).toHaveBeenCalledTimes(1);
        await waitFor(() => expect(screen.queryByLabelText('OpenAI API Key')).toBeNull());
    });

    it('asks for a key instead of calling the backend when the field is empty', async () => {
        render(<OpenAiKeyModal />);
        await openModal();

        await act(async () => { fireEvent.click(screen.getByText('Save key')); });

        expect(saveOpenAiKey).not.toHaveBeenCalled();
        expect(screen.getByRole('alert')).toHaveTextContent('Paste your key to continue.');
    });

    it('shows the backend reason when the key is refused and stays open', async () => {
        saveOpenAiKey.mockRejectedValue({ response: { status: 400, data: { detail: 'That key was rejected by OpenAI.' } } });
        const onSaved = vi.fn();
        render(<OpenAiKeyModal />);
        const input = await openModal({ onSaved });

        fireEvent.change(input, { target: { value: 'sk-bad' } });
        await act(async () => { fireEvent.click(screen.getByText('Save key')); });

        expect(screen.getByRole('alert')).toHaveTextContent('That key was rejected by OpenAI.');
        expect(onSaved).not.toHaveBeenCalled();
        expect(screen.getByLabelText('OpenAI API Key')).toBeInTheDocument();
    });

    it('broadcasts the save so upfront "no key yet" notes can clear themselves', async () => {
        const saved = vi.fn();
        const unsubscribe = subscribeOpenAiKeySaved(saved);
        render(<OpenAiKeyModal />);
        const input = await openModal();

        fireEvent.change(input, { target: { value: 'sk-test-key' } });
        await act(async () => { fireEvent.click(screen.getByText('Save key')); });

        expect(saved).toHaveBeenCalledTimes(1);
        unsubscribe();
    });

    it('does not broadcast when the backend refuses the key', async () => {
        saveOpenAiKey.mockRejectedValue({ response: { status: 400, data: { detail: 'nope' } } });
        const saved = vi.fn();
        const unsubscribe = subscribeOpenAiKeySaved(saved);
        render(<OpenAiKeyModal />);
        const input = await openModal();

        fireEvent.change(input, { target: { value: 'sk-bad' } });
        await act(async () => { fireEvent.click(screen.getByText('Save key')); });

        expect(saved).not.toHaveBeenCalled();
        unsubscribe();
    });

    it('keeps what the user typed when a second failure prompts again', async () => {
        render(<OpenAiKeyModal />);
        const input = await openModal();
        fireEvent.change(input, { target: { value: 'sk-half-typed' } });

        await act(async () => { promptForOpenAiKey(); });

        expect(screen.getByLabelText('OpenAI API Key')).toHaveValue('sk-half-typed');
    });
});
