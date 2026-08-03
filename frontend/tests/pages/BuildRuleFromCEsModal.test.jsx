// Tests for BuildRuleFromCEsModal — the wrapper's confirm-before-discard guard.
//
// Closing the modal unmounts the wizard body, whose cleanup discards the
// provisional rule. The wrapper must therefore confirm BEFORE unmounting when
// the body reports progress (via dirtyRef), close silently when the wizard is
// pristine, and never prompt on the programmatic close after the build was
// committed to the background.
//
// Strategy mirrors BuildRuleFromCEs.test.jsx (mock api + confirmDialog, stub
// RuleDefaultsStep) plus a GlassModal stub whose close button stands in for
// both the backdrop click and the X — they share onClose.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TaskTrayProvider } from '../../src/contexts/TaskTrayContext';

vi.mock('../../src/api', () => ({
    getCEBookmarks: vi.fn(),
    getAllCategories: vi.fn(),
    createDraftRuleFromBookmarks: vi.fn(),
    finalizeRule: vi.fn(),
    discardUnreadyRule: vi.fn(),
}));

const confirm = vi.hoisted(() => ({
    showConfirmDialog: vi.fn(),
    showAlertDialog: vi.fn(() => Promise.resolve()),
}));
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showConfirmDialog: confirm.showConfirmDialog,
    showAlertDialog: confirm.showAlertDialog,
    default: confirm.showConfirmDialog,
}));

vi.mock('../../src/components/GlassModal/GlassModal', () => ({
    default: ({ isOpen, title, children, onClose }) =>
        isOpen ? (
            <div data-testid="glass">
                <span data-testid="glass-title">{title}</span>
                <button onClick={onClose}>glass-close</button>
                {children}
            </div>
        ) : null,
}));

// Stub the final step so step 5 is deterministic (onDone = user kicked off
// the background build → handleStarted).
vi.mock('../../src/components/RuleDefaults/RuleDefaultsStep', () => ({
    default: ({ onDone }) => (
        <div data-testid="rule-defaults-step">
            <button onClick={onDone}>finish-defaults</button>
        </div>
    ),
}));

import {
    getCEBookmarks,
    getAllCategories,
    createDraftRuleFromBookmarks,
    finalizeRule,
    discardUnreadyRule,
} from '../../src/api';
import BuildRuleFromCEsModal from '../../src/pages/BuildRuleFromCEsModal';

const BOOKMARKS = [
    { ce_id: 1, name: 'Alpha CE', category: 'Safety' },
    { ce_id: 2, name: 'Beta CE', category: 'Privacy' },
];

let onCloseMock = vi.fn();
const tree = (open) => (
    <TaskTrayProvider>
        <BuildRuleFromCEsModal open={open} onClose={onCloseMock} />
    </TaskTrayProvider>
);
const renderModal = () => render(tree(true));

const waitLoaded = async () =>
    waitFor(() => expect(screen.queryByText(/Loading your bookmarked CEs/i)).not.toBeInTheDocument());

const toggleCe = (name) => {
    const label = screen.getByText(name).closest('label');
    fireEvent.click(within(label).getByRole('checkbox'));
};

// Flush the pending microtask chain of the async close handler.
const flush = () => new Promise((r) => setTimeout(r, 0));

describe('BuildRuleFromCEsModal discard guard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        onCloseMock = vi.fn();
        sessionStorage.setItem('token', 'fake-token');
        sessionStorage.setItem('user', JSON.stringify({ user_id: 42, email: 'a@b.c' }));
        getCEBookmarks.mockResolvedValue({ data: { bookmarks: BOOKMARKS } });
        getAllCategories.mockResolvedValue({ data: ['Safety'] });
        createDraftRuleFromBookmarks.mockResolvedValue({ data: { rule_id: 99 } });
        finalizeRule.mockResolvedValue({ data: {} });
        discardUnreadyRule.mockResolvedValue({ data: {} });
        confirm.showConfirmDialog.mockResolvedValue(false);
    });

    it('closes silently when the wizard is pristine (nothing entered yet)', async () => {
        renderModal();
        await waitLoaded();
        fireEvent.click(screen.getByText('glass-close'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalledTimes(1));
        expect(confirm.showConfirmDialog).not.toHaveBeenCalled();
    });

    it('prompts once dirty; cancel keeps the wizard mounted, confirm closes', async () => {
        renderModal();
        await waitLoaded();
        toggleCe('Alpha CE'); // picking a CE makes the wizard dirty

        fireEvent.click(screen.getByText('glass-close'));
        await waitFor(() => expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1));
        await flush();
        // Cancelled → still open, nothing discarded.
        expect(onCloseMock).not.toHaveBeenCalled();
        expect(screen.getByText('Alpha CE')).toBeInTheDocument();
        expect(discardUnreadyRule).not.toHaveBeenCalled();

        confirm.showConfirmDialog.mockResolvedValue(true);
        fireEvent.click(screen.getByText('glass-close'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalledTimes(1));
    });

    it('confirmed close of a provisional rule lets the unmount discard it (guard prevents, never undoes)', async () => {
        confirm.showConfirmDialog.mockResolvedValue(true);
        const { rerender } = renderModal();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // -> 2
        fireEvent.click(screen.getByText('Next')); // -> 3
        fireEvent.click(screen.getByText('Next')); // -> 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        fireEvent.click(screen.getByText('glass-close'));
        await waitFor(() => expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalledTimes(1));
        // The guard itself discards nothing — the unmount (parent flipping
        // open=false) does, exactly as before.
        expect(discardUnreadyRule).not.toHaveBeenCalled();
        rerender(tree(false));
        await waitFor(() => expect(discardUnreadyRule).toHaveBeenCalledWith(99));
    });

    it('does NOT prompt on the programmatic close after the build is committed', async () => {
        renderModal();
        await waitLoaded();
        toggleCe('Alpha CE');
        toggleCe('Beta CE');
        fireEvent.click(screen.getByText('Next')); // -> 2
        fireEvent.click(screen.getByText('Next')); // -> 3
        fireEvent.click(screen.getByText('Next')); // -> 4
        fireEvent.change(screen.getByPlaceholderText(/phishing_content_creation/i), { target: { value: 'r' } });
        fireEvent.click(screen.getByRole('button', { name: 'Safety' }));
        fireEvent.click(screen.getByText('Create Rule'));
        await waitFor(() => expect(screen.getByTestId('rule-defaults-step')).toBeInTheDocument());

        // Kicking off the background build closes via the same guarded close —
        // committed, so no prompt and no discard.
        fireEvent.click(screen.getByText('finish-defaults'));
        await waitFor(() => expect(onCloseMock).toHaveBeenCalledTimes(1));
        expect(confirm.showConfirmDialog).not.toHaveBeenCalled();
    });
});
