// Tests for WizardModal — the in-page modal host for the generation wizards.
// It owns the run lifecycle (bootstrap on open, patchStep, advance) and renders
// the active step inside a GlassModal. We stub GlassModal (render children when
// open) and use tiny step stubs that expose the shell callbacks as buttons.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const api = vi.hoisted(() => ({ getPipelineRun: vi.fn(), updatePipelineStep: vi.fn() }));
vi.mock('../../src/api', () => api);

// The discard guard prompts through showConfirmDialog; mock it so tests can
// choose confirm/cancel without a real Swal popup.
const confirm = vi.hoisted(() => ({ showConfirmDialog: vi.fn() }));
vi.mock('../../src/components/ConfirmDialog/confirmDialog', () => ({
    showConfirmDialog: confirm.showConfirmDialog,
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

import WizardModal from '../../src/pages/WizardModal';

const STEPS = [
    { key: '1', short: '1', title: 'One', hint: '' },
    { key: '2A', short: '2A', title: 'Two', hint: '' },
];
function Step1({ onAdvance }) {
    return <div data-testid="s1"><button onClick={() => onAdvance()}>advance-1</button></div>;
}
function Step2({ onAdvance }) {
    return <div data-testid="s2"><button onClick={() => onAdvance()}>approve</button></div>;
}
const STEP_COMPONENTS = { '1': Step1, '2A': Step2 };

const renderModal = (props = {}) =>
    render(
        <WizardModal
            open
            onClose={props.onClose || vi.fn()}
            title="Test Wizard"
            steps={STEPS}
            stepComponents={STEP_COMPONENTS}
            bootstrap={props.bootstrap || (() => Promise.resolve({ run_id: 5, current_step: '1', steps: {} }))}
            onFinish={props.onFinish || vi.fn()}
            {...props}
        />,
    );

beforeEach(() => {
    vi.clearAllMocks();
    // Each transition returns a run advanced to `advanceTo`.
    api.updatePipelineStep.mockImplementation((id, { advanceTo, stepId }) =>
        Promise.resolve({ data: { run_id: id, current_step: advanceTo || stepId || '1', steps: {} } }),
    );
    confirm.showConfirmDialog.mockResolvedValue(false);
});

// A step-1 run where the user already sent a chat message → dirty.
const DIRTY_RUN = {
    run_id: 6,
    current_step: '1',
    steps: { 1: { data: { messages: [{ role: 'assistant', content: 'hi' }, { role: 'user', content: 'catch phishing' }] } } },
};

// Flush the pending microtask chain of the async close handler.
const flush = () => new Promise((r) => setTimeout(r, 0));

describe('WizardModal', () => {
    it('renders nothing when closed', () => {
        const { container } = render(
            <WizardModal open={false} onClose={vi.fn()} title="x" steps={STEPS}
                stepComponents={STEP_COMPONENTS} bootstrap={vi.fn()} onFinish={vi.fn()} />,
        );
        expect(container.querySelector('[data-testid="glass"]')).toBeNull();
    });

    it('bootstraps on open and renders the active step', async () => {
        renderModal();
        expect(await screen.findByTestId('s1')).toBeInTheDocument();
        expect(screen.getByTestId('glass-title')).toHaveTextContent('Test Wizard');
    });

    it('advances from step 1 to step 2A', async () => {
        renderModal();
        await screen.findByTestId('s1');
        fireEvent.click(screen.getByText('advance-1'));
        expect(await screen.findByTestId('s2')).toBeInTheDocument();
        expect(api.updatePipelineStep).toHaveBeenCalled();
    });

    it('approving on the last step calls onFinish then closes', async () => {
        const onFinish = vi.fn(() => Promise.resolve());
        const onClose = vi.fn();
        renderModal({
            onFinish, onClose,
            bootstrap: () => Promise.resolve({ run_id: 9, current_step: '2A', steps: {} }),
        });
        await screen.findByTestId('s2');
        fireEvent.click(screen.getByText('approve'));
        await waitFor(() => expect(onFinish).toHaveBeenCalledWith(
            expect.objectContaining({ run_id: 9 }),
        ));
        await waitFor(() => expect(onClose).toHaveBeenCalled());
    });

    it('surfaces a bootstrap error', async () => {
        renderModal({ bootstrap: () => Promise.reject({ response: { data: { detail: 'boom' } } }) });
        expect(await screen.findByText('boom')).toBeInTheDocument();
    });

    it('abandons the run when closed WITHOUT approving (reset)', async () => {
        const onAbandon = vi.fn();
        const props = {
            title: 't', steps: STEPS, stepComponents: STEP_COMPONENTS,
            bootstrap: () => Promise.resolve({ run_id: 7, current_step: '1', steps: {} }),
            onFinish: vi.fn(), onAbandon,
        };
        const { rerender } = render(<WizardModal open onClose={vi.fn()} {...props} />);
        await screen.findByTestId('s1');
        // User closes the modal.
        rerender(<WizardModal open={false} onClose={vi.fn()} {...props} />);
        await waitFor(() => expect(onAbandon).toHaveBeenCalledWith(
            expect.objectContaining({ run_id: 7 }),
        ));
    });

    it('does NOT abandon after the user approves (background build owns the run)', async () => {
        const onAbandon = vi.fn();
        const onFinish = vi.fn(() => Promise.resolve());
        const props = {
            title: 't', steps: STEPS, stepComponents: STEP_COMPONENTS,
            bootstrap: () => Promise.resolve({ run_id: 8, current_step: '2A', steps: {} }),
            onFinish, onAbandon,
        };
        const onClose = vi.fn();
        const { rerender } = render(<WizardModal open onClose={onClose} {...props} />);
        await screen.findByTestId('s2');
        fireEvent.click(screen.getByText('approve'));
        await waitFor(() => expect(onFinish).toHaveBeenCalled());
        // Parent closes the modal in response.
        rerender(<WizardModal open={false} onClose={onClose} {...props} />);
        await waitFor(() => expect(onClose).toHaveBeenCalled());
        expect(onAbandon).not.toHaveBeenCalled();
    });

    // ---- confirm-before-discard guard (backdrop / X → GlassModal onClose) ----

    describe('discard guard', () => {
        it('closes a pristine just-bootstrapped run WITHOUT prompting', async () => {
            const onClose = vi.fn();
            // Default bootstrap: step 1, no messages — nothing to lose.
            renderModal({ onClose });
            await screen.findByTestId('s1');
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(onClose).toHaveBeenCalled());
            expect(confirm.showConfirmDialog).not.toHaveBeenCalled();
        });

        it('prompts on a dirty run; cancelling keeps it open and does not abandon', async () => {
            const onClose = vi.fn();
            const onAbandon = vi.fn();
            confirm.showConfirmDialog.mockResolvedValue(false);
            renderModal({ onClose, onAbandon, bootstrap: () => Promise.resolve(DIRTY_RUN) });
            await screen.findByTestId('s1');
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1));
            await flush();
            expect(onClose).not.toHaveBeenCalled();
            expect(onAbandon).not.toHaveBeenCalled();
            // The wizard is still up.
            expect(screen.getByTestId('s1')).toBeInTheDocument();
        });

        it('confirming the prompt closes, and the parent close still abandons the run', async () => {
            const onClose = vi.fn();
            const onAbandon = vi.fn();
            confirm.showConfirmDialog.mockResolvedValue(true);
            const props = {
                title: 't', steps: STEPS, stepComponents: STEP_COMPONENTS,
                bootstrap: () => Promise.resolve(DIRTY_RUN),
                onFinish: vi.fn(), onAbandon,
            };
            const { rerender } = render(<WizardModal open onClose={onClose} {...props} />);
            await screen.findByTestId('s1');
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
            expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1);
            // Parent flips open=false in response → abandon semantics unchanged.
            rerender(<WizardModal open={false} onClose={onClose} {...props} />);
            await waitFor(() => expect(onAbandon).toHaveBeenCalledWith(
                expect.objectContaining({ run_id: 6 }),
            ));
        });

        it('prompts when the run progressed past step 1 (even with no chat messages)', async () => {
            const onClose = vi.fn();
            renderModal({
                onClose,
                bootstrap: () => Promise.resolve({ run_id: 11, current_step: '2A', steps: {} }),
            });
            await screen.findByTestId('s2');
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1));
            await flush();
            expect(onClose).not.toHaveBeenCalled();
        });

        it('never prompts after Approve & Build (finished run)', async () => {
            const onClose = vi.fn();
            const onFinish = vi.fn(() => Promise.resolve());
            renderModal({
                onClose, onFinish,
                bootstrap: () => Promise.resolve({ run_id: 12, current_step: '2A', steps: {} }),
            });
            await screen.findByTestId('s2');
            fireEvent.click(screen.getByText('approve'));
            // advance() closes programmatically — no prompt.
            await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
            // And a user close after finishing doesn't prompt either.
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(onClose).toHaveBeenCalledTimes(2));
            expect(confirm.showConfirmDialog).not.toHaveBeenCalled();
        });

        it('a second close click while the confirm is open does not stack a second prompt', async () => {
            const onClose = vi.fn();
            let resolveConfirm;
            confirm.showConfirmDialog.mockImplementation(
                () => new Promise((res) => { resolveConfirm = res; }),
            );
            renderModal({ onClose, bootstrap: () => Promise.resolve(DIRTY_RUN) });
            await screen.findByTestId('s1');
            fireEvent.click(screen.getByText('glass-close'));
            await waitFor(() => expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1));
            fireEvent.click(screen.getByText('glass-close'));
            await flush();
            expect(confirm.showConfirmDialog).toHaveBeenCalledTimes(1);
            resolveConfirm(false);
            await flush();
            expect(onClose).not.toHaveBeenCalled();
        });
    });
});
