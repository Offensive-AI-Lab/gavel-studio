// Tests for WizardModal — the in-page modal host for the generation wizards.
// It owns the run lifecycle (bootstrap on open, patchStep, advance) and renders
// the active step inside a GlassModal. We stub GlassModal (render children when
// open) and use tiny step stubs that expose the shell callbacks as buttons.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const api = vi.hoisted(() => ({ getPipelineRun: vi.fn(), updatePipelineStep: vi.fn() }));
vi.mock('../../src/api', () => api);

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
});

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
});
