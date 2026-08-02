// Behavior tests for ComputeBadge — the "which GPU am I on" indicator.
//
// It polls getComputeStatus() and labels the active provider for a given
// workload. We mock the api module so nothing hits the network and drive the
// three branches: remote GPU, this machine, and a failed probe (renders nothing).

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

vi.mock('../../../src/api', () => ({
    getComputeStatus: vi.fn(),
}));

import ComputeBadge from '../../../src/components/ComputeBadge/ComputeBadge';
import { getComputeStatus } from '../../../src/api';

describe('ComputeBadge', () => {
    beforeEach(() => vi.clearAllMocks());

    it('labels the remote_worker provider as "Remote GPU" with the prefix and accelerator', async () => {
        getComputeStatus.mockResolvedValue({
            data: { workloads: { training: { provider: 'remote_worker', accelerator: 'cuda', detail: 'RunPod RTX 4090' } } },
        });
        render(<ComputeBadge workload="training" prefix="Training on" />);
        await waitFor(() => expect(screen.getByText('Remote GPU')).toBeInTheDocument());
        expect(screen.getByText('Training on:')).toBeInTheDocument();
        expect(screen.getByText('· CUDA')).toBeInTheDocument();
    });

    it('labels the local provider as "This machine" and shows the CPU accelerator', async () => {
        getComputeStatus.mockResolvedValue({
            data: { workloads: { training: { provider: 'local', accelerator: 'cpu' } } },
        });
        render(<ComputeBadge workload="training" prefix="Training on" />);
        await waitFor(() => expect(screen.getByText('This machine')).toBeInTheDocument());
        // CPU is shown explicitly so a no-GPU machine is clearly marked.
        expect(screen.getByText('· CPU')).toBeInTheDocument();
    });

    it('hides the meaningless REMOTE accelerator tag (remote worker)', async () => {
        getComputeStatus.mockResolvedValue({
            data: { workloads: { training: { provider: 'remote_worker', accelerator: 'remote' } } },
        });
        render(<ComputeBadge workload="training" prefix="Training on" />);
        await waitFor(() => expect(screen.getByText('Remote GPU')).toBeInTheDocument());
        expect(screen.queryByText('· REMOTE')).toBeNull();
    });

    it('reads the workload it is told to (realtime)', async () => {
        getComputeStatus.mockResolvedValue({
            data: { workloads: { training: { provider: 'local' }, realtime: { provider: 'remote_worker' } } },
        });
        render(<ComputeBadge workload="realtime" prefix="Running on" />);
        await waitFor(() => expect(screen.getByText('Remote GPU')).toBeInTheDocument());
    });

    it('renders nothing when the probe fails', async () => {
        getComputeStatus.mockRejectedValue(new Error('boom'));
        const { container } = render(<ComputeBadge workload="training" />);
        await waitFor(() => expect(getComputeStatus).toHaveBeenCalled());
        expect(container.querySelector('span')).toBeNull();
    });
});
