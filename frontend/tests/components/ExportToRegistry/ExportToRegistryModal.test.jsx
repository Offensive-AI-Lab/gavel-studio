// Behavior tests for the gavel-rules export modal.
//
// The modal turns a local rule / CE / rule set into the registry's own files.
// What matters here: the right fields are asked for per artifact kind, the
// preflight problems reach the user BEFORE they download (errors block the
// download outright, warnings are advisory), every listed file is actually
// downloaded, the GitHub handle is remembered, and the "open a pull request"
// link only pre-fills GitHub when the payload is a single small file.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../../src/api', () => ({
    getExportSettings: vi.fn(),
    saveExportSettings: vi.fn(),
    getRegistryExportPreflight: vi.fn(),
    downloadRegistryExportFile: vi.fn(),
}));

import * as api from '../../../src/api';
import ExportToRegistryModal from '../../../src/components/ExportToRegistry/ExportToRegistryModal';

const rulesetPreflight = (over = {}) => ({
    kind: 'ruleset',
    name: 'scam_detector',
    title: 'Scam Detector',
    directory: 'rulesets',
    members: ['phishing_content_creation'],
    local_only: [],
    files: [{ filename: 'scam_detector.yaml', path: 'rulesets/scam_detector.yaml', kind: 'yaml' }],
    errors: [],
    warnings: [],
    preview: 'name: scam_detector\ntitle: Scam Detector\n',
    ...over,
});

const cePreflight = (over = {}) => ({
    kind: 'ce',
    name: 'trust_seeding',
    title: 'Trust Seeding',
    role: 'llm_behavior',
    roles: ['directive_to_user', 'llm_task', 'llm_behavior', 'topic'],
    directory: 'ces/trust_seeding',
    files: [
        { filename: 'ce.yaml', path: 'ces/trust_seeding/ce.yaml', kind: 'yaml' },
        { filename: 'excitation.json', path: 'ces/trust_seeding/excitation.json', kind: 'json' },
        { filename: 'calibration.json', path: 'ces/trust_seeding/calibration.json', kind: 'json' },
    ],
    errors: [],
    warnings: [],
    preview: 'name: trust_seeding\n',
    ...over,
});

const renderModal = (props = {}) => render(
    <ExportToRegistryModal
        open onClose={vi.fn()} kind="ruleset" entityId={7} defaultTitle="Scam Detector"
        {...props}
    />,
);

// The preflight call is debounced inside the modal.
const waitForPreflight = () =>
    waitFor(() => expect(api.getRegistryExportPreflight).toHaveBeenCalled());

beforeEach(() => {
    vi.clearAllMocks();
    api.getExportSettings.mockResolvedValue({ data: { github_username: 'octocat' } });
    api.saveExportSettings.mockResolvedValue({ data: {} });
    api.getRegistryExportPreflight.mockResolvedValue({ data: rulesetPreflight() });
    api.downloadRegistryExportFile.mockResolvedValue('scam_detector.yaml');
});

describe('ExportToRegistryModal — framing', () => {
    it('says nothing is published from the studio', async () => {
        renderModal();
        // GlassModal mounts on an animation frame, so nothing is in the DOM yet.
        const blurb = await screen.findByText(/you submit the files as a pull request/i);
        expect(blurb.textContent).toMatch(/Nothing is published/i);
        await waitForPreflight();
    });

    it('prefills the remembered GitHub username', async () => {
        renderModal();
        await waitFor(() =>
            expect(screen.getByLabelText(/GitHub username/i)).toHaveValue('octocat'));
    });

    it('still renders when the remembered username cannot be loaded', async () => {
        api.getExportSettings.mockRejectedValue(new Error('offline'));
        renderModal();
        await waitFor(() =>
            expect(screen.getByLabelText(/GitHub username/i)).toHaveValue(''));
    });
});

describe('ExportToRegistryModal — fields per artifact kind', () => {
    it('asks a rule set for a description', async () => {
        renderModal({ kind: 'ruleset' });
        expect(await screen.findByLabelText(/Description/i)).toBeInTheDocument();
        expect(screen.queryByLabelText(/^Role$/i)).toBeNull();
        await waitForPreflight();
    });

    it('asks a CE for a role, offering the registry enum', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({ data: cePreflight() });
        renderModal({ kind: 'ce' });
        await waitForPreflight();
        await waitFor(() => expect(screen.getByLabelText(/Role/i)).toBeInTheDocument());
        expect(screen.queryByLabelText(/Description/i)).toBeNull();
        for (const role of ['directive_to_user', 'llm_task', 'llm_behavior', 'topic']) {
            expect(screen.getByRole('option', { name: role })).toBeInTheDocument();
        }
    });

    it('asks a rule for neither description nor role', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({
            data: { kind: 'rule', name: 'r', directory: 'rules/r', files: [], errors: [], warnings: [], preview: '' },
        });
        renderModal({ kind: 'rule' });
        await waitForPreflight();
        expect(screen.queryByLabelText(/Description/i)).toBeNull();
        expect(screen.queryByLabelText(/^Role$/i)).toBeNull();
    });

    it('passes the edited fields to preflight', async () => {
        renderModal({ kind: 'ruleset' });
        await waitForPreflight();
        fireEvent.change(screen.getByLabelText(/Description/i), {
            target: { value: 'Scam-adjacent rules.' },
        });
        await waitFor(() => {
            const last = api.getRegistryExportPreflight.mock.calls.at(-1);
            expect(last[2]).toMatchObject({ description: 'Scam-adjacent rules.' });
        });
    });
});

describe('ExportToRegistryModal — errors and warnings', () => {
    it('separates blocking errors from advisory warnings', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({
            data: rulesetPreflight({
                errors: ['description is empty — the registry requires one'],
                warnings: ["rule 'my_local' exists only in this studio"],
            }),
        });
        renderModal();
        expect(await screen.findByText(/description is empty/i)).toBeInTheDocument();
        expect(screen.getByText(/registry would reject this/i)).toBeInTheDocument();
        expect(screen.getByText(/exists only in this studio/i)).toBeInTheDocument();
        expect(screen.getByText(/Needs attention/i)).toBeInTheDocument();
    });

    it('blocks the download and the GitHub hand-off while errors exist', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({
            data: rulesetPreflight({
                errors: ['description is empty — the registry requires one'],
            }),
        });
        renderModal();
        await screen.findByText(/description is empty/i);
        expect(screen.getByRole('button', { name: /Download/i })).toBeDisabled();
        expect(screen.queryByRole('link', { name: /GitHub/i })).toBeNull();
    });

    it('does not block the download for warnings alone', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({
            data: rulesetPreflight({
                warnings: ["rule 'my_local' exists only in this studio"],
            }),
        });
        renderModal();
        await screen.findByText(/exists only in this studio/i);
        expect(screen.getByRole('button', { name: /Download/i })).toBeEnabled();
    });

    it('shows no panels when the artifact is clean', async () => {
        renderModal();
        await waitForPreflight();
        await waitFor(() => expect(screen.queryByText(/Needs attention/i)).toBeNull());
        expect(screen.queryByText(/registry would reject this/i)).toBeNull();
    });

    it('reports a non-preflight 200 instead of rendering a dead button', async () => {
        // What a dev-proxy gap looks like: the SPA's index.html comes back 200.
        api.getRegistryExportPreflight.mockResolvedValue({ data: '<!doctype html><html></html>' });
        renderModal();
        expect(await screen.findByText(/did not respond as expected/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Download/i })).toBeDisabled();
        expect(screen.queryByRole('link', { name: /GitHub/i })).toBeNull();
    });

    it('reports a failed preflight instead of offering a broken download', async () => {
        api.getRegistryExportPreflight.mockRejectedValue({
            response: { data: { detail: 'Rule set has no rules to export' } },
        });
        renderModal();
        expect(await screen.findByText(/no rules to export/i)).toBeInTheDocument();
        await waitFor(() =>
            expect(screen.getByRole('button', { name: /Download/i })).toBeDisabled());
    });
});

describe('ExportToRegistryModal — downloading', () => {
    it('lists each file with its path inside the repo', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({ data: cePreflight() });
        renderModal({ kind: 'ce' });
        expect(await screen.findByText('ces/trust_seeding/ce.yaml')).toBeInTheDocument();
        expect(screen.getByText('ces/trust_seeding/excitation.json')).toBeInTheDocument();
        expect(screen.getByText('ces/trust_seeding/calibration.json')).toBeInTheDocument();
    });

    it('downloads every listed file, not just the YAML', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({ data: cePreflight() });
        renderModal({ kind: 'ce' });
        await screen.findByText('ces/trust_seeding/ce.yaml');
        fireEvent.click(screen.getByRole('button', { name: /Download .* files?/i }));
        await waitFor(() =>
            expect(api.downloadRegistryExportFile).toHaveBeenCalledTimes(3));
        const names = api.downloadRegistryExportFile.mock.calls.map((c) => c[2]);
        expect(names).toEqual(['ce.yaml', 'excitation.json', 'calibration.json']);
    });

    it('passes the author through so provenance is filled in', async () => {
        renderModal();
        await waitForPreflight();
        fireEvent.click(screen.getByRole('button', { name: /Download/i }));
        await waitFor(() => expect(api.downloadRegistryExportFile).toHaveBeenCalled());
        expect(api.downloadRegistryExportFile.mock.calls[0][3])
            .toMatchObject({ author: 'octocat' });
    });

    it('remembers the GitHub username after a successful export', async () => {
        renderModal();
        await waitForPreflight();
        fireEvent.change(screen.getByLabelText(/GitHub username/i),
            { target: { value: 'newhandle' } });
        fireEvent.click(screen.getByRole('button', { name: /Download/i }));
        await waitFor(() =>
            expect(api.saveExportSettings).toHaveBeenCalledWith('newhandle'));
    });

    it('reports a download failure instead of claiming success', async () => {
        api.downloadRegistryExportFile.mockRejectedValue(new Error('disk on fire'));
        renderModal();
        await waitForPreflight();
        fireEvent.click(screen.getByRole('button', { name: /Download/i }));
        expect(await screen.findByText(/disk on fire/i)).toBeInTheDocument();
        expect(screen.queryByText(/^Saved\./)).toBeNull();
    });

    it('confirms once the files are saved', async () => {
        renderModal();
        await waitForPreflight();
        fireEvent.click(screen.getByRole('button', { name: /Download/i }));
        expect(await screen.findByText(/Saved\./)).toBeInTheDocument();
    });
});

describe('ExportToRegistryModal — GitHub hand-off', () => {
    it('pre-fills the GitHub editor for a single small rule set file', async () => {
        renderModal();
        const link = await screen.findByRole('link', { name: /Open a pull request/i });
        const href = link.getAttribute('href');
        expect(href).toContain('/Offensive-AI-Lab/gavel-rules/new/main');
        expect(href).toContain(`filename=${encodeURIComponent('rulesets/scam_detector.yaml')}`);
        expect(href).toContain('value=');
    });

    it('falls back to the upload page for a multi-file CE', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({ data: cePreflight() });
        renderModal({ kind: 'ce' });
        const link = await screen.findByRole('link', { name: /Open GitHub to upload/i });
        expect(link).toHaveAttribute(
            'href', 'https://github.com/Offensive-AI-Lab/gavel-rules/upload/main/ces/trust_seeding');
        expect(screen.queryByRole('link', { name: /Open a pull request/i })).toBeNull();
    });

    it('falls back to upload when the YAML is too big for a URL', async () => {
        api.getRegistryExportPreflight.mockResolvedValue({
            data: rulesetPreflight({ preview: 'x'.repeat(7000) }),
        });
        renderModal();
        expect(await screen.findByRole('link', { name: /Open GitHub to upload/i })).toBeInTheDocument();
        expect(screen.queryByRole('link', { name: /Open a pull request/i })).toBeNull();
    });

    it('opens GitHub in a new tab rather than hijacking the studio', async () => {
        renderModal();
        const link = await screen.findByRole('link', { name: /Open a pull request/i });
        expect(link).toHaveAttribute('target', '_blank');
        expect(link).toHaveAttribute('rel', expect.stringContaining('noreferrer'));
    });
});
