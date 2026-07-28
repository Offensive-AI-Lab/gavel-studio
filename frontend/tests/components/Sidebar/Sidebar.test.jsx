// Behavior tests for Sidebar (redesigned).
//
// Sidebar renders the brand header, a library-sync indicator (synced /
// available / pulling), three nav sections — EXPLORE (Community, Create),
// MY WORKSPACE (Your Library, Guardrails), RECENTS (Guardrails / Rules / CEs,
// client-side localStorage) — and a footer with user info + logout. There is
// no longer a Models item or models tree.
//
// We mock ../../api (only syncLibrary is used now), the SyncStatusContext hook
// (to drive the indicator), and spy useNavigate. useLocation is real, driven by
// MemoryRouter so we can assert the active class per route.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent } from '@testing-library/react';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

const empty = (data = {}) => Promise.resolve({ data });
vi.mock('../../../src/api', () => ({
    syncLibrary: vi.fn(() => empty({})),
}));

const syncValue = { status: 'synced', pulling: false, setStatus: vi.fn(), setPulling: vi.fn() };
vi.mock('../../../src/contexts/SyncStatusContext', () => ({
    useSyncStatus: () => syncValue,
}));

import Sidebar from '../../../src/components/Sidebar/Sidebar';
import * as api from '../../../src/api';

const setUser = (user = { user_id: 7, username: 'alice', email: 'a@b.c' }) =>
    sessionStorage.setItem('user', JSON.stringify(user));

const renderSidebar = (path = '/guardrails') =>
    render(
        <MemoryRouter initialEntries={[path]}>
            <Sidebar />
        </MemoryRouter>,
    );

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    setUser();
    syncValue.status = 'synced';
    syncValue.pulling = false;
});

describe('Sidebar — header & nav', () => {
    it('renders the brand and the Explore / Workspace / Recents sections', () => {
        renderSidebar();
        expect(screen.getByText('GAVEL')).toBeInTheDocument();
        expect(screen.getByText('EXPLORE')).toBeInTheDocument();
        expect(screen.getByText('Community')).toBeInTheDocument();
        expect(screen.getByText('Create')).toBeInTheDocument();
        expect(screen.getByText('MY WORKSPACE')).toBeInTheDocument();
        expect(screen.getByText('Your Library')).toBeInTheDocument();
        expect(screen.getByText('RECENTS')).toBeInTheDocument();
        // "Rule Sets" appears twice — the workspace item and the recents group.
        expect(screen.getAllByText('Rule Sets').length).toBeGreaterThanOrEqual(2);
        // No Logout: there is no login.
        expect(screen.queryByText('Logout')).not.toBeInTheDocument();
    });

    it('no longer renders a Models item or tree', () => {
        renderSidebar();
        expect(screen.queryByText('Models')).not.toBeInTheDocument();
        expect(screen.queryByText('All Models')).not.toBeInTheDocument();
    });

    it('navigates when nav items are clicked', () => {
        renderSidebar();
        fireEvent.click(screen.getByText('Community'));
        expect(mockNavigate).toHaveBeenCalledWith('/community');
        fireEvent.click(screen.getByText('Your Library'));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks');
        fireEvent.click(screen.getAllByText('Rule Sets')[0]); // workspace item (first in DOM)
        expect(mockNavigate).toHaveBeenCalledWith('/guardrails');
        fireEvent.click(screen.getByText('GAVEL'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
    });
});

describe('Sidebar — active-route highlighting', () => {
    it('marks Rule Sets active on /guardrails', () => {
        renderSidebar('/guardrails');
        expect(screen.getAllByText('Rule Sets')[0].closest('.nav-item')).toHaveClass('active');
    });

    it('marks Your Library active on /bookmarks', () => {
        renderSidebar('/bookmarks/ces');
        expect(screen.getByText('Your Library').closest('.nav-item')).toHaveClass('active');
    });

    it('marks Community active on /community', () => {
        renderSidebar('/community');
        expect(screen.getByText('Community').closest('.nav-item')).toHaveClass('active');
    });
});

describe('Sidebar — recents (client-side)', () => {
    it('shows recently-opened guardrails (group is open by default) and navigates', () => {
        localStorage.setItem('gavel_recents_guardrail', JSON.stringify([
            { id: 9, name: 'Tox Guard', path: '/classifiers/9/rules' },
        ]));
        renderSidebar();
        fireEvent.click(screen.getByText('Tox Guard'));
        expect(mockNavigate).toHaveBeenCalledWith('/classifiers/9/rules');
    });

    it('reveals a collapsed group only after clicking its header', () => {
        localStorage.setItem('gavel_recents_rule', JSON.stringify([
            { id: 5, name: 'My Rule', path: '/rules/5' },
        ]));
        renderSidebar();
        expect(screen.queryByText('My Rule')).not.toBeInTheDocument(); // Rules collapsed
        fireEvent.click(screen.getByText('Rules'));
        expect(screen.getByText('My Rule')).toBeInTheDocument();
    });
});

describe('Sidebar — sync indicator', () => {
    it('renders the synced state', () => {
        syncValue.status = 'synced';
        renderSidebar();
        expect(screen.getByText('Library synced')).toBeInTheDocument();
    });

    it('renders "Updates available" and pulls on click', () => {
        syncValue.status = 'available';
        renderSidebar();
        fireEvent.click(screen.getByText('Updates available'));
        expect(api.syncLibrary).toHaveBeenCalled();
    });

    it('renders the pulling state', () => {
        syncValue.pulling = true;
        renderSidebar();
        expect(screen.getByText('Updating…')).toBeInTheDocument();
    });
});

describe('Sidebar — footer', () => {
    it('renders the user name + email', () => {
        renderSidebar();
        expect(screen.getByText('alice')).toBeInTheDocument();
        expect(screen.getByText('a@b.c')).toBeInTheDocument();
    });

});
