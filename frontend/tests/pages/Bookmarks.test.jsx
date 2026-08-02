// Behavior tests for the unified "My Bookmarks" page (Bookmarks.jsx).
//
// Bookmarks merges the former "My Rule Bookmarks" and "My CE Bookmarks" pages
// into one Layout with internal Rules / CEs tabs (mirroring the Drafts page).
// It renders <BookmarksRules embedded /> or <BookmarksCEs embedded /> based on
// the active tab, initialized from the URL. Here we mock the child pages so we
// can assert the shell (header, tabs, tab switching + URL navigation) without
// pulling in their full network surface.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { render, screen, fireEvent } from '@testing-library/react';

// --- Router: spy navigate, keep useLocation real (driven by initialEntries). ---
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// --- Stub the child pages so we only test the Bookmarks shell. They assert
// they were rendered in embedded mode (no own Layout/header). ---
vi.mock('../../src/pages/BookmarksRules', () => ({
    default: ({ embedded }) => <div data-testid="rules-body" data-embedded={String(!!embedded)} />,
}));
vi.mock('../../src/pages/BookmarksCEs', () => ({
    default: ({ embedded }) => <div data-testid="ces-body" data-embedded={String(!!embedded)} />,
}));

// --- Create-entry-point modals are stubbed (CreateActions renders them). ---
vi.mock('../../src/pages/RuleGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="rule-modal-open" /> : null),
}));
vi.mock('../../src/pages/BuildRuleFromCEsModal', () => ({
    default: ({ open }) => (open ? <div data-testid="build-from-ces-modal-open" /> : null),
}));
vi.mock('../../src/pages/CEGenerationModal', () => ({
    default: ({ open }) => (open ? <div data-testid="ce-modal-open" /> : null),
}));

// --- Stub Sidebar; Layout renders it and its fetches are irrelevant. ---
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    return { default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()) } };
});

vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));

import Bookmarks from '../../src/pages/Bookmarks';

const setUser = () => {
    sessionStorage.setItem('token', 'fake-token');
    sessionStorage.setItem('user', JSON.stringify({ user_id: 7, email: 'a@b.c' }));
};

const renderBookmarks = (path = '/bookmarks') =>
    render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/bookmarks" element={<Bookmarks />} />
                <Route path="/bookmarks/rules" element={<Bookmarks />} />
                <Route path="/bookmarks/ces" element={<Bookmarks />} />
            </Routes>
        </MemoryRouter>,
    );

beforeEach(() => {
    vi.clearAllMocks();
    setUser();
});

describe('Bookmarks — shell', () => {
    it('renders the header, intro copy and both tab buttons', () => {
        renderBookmarks();
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Your Library' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Rules' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'CEs' })).toBeInTheDocument();
        expect(screen.getByText('Hub')).toBeInTheDocument();
    });

    it('Hub breadcrumb navigates to /workspace', () => {
        renderBookmarks();
        fireEvent.click(screen.getByText('Hub'));
        expect(mockNavigate).toHaveBeenCalledWith('/workspace');
    });
});

describe('Bookmarks — tab initialization from URL', () => {
    it('defaults to the Rules tab on /bookmarks', () => {
        renderBookmarks('/bookmarks');
        const body = screen.getByTestId('rules-body');
        expect(body).toBeInTheDocument();
        // Child is rendered embedded (no own Layout/header).
        expect(body).toHaveAttribute('data-embedded', 'true');
        expect(screen.queryByTestId('ces-body')).not.toBeInTheDocument();
    });

    it('opens the CEs tab when the path ends with /ces', () => {
        renderBookmarks('/bookmarks/ces');
        expect(screen.getByTestId('ces-body')).toBeInTheDocument();
        expect(screen.queryByTestId('rules-body')).not.toBeInTheDocument();
    });
});

describe('Bookmarks — tab switching', () => {
    it('switches to the CEs body and navigates the URL', () => {
        renderBookmarks('/bookmarks');
        expect(screen.getByTestId('rules-body')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'CEs' }));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/ces');
        expect(screen.getByTestId('ces-body')).toBeInTheDocument();
        expect(screen.queryByTestId('rules-body')).not.toBeInTheDocument();
    });

    it('switches back to the Rules body and navigates the URL', () => {
        renderBookmarks('/bookmarks/ces');
        expect(screen.getByTestId('ces-body')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Rules' }));
        expect(mockNavigate).toHaveBeenCalledWith('/bookmarks/rules');
        expect(screen.getByTestId('rules-body')).toBeInTheDocument();
        expect(screen.queryByTestId('ces-body')).not.toBeInTheDocument();
    });
});
