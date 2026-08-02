// Behavior tests for Tutorial — the first-login onboarding modal.
//
// Two render modes driven by useTutorial():
//   * welcome slides: a 5-slide overview with Back/Next/skip/Got it and
//     a progress dot strip. The final slide swaps in two CTA buttons.
//   * page mode: when a page has registered pageContent (and mode isn't
//     'welcome'), it renders that single help card with sections + CTAs.
//
// On finish/skip/CTA the component PUTs markTutorialSeen and updates the
// cached localStorage user blob, then calls dismiss() (and maybe navigate).
// We mock useTutorial (to drive open/mode/pageContent + capture dismiss),
// useNavigate (to assert routing), and the api (markTutorialSeen).

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => mockNavigate };
});

// useTutorial is the single source of truth for what Tutorial renders.
const mockUseTutorial = vi.fn();
vi.mock('../../../src/contexts/TutorialContext', () => ({
    useTutorial: () => mockUseTutorial(),
}));

const markTutorialSeen = vi.fn(() => Promise.resolve({ data: {} }));
vi.mock('../../../src/api', () => ({
    markTutorialSeen: (...a) => markTutorialSeen(...a),
}));

import Tutorial from '../../../src/components/Tutorial/Tutorial';

const mockDismiss = vi.fn();

// Default: welcome slides open in auto mode, no registered page content.
const setTutorial = (over = {}) => {
    mockUseTutorial.mockReturnValue({
        open: true,
        mode: 'auto',
        pageContent: null,
        dismiss: mockDismiss,
        ...over,
    });
};

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    markTutorialSeen.mockResolvedValue({ data: {} });
    setTutorial();
});

describe('Tutorial — closed', () => {
    it('renders nothing when not open', () => {
        setTutorial({ open: true }); // overwritten below
        mockUseTutorial.mockReturnValue({ open: false, mode: 'auto', pageContent: null, dismiss: mockDismiss });
        const { container } = render(<Tutorial />);
        expect(container.firstChild).toBeNull();
    });
});

describe('Tutorial — welcome slides', () => {
    it('opens on the first slide with its kicker and title', () => {
        render(<Tutorial />);
        expect(screen.getByText('Welcome')).toBeInTheDocument();
        expect(screen.getByText(/Build AI safety rule sets/)).toBeInTheDocument();
        // Progress strip reports slide 1 of 5.
        expect(screen.getByLabelText('Slide 1 of 5')).toBeInTheDocument();
    });

    it('hides Back on the first slide and shows Next + skip', () => {
        render(<Tutorial />);
        expect(screen.queryByRole('button', { name: 'Back' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Next' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /I'll explore on my own/ })).toBeInTheDocument();
    });

    it('advances through slides with Next and exposes Back after the first', () => {
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        expect(screen.getByText('The vocabulary')).toBeInTheDocument();
        expect(screen.getByLabelText('Slide 2 of 5')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Back' })).toBeInTheDocument();
    });

    it('Back returns to the previous slide', () => {
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        fireEvent.click(screen.getByRole('button', { name: 'Back' }));
        expect(screen.getByText('Welcome')).toBeInTheDocument();
        expect(screen.getByLabelText('Slide 1 of 5')).toBeInTheDocument();
    });

    it('reaches the final slide where Next becomes Got it and skip disappears', () => {
        render(<Tutorial />);
        const next = () => fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        next(); next(); next(); next();
        expect(screen.getByLabelText('Slide 5 of 5')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Got it' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /I'll explore on my own/ })).not.toBeInTheDocument();
    });

    it('shows the two CTA buttons only on the final slide', () => {
        render(<Tutorial />);
        expect(screen.queryByRole('button', { name: /Browse the library/ })).not.toBeInTheDocument();
        const next = () => fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        next(); next(); next(); next();
        expect(screen.getByRole('button', { name: /Browse the library/ })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Open my workspace/ })).toBeInTheDocument();
    });
});

describe('Tutorial — finish / skip persistence', () => {
    it('skip ("explore on my own") marks seen, updates localStorage and dismisses', async () => {
        sessionStorage.setItem('user', JSON.stringify({ user_id: 1, tutorial_seen: false }));
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: /I'll explore on my own/ }));
        await waitFor(() => expect(markTutorialSeen).toHaveBeenCalledTimes(1));
        expect(mockDismiss).toHaveBeenCalledTimes(1);
        const stored = JSON.parse(sessionStorage.getItem('user'));
        expect(stored.tutorial_seen).toBe(true);
    });

    it('Got it on the final slide finishes without navigating', async () => {
        render(<Tutorial />);
        const next = () => fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        next(); next(); next(); next();
        fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
        await waitFor(() => expect(markTutorialSeen).toHaveBeenCalled());
        expect(mockDismiss).toHaveBeenCalledTimes(1);
        expect(mockNavigate).not.toHaveBeenCalled();
    });

    it('CTA "Browse the library" finishes and navigates to /browse', async () => {
        render(<Tutorial />);
        const next = () => fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        next(); next(); next(); next();
        fireEvent.click(screen.getByRole('button', { name: /Browse the library/ }));
        await waitFor(() => expect(markTutorialSeen).toHaveBeenCalled());
        expect(mockNavigate).toHaveBeenCalledWith('/browse');
        expect(mockDismiss).toHaveBeenCalledTimes(1);
    });

    it('CTA "Open my workspace" navigates to /workspace', async () => {
        render(<Tutorial />);
        const next = () => fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        next(); next(); next(); next();
        fireEvent.click(screen.getByRole('button', { name: /Open my workspace/ }));
        await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/workspace'));
    });

    it('still dismisses even when markTutorialSeen rejects (best-effort)', async () => {
        markTutorialSeen.mockRejectedValue(new Error('network'));
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: /I'll explore on my own/ }));
        await waitFor(() => expect(mockDismiss).toHaveBeenCalledTimes(1));
    });

    it('tolerates corrupted localStorage user blob on finish', async () => {
        sessionStorage.setItem('user', '{not-json');
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: /I'll explore on my own/ }));
        await waitFor(() => expect(mockDismiss).toHaveBeenCalledTimes(1));
    });
});

describe('Tutorial — backdrop dismiss', () => {
    it('clicking the backdrop dismisses WITHOUT marking seen', () => {
        const { container } = render(<Tutorial />);
        const backdrop = container.querySelector('.tutorial-backdrop');
        fireEvent.click(backdrop);
        expect(mockDismiss).toHaveBeenCalledTimes(1);
        expect(markTutorialSeen).not.toHaveBeenCalled();
    });

    it('clicking inside the card does NOT dismiss', () => {
        const { container } = render(<Tutorial />);
        fireEvent.click(container.querySelector('.tutorial-card'));
        expect(mockDismiss).not.toHaveBeenCalled();
    });
});

describe('Tutorial — page mode', () => {
    const pageContent = {
        title: 'About this page',
        summary: 'What you can do here.',
        sections: [
            { heading: 'Now', bullets: ['Do A', 'Do B'] },
        ],
        ctas: [
            { label: 'Go to Drafts', to: '/drafts', primary: true },
        ],
    };

    it('renders registered page content instead of the welcome slides', () => {
        setTutorial({ mode: 'auto', pageContent });
        render(<Tutorial />);
        expect(screen.getByText('About this page')).toBeInTheDocument();
        expect(screen.getByText('What you can do here.')).toBeInTheDocument();
        expect(screen.getByText('Do A')).toBeInTheDocument();
        expect(screen.getByText('Do B')).toBeInTheDocument();
        // Page mode has a single "Got it" action, no slide navigation.
        expect(screen.queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Got it' })).toBeInTheDocument();
    });

    it('mode="welcome" forces the slides even when page content exists', () => {
        setTutorial({ mode: 'welcome', pageContent });
        render(<Tutorial />);
        expect(screen.getByText('Welcome')).toBeInTheDocument();
        expect(screen.queryByText('About this page')).not.toBeInTheDocument();
    });

    it('page CTA closes the modal and navigates', () => {
        setTutorial({ mode: 'auto', pageContent });
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: /Go to Drafts/ }));
        expect(mockDismiss).toHaveBeenCalledTimes(1);
        expect(mockNavigate).toHaveBeenCalledWith('/drafts');
    });

    it('page-mode "Got it" dismisses', () => {
        setTutorial({ mode: 'auto', pageContent });
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: 'Got it' }));
        expect(mockDismiss).toHaveBeenCalledTimes(1);
    });

    it('runs a CTA onClick handler when provided', () => {
        const onClick = vi.fn();
        setTutorial({
            mode: 'auto',
            pageContent: { ...pageContent, ctas: [{ label: 'Run it', onClick }] },
        });
        render(<Tutorial />);
        fireEvent.click(screen.getByRole('button', { name: /Run it/ }));
        expect(onClick).toHaveBeenCalledTimes(1);
        expect(mockDismiss).toHaveBeenCalledTimes(1);
    });

    it('tolerates page content with no sections/ctas', () => {
        setTutorial({ mode: 'auto', pageContent: { title: 'Bare', summary: '' } });
        render(<Tutorial />);
        expect(screen.getByText('Bare')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Got it' })).toBeInTheDocument();
    });
});
