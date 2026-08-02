// TutorialContext — open/dismiss handle PLUS per-page content registry
// for the tutorial modal.
//
// Two display modes:
//
//   Welcome mode  — the 5-slide overview that auto-fires on a new
//                   user's first /workspace mount. No page content
//                   registered.
//
//   Page mode     — context-aware help for whichever page the user is
//                   on, with content tailored to the page's current
//                   state (e.g., RulesManager shows different sections
//                   when a guardrail has 0 rules vs trained vs not).
//
// Pages register their content via `useTutorialContent(content)`. The
// hook stores the latest content in context and clears it on unmount,
// so navigating away resets the modal back to its welcome-mode default.
//
// `mode` is forced to 'welcome' by Workspace's auto-fire path; the
// sidebar's Tutorial button passes nothing (default 'auto'), which
// renders page mode if any page is registered, else welcome.

import { createContext, useContext, useEffect, useState } from 'react';

const TutorialContext = createContext({
    open: false,
    mode: 'auto',         // 'welcome' | 'page' | 'auto'
    pageContent: null,
    show: () => {},
    showWelcome: () => {},
    dismiss: () => {},
    setPageContent: () => {},
});

export const TutorialProvider = ({ children }) => {
    const [open, setOpen] = useState(false);
    const [mode, setMode] = useState('auto');
    const [pageContent, setPageContent] = useState(null);
    const value = {
        open,
        mode,
        pageContent,
        // `show` defaults to auto — page content renders if registered,
        // otherwise the welcome slides. Sidebar Tutorial button uses this.
        show: () => { setMode('auto'); setOpen(true); },
        // `showWelcome` forces the 5-slide overview regardless of any
        // registered page content. Used by the first-login auto-fire.
        showWelcome: () => { setMode('welcome'); setOpen(true); },
        dismiss: () => setOpen(false),
        setPageContent,
    };
    return <TutorialContext.Provider value={value}>{children}</TutorialContext.Provider>;
};

export const useTutorial = () => useContext(TutorialContext);

// Hook each page calls to register its tutorial content. The page
// passes a fresh content object on every render reflecting current
// state; the [serialized] effect dep keeps spurious updates out
// (re-renders with structurally-identical content don't re-fire the
// effect, so no context update on every keystroke).
//
// `content` shape:
//   {
//     title: string                — page name shown at the top of the modal
//     summary: string              — 1-2 sentence "what this page does"
//     sections: [{ heading, bullets: string[] }]   — state-derived
//   }
//
// Cleanup on unmount clears the registration so navigating to a page
// without useTutorialContent falls back to welcome mode.
//
// Important: do NOT add a ref-based guard around `setPageContent(content)`.
// React 18 StrictMode (Vite dev default) double-fires every mount as
// effect → cleanup → effect. A guard that updates a "last serialized"
// ref inside the first run will short-circuit the second run, but the
// cleanup between them already wiped pageContent — leaving the page
// registered as null. The [serialized] dep already prevents re-runs on
// no-op renders, so an extra ref check is both redundant and load-
// bearing for the bug.
export const useTutorialContent = (content) => {
    const { setPageContent } = useTutorial();
    const serialized = content ? JSON.stringify(content) : null;
    useEffect(() => {
        setPageContent(content);
        return () => {
            // Only clear if THIS page was the registrant. Without this
            // check, fast navigations could clear a content registered
            // by the new page mid-mount.
            setPageContent((prev) => (prev === content ? null : prev));
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [serialized]);
};

export default TutorialContext;
