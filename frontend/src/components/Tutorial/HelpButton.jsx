// HelpButton — a small floating "?" that opens the page's contextual help.
//
// Mounted once, globally (next to <Tutorial/>). Clicking it calls the
// tutorial context's auto-mode `show()`, which renders the current page's
// content registered via useTutorialContent — or the welcome overview as a
// fallback when a page hasn't registered any. Hidden on the chromeless
// landing/auth routes and while the modal itself is open (so it never
// overlaps the dialog).
import { useLocation } from 'react-router-dom';
import { FiHelpCircle } from 'react-icons/fi';
import { useTutorial } from '../../contexts/TutorialContext';

// '/' now redirects straight into the app, and there are no auth pages, so
// there is nowhere left to hide the help button.
const HIDDEN_ON = new Set();

const HelpButton = () => {
    const { show, open } = useTutorial();
    const { pathname } = useLocation();

    // Also hidden on the realtime monitor — its right-hand panel runs into the
    // bottom-right corner, so the floating button would cover the Rule Triggers.
    if (open || HIDDEN_ON.has(pathname) || pathname.endsWith('/monitor')) return null;

    return (
        <button
            type="button"
            className="tutorial-help-fab"
            onClick={show}
            aria-label="Help — what's on this page"
            title="What's on this page?"
        >
            <FiHelpCircle />
        </button>
    );
};

export default HelpButton;
