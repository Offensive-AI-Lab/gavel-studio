import { useState } from 'react';

// "About this step / page" guidance panel. Used at the top of pages
// where the user benefits from a paragraph explaining what the page
// does and what they need to do next.
//
// Took the place of the global Tutorial popup from the sidebar — that
// modal was always-on-top and forced the user to dismiss it before
// reading the page. An inline panel sits in the natural reading flow
// and can be skimmed once or ignored thereafter.
//
// Usage:
//   <Explainer title="About this step">
//     <p>Plain-language paragraph...</p>
//     <ul><li>...</li></ul>
//   </Explainer>
//
// `title` defaults to "About this page".
//
// `collapsible` (opt-in) turns the header into a toggle so dense pages (e.g. the
// realtime monitor) can keep the guidance available without it eating the
// screen. `defaultOpen` sets the initial state when collapsible. Non-collapsible
// usage is unchanged — always expanded.
const Explainer = ({ title = 'About this page', children, collapsible = false, defaultOpen = true }) => {
    const [open, setOpen] = useState(collapsible ? defaultOpen : true);
    return (
        <div style={panel}>
            <h3
                style={{
                    ...panelTitle,
                    margin: open ? '0 0 8px' : 0,
                    cursor: collapsible ? 'pointer' : 'default',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
                }}
                onClick={collapsible ? () => setOpen((o) => !o) : undefined}
            >
                <span>{title}</span>
                {collapsible && (
                    <span style={{ fontSize: 11, fontWeight: 600, opacity: 0.85, textTransform: 'none', letterSpacing: 0 }}>
                        {open ? 'hide ▲' : 'show ▼'}
                    </span>
                )}
            </h3>
            {open && <div style={body}>{children}</div>}
        </div>
    );
};

// Indigo-tinted panel so it reads as guidance and not as a control.
// Matches the rest of the dark glass theme.
const panel = {
    background: 'rgba(99, 102, 241, 0.10)',
    border: '1px solid rgba(129, 140, 248, 0.30)',
    borderRadius: 12,
    padding: '18px 20px',
    marginBottom: 16,
    color: '#e2e8f0',
};

const panelTitle = {
    margin: '0 0 8px',
    fontSize: 14,
    fontWeight: 700,
    color: '#c7d2fe',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
};

// Style child elements consistently without forcing each call site to
// pass their own style props. Picks up `p`, `ul`, `li`, `strong`, `em`
// inside Explainer's children.
const body = {
    fontSize: 13,
    lineHeight: 1.55,
    color: '#cbd5e1',
};

export default Explainer;
