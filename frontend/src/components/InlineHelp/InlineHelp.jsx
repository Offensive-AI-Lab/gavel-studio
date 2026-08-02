// InlineHelp — renders a page's explanation text directly ON the page, instead
// of hiding it behind the "?" tutorial modal. Collapsible, expanded by default
// (the whole point is the explanation is visible without a click).
//
// Content shape (a superset of what `useTutorialContent` registers, so simple
// page-help still works unchanged):
//   {
//     title:   string,
//     summary: string,                    // intro paragraph(s); \n\n splits
//     sections: [{
//        heading: string,
//        body:    string | string[],      // paragraph(s) under the heading
//        bullets: string[],               // a bullet list
//     }]
//   }
// `**bold**` is honoured inline in summary / body / bullets.
import { useState } from 'react';
import { FiInfo, FiChevronDown, FiChevronUp } from 'react-icons/fi';

// Render a string with **bold** spans → array of text / <strong> nodes.
const renderInline = (text) => {
    const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, i) => (
        p.startsWith('**') && p.endsWith('**')
            ? <strong key={i} style={{ color: '#e2e8f0', fontWeight: 700 }}>{p.slice(2, -2)}</strong>
            : <span key={i}>{p}</span>
    ));
};

// Split a paragraph blob on blank lines into <p> nodes.
const Paragraphs = ({ text, style }) =>
    String(text).split(/\n\s*\n/).map((para, i) => (
        <p key={i} style={{ ...style, marginTop: i === 0 ? 0 : 8 }}>{renderInline(para.trim())}</p>
    ));

const InlineHelp = ({ content, defaultOpen = true, style = {} }) => {
    // Remember whether the user collapsed THIS explanation (keyed by its title),
    // so it stays in the state they left it in — we don't force it open again.
    const storageKey = content?.title ? `gavel_help_open_${content.title}` : null;
    const [open, setOpen] = useState(() => {
        if (storageKey) {
            try {
                const v = localStorage.getItem(storageKey);
                if (v === '0') return false;
                if (v === '1') return true;
            } catch { /* ignore */ }
        }
        return defaultOpen;
    });
    const toggle = () => setOpen((o) => {
        const next = !o;
        if (storageKey) { try { localStorage.setItem(storageKey, next ? '1' : '0'); } catch { /* best-effort */ } }
        return next;
    });
    if (!content || (!content.summary && !((content.sections || []).length))) return null;
    const { title, summary, sections = [] } = content;
    return (
        <div style={{ ...wrapStyle, ...style }}>
            <button type="button" onClick={toggle} style={headerStyle} aria-expanded={open}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                    <FiInfo style={{ color: '#a5b4fc', flexShrink: 0 }} />
                    <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: '0.95rem' }}>{title || 'About this page'}</span>
                </span>
                {open ? <FiChevronUp style={{ color: '#94a3b8', flexShrink: 0 }} /> : <FiChevronDown style={{ color: '#94a3b8', flexShrink: 0 }} />}
            </button>
            {open && (
                <div style={bodyStyle}>
                    {summary && <Paragraphs text={summary} style={summaryStyle} />}
                    {sections.map((sec, i) => {
                        const bodies = sec.body == null ? [] : (Array.isArray(sec.body) ? sec.body : [sec.body]);
                        return (
                            <div key={i} style={{ marginTop: (i === 0 && !summary) ? 0 : 14 }}>
                                {sec.heading && <div style={headingStyle}>{sec.heading}</div>}
                                {bodies.map((b, k) => <Paragraphs key={k} text={b} style={summaryStyle} />)}
                                {(sec.bullets || []).length > 0 && (
                                    <ul style={ulStyle}>
                                        {sec.bullets.map((b, j) => <li key={j} style={liStyle}>{renderInline(b)}</li>)}
                                    </ul>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

const wrapStyle = {
    background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.55) 100%)',
    border: '1px solid rgba(129, 140, 248, 0.22)',
    borderRadius: 12,
    marginBottom: 18,
    overflow: 'hidden',
};
const headerStyle = {
    width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
    padding: '12px 16px', background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left',
};
const bodyStyle = { padding: '0 16px 16px' };
const summaryStyle = { margin: 0, color: '#cbd5e1', fontSize: '0.88rem', lineHeight: 1.55 };
const headingStyle = { fontSize: '0.72rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a5b4fc', marginBottom: 6 };
const ulStyle = { margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 };
const liStyle = { color: '#cbd5e1', fontSize: '0.85rem', lineHeight: 1.5 };

export default InlineHelp;
