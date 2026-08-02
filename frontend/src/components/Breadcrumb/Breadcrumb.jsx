// Breadcrumb — a clickable hierarchy that replaces "back" arrows across pages.
//
// `items` is an ordered list from root → current page:
//   [{ label, icon?, to? }]
// Every item except the last (and any without `to`) is clickable and navigates
// to `to`. The last item is the current page (bold, not clickable).
import { useNavigate } from 'react-router-dom';
import { FiChevronRight } from 'react-icons/fi';

const Breadcrumb = ({ items = [], style = {} }) => {
    const navigate = useNavigate();
    return (
        <nav style={{ ...wrapStyle, ...style }} aria-label="Breadcrumb">
            {items.map((it, i) => {
                const last = i === items.length - 1;
                const Icon = it.icon;
                const clickable = !last && it.to != null;
                return (
                    <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        {i > 0 && <FiChevronRight size={13} style={sepStyle} />}
                        {clickable ? (
                            <button onClick={() => navigate(it.to)} style={btnStyle}>
                                {Icon && <Icon size={13} />} {it.label}
                            </button>
                        ) : (
                            <span style={last ? currentStyle : btnStyle}>
                                {Icon && <Icon size={13} />} {it.label}
                            </span>
                        )}
                    </span>
                );
            })}
        </nav>
    );
};

const wrapStyle = { display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 8 };
const btnStyle = { display: 'inline-flex', alignItems: 'center', gap: 5, background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 500, padding: '2px 4px', borderRadius: 6 };
const sepStyle = { color: '#475569', flexShrink: 0 };
const currentStyle = { display: 'inline-flex', alignItems: 'center', gap: 5, color: '#e2e8f0', fontSize: '0.85rem', fontWeight: 700, padding: '2px 4px' };

export default Breadcrumb;
