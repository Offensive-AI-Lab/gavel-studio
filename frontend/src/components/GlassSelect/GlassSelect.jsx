// GlassSelect — a themed dropdown that replaces the native <select> inside the
// dark glass modals. The native option list is OS-rendered (un-themable), so we
// render our own panel. The panel is portalled to <body> and positioned with the
// trigger's rect, so the modal's `overflow: hidden` can't clip it.
import { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronDown, FiCheck } from 'react-icons/fi';
import './GlassSelect.css';

const GlassSelect = ({ value, onChange, options = [], placeholder = 'Select…' }) => {
    const [open, setOpen] = useState(false);
    const [rect, setRect] = useState(null);
    const triggerRef = useRef(null);
    const panelRef = useRef(null);

    const selected = options.find(o => String(o.value) === String(value));

    useLayoutEffect(() => {
        if (open && triggerRef.current) {
            const r = triggerRef.current.getBoundingClientRect();
            setRect({ left: r.left, top: r.bottom + 6, width: r.width });
        }
    }, [open]);

    useEffect(() => {
        if (!open) return;
        const onDoc = (e) => {
            if (triggerRef.current?.contains(e.target)) return;
            if (panelRef.current?.contains(e.target)) return;
            setOpen(false);
        };
        const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
        document.addEventListener('mousedown', onDoc);
        document.addEventListener('keydown', onKey);
        return () => {
            document.removeEventListener('mousedown', onDoc);
            document.removeEventListener('keydown', onKey);
        };
    }, [open]);

    return (
        <div className="glass-select">
            <button
                type="button"
                ref={triggerRef}
                className={`glass-select__trigger${open ? ' open' : ''}`}
                onClick={() => setOpen(o => !o)}
            >
                <span className={selected ? '' : 'glass-select__placeholder'}>
                    {selected ? selected.label : placeholder}
                </span>
                <FiChevronDown size={18} className="glass-select__chevron" />
            </button>
            {open && rect && createPortal(
                <div
                    ref={panelRef}
                    className="glass-select__panel"
                    style={{ left: rect.left, top: rect.top, width: rect.width }}
                >
                    {options.map(o => {
                        const isSel = String(o.value) === String(value);
                        return (
                            <button
                                key={o.value}
                                type="button"
                                className={`glass-select__option${isSel ? ' selected' : ''}`}
                                onClick={() => { onChange(o.value); setOpen(false); }}
                            >
                                <span>{o.label}</span>
                                {isSel && <FiCheck size={16} className="glass-select__check" />}
                            </button>
                        );
                    })}
                </div>,
                document.body
            )}
        </div>
    );
};

export default GlassSelect;
