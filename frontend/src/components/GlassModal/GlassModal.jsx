import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import './GlassModal.css';

const GlassModal = ({ isOpen, onClose, title, children, size = 'default' }) => {
    const [isMounted, setIsMounted] = useState(false);
    const [isAnimate, setIsAnimate] = useState(false);

    useEffect(() => {
        if (isOpen) {
            // Small delay to let the DOM render before triggering the animation
            const mountFrame = requestAnimationFrame(() => {
                setIsMounted(true);
                requestAnimationFrame(() => {
                    setIsAnimate(true);
                    document.body.style.overflow = 'hidden'; // Lock scroll
                });
            });
            return () => cancelAnimationFrame(mountFrame);
        } else {
            const closeFrame = requestAnimationFrame(() => {
                setIsAnimate(false);
            });
            // Wait for the CSS transition (300ms) to finish before removing from DOM
            const timer = setTimeout(() => {
                setIsMounted(false);
                document.body.style.overflow = 'unset'; // Unlock scroll
            }, 300);
            return () => {
                cancelAnimationFrame(closeFrame);
                clearTimeout(timer);
            };
        }
    }, [isOpen]);

    if (!isMounted) return null;

    // Portal to <body> so the overlay is never clipped or re-positioned by an
    // ancestor's overflow/transform/stacking context (e.g. a RuleCard, which has
    // overflow:hidden + a hover transform). Without this, opening the modal from
    // inside a card rendered it trapped/clipped within the card's box.
    return createPortal(
        <div
            className={`modal-backdrop ${isAnimate ? 'open' : ''}`}
            onClick={onClose}
        >
            <div
                className={`modal-container ${size === 'wide' ? 'wide' : ''} ${isAnimate ? 'open' : ''}`}
                onClick={(e) => e.stopPropagation()}
            >
                <div className="modal-header">
                    <h3>{title}</h3>
                    <button className="close-btn" onClick={onClose}>
                        <FiX size={20} />
                    </button>
                </div>
                <div className="modal-content">
                    {children}
                </div>
            </div>
        </div>,
        document.body,
    );
};

export default GlassModal;