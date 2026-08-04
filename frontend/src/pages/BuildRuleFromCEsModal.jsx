// In-page modal wrapper for "Build Rule from Bookmarked CEs".
//
// Opened from the Browse page's "Build Rule from CEs" button — no route change,
// mirroring the AI Rule/CE generation modals. The body (BuildRuleFromCEs) is
// mounted FRESH on each open via `{open && …}`, so reopening always starts a
// clean wizard; closing unmounts it, which fires the body's cleanup (discarding
// the provisional is_ready=FALSE rule if the user hadn't kicked off the build).
// Because that unmount is destructive, a close with real progress (the body
// reports it through `dirtyRef`) asks for confirmation BEFORE unmounting.
import { useRef, useCallback } from 'react';
import GlassModal from '../components/GlassModal/GlassModal';
import { showConfirmDialog } from '../components/ConfirmDialog/confirmDialog';
import BuildRuleFromCEs from './BuildRuleFromCEs';

export default function BuildRuleFromCEsModal({ open, onClose, baseRule = null }) {
    // BuildRuleFromCEs keeps this current: true once the wizard holds progress
    // worth keeping (past Pick CEs, name/selection changed, or a provisional
    // rule exists) and the build wasn't committed. A ref so the close handler
    // always reads the latest value without re-rendering the wrapper.
    const dirtyRef = useRef(false);
    // True while the discard-confirm dialog is showing, so a second backdrop
    // click (or the X) can't stack a second prompt.
    const confirmingRef = useRef(false);

    const requestClose = useCallback(async () => {
        if (!dirtyRef.current) { onClose?.(); return; }
        if (confirmingRef.current) return; // a confirm is already showing
        confirmingRef.current = true;
        try {
            const ok = await showConfirmDialog({
                title: 'Discard this rule draft?',
                message: 'Closing now discards your progress — including the provisional rule if one was created. This wizard cannot be resumed later.',
                confirmText: 'Discard',
                cancelText: 'Keep working',
                variant: 'warning',
            });
            if (ok) onClose?.();
        } finally {
            confirmingRef.current = false;
        }
    }, [onClose]);

    return (
        <GlassModal
            isOpen={open}
            onClose={requestClose}
            title={baseRule ? `Edit Rule — ${baseRule.name}` : 'Build Rule from Bookmarked CEs'}
            size="wide"
        >
            {open && <BuildRuleFromCEs onClose={requestClose} baseRule={baseRule} dirtyRef={dirtyRef} />}
        </GlassModal>
    );
}
