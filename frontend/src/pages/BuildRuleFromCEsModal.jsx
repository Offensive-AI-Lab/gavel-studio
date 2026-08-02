// In-page modal wrapper for "Build Rule from Bookmarked CEs".
//
// Opened from the Browse page's "Build Rule from CEs" button — no route change,
// mirroring the AI Rule/CE generation modals. The body (BuildRuleFromCEs) is
// mounted FRESH on each open via `{open && …}`, so reopening always starts a
// clean wizard; closing unmounts it, which fires the body's cleanup (discarding
// the provisional is_ready=FALSE rule if the user hadn't kicked off the build).
import GlassModal from '../components/GlassModal/GlassModal';
import BuildRuleFromCEs from './BuildRuleFromCEs';

export default function BuildRuleFromCEsModal({ open, onClose, baseRule = null }) {
    return (
        <GlassModal
            isOpen={open}
            onClose={onClose}
            title={baseRule ? `Edit Rule — ${baseRule.name}` : 'Build Rule from Bookmarked CEs'}
            size="wide"
        >
            {open && <BuildRuleFromCEs onClose={onClose} baseRule={baseRule} />}
        </GlassModal>
    );
}
