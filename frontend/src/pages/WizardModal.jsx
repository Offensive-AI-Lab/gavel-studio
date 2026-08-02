// In-page modal host for the generation wizards.
//
// Same run-management as WizardShell (bootstrap on open, patchStep, advance),
// but rendered inside a GlassModal overlay on whatever page the user is on —
// no route change. The flows auto-advance (step 1 finalizes -> step 2), and
// the step-2 "Approve & Build" calls advance() past the last step, which fires
// onFinish (the background build) and closes the modal. So there's no sidebar
// or Prev/Next footer here — just the active step.
import { useState, useEffect, useCallback, useRef } from 'react';
import GlassModal from '../components/GlassModal/GlassModal';
import { updatePipelineStep } from '../api';

export default function WizardModal({
    open,
    onClose,
    title,
    steps,
    stepComponents,
    classifierId = null,
    bootstrap,
    onFinish,
    onAbandon,
}) {
    const [run, setRun] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    // Mirror `run` into a ref so the close handler can read the latest value
    // without re-subscribing the effect. `finishedRef` tells us whether the
    // user approved (→ background build owns the run) vs abandoned it.
    const runRef = useRef(null);
    const finishedRef = useRef(false);
    useEffect(() => { runRef.current = run; }, [run]);

    // Each open starts a BRAND-NEW run (the wrapper's bootstrap creates one —
    // it never resumes), so reopening after a close always gives a fresh
    // conversation. On close, if the run wasn't approved, abandon it (and its
    // half-built drafts) so nothing lingers.
    useEffect(() => {
        if (!open) {
            const r = runRef.current;
            if (r && !finishedRef.current) {
                try { onAbandon?.(r); } catch { /* best-effort */ }
            }
            setRun(null); setError(null);
            return;
        }
        finishedRef.current = false;
        let cancelled = false;
        (async () => {
            try {
                setLoading(true); setError(null); setRun(null);
                const r = await bootstrap();
                if (!cancelled && r) setRun(r);
            } catch (e) {
                if (!cancelled) setError(e?.response?.data?.detail || e.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open]);

    // IMPORTANT: read/write through runRef, not the closure `run`. Steps often
    // patch their data and then immediately advance in the same tick; using the
    // stale closure value here would re-write the just-saved step with old data
    // (jsonb_set merges per-key, so that silently WIPES the field). Keeping
    // runRef synchronously in sync — and trusting the server's merged response —
    // avoids that.
    const apply = (next) => { runRef.current = next; setRun(next); return next; };

    const patchStep = useCallback(async (stepId, { status, data, advanceTo } = {}) => {
        const cur = runRef.current;
        if (!cur) return null;
        try {
            const res = await updatePipelineStep(cur.run_id, {
                stepId, status: status || 'in_progress', data, advanceTo,
            });
            return apply(res.data);
        } catch (e) {
            setError(e?.response?.data?.detail || e.message);
            return null;
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const goToStep = useCallback(async (targetKey) => {
        const cur = runRef.current;
        if (!cur) return;
        try {
            const res = await updatePipelineStep(cur.run_id, {
                stepId: cur.current_step,
                status: cur.steps?.[cur.current_step]?.status || 'in_progress',
                data: cur.steps?.[cur.current_step]?.data,
                advanceTo: targetKey,
            });
            apply(res?.data || { ...cur, current_step: targetKey });
        } catch (e) {
            console.error('goToStep failed:', e);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const advance = useCallback(async () => {
        const cur = runRef.current;
        if (!cur) return;
        const i = steps.findIndex(s => s.key === cur.current_step);
        const next = steps[i + 1]?.key;
        if (next) {
            await goToStep(next);
        } else {
            // Approved → the background build now owns this run; mark finished so
            // the close handler doesn't abandon it.
            finishedRef.current = true;
            try { await onFinish?.(cur); } catch (e) { console.error('onFinish failed:', e); }
            onClose?.();
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [steps, goToStep, onFinish, onClose]);

    const ActiveStep = run ? stepComponents[run.current_step] : null;

    return (
        <GlassModal isOpen={open} onClose={onClose} title={title} size="wide">
            {loading && <div style={{ padding: 20, color: '#cbd5e1' }}>Loading…</div>}
            {error && <div style={{ padding: 20, color: '#fca5a5' }}>{error}</div>}
            {!loading && !error && ActiveStep && (
                <ActiveStep
                    run={run}
                    classifierId={classifierId}
                    onPatchStep={patchStep}
                    onAdvance={advance}
                    onSkip={() => {}}
                    onBack={() => {}}
                    setRun={setRun}
                />
            )}
        </GlassModal>
    );
}
