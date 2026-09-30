// Is a local LLM set? — mirrors useOpenAiKeyStatus.js's read-only status
// hook. AI surfaces combine this with useOpenAiKeyStatus to decide whether
// to show the "needs a key or a local model" note before the user types.
//
// A status call that itself fails counts as CONFIGURED, same reasoning as
// useOpenAiKeyStatus: a transient error must never accuse the user of
// missing access — the real 503 contract error still catches it at call time.
import { useCallback, useEffect, useRef, useState } from 'react';
import { getLocalLlmStatus } from '../api';
import { subscribeLocalLlmSaved } from '../components/LocalLlmModal/localLlmPrompt';

export default function useLocalLlmStatus() {
    const [configured, setConfigured] = useState(true);
    const [checked, setChecked] = useState(false);
    const aliveRef = useRef(true);

    const refresh = useCallback(async () => {
        try {
            const res = await getLocalLlmStatus();
            if (!aliveRef.current) return;
            const value = res?.data?.configured;
            setConfigured(typeof value === 'boolean' ? value : true);
        } catch {
            if (!aliveRef.current) return;
            setConfigured(true);
        } finally {
            if (aliveRef.current) setChecked(true);
        }
    }, []);

    useEffect(() => {
        aliveRef.current = true;
        refresh();
        return () => { aliveRef.current = false; };
    }, [refresh]);

    useEffect(() => subscribeLocalLlmSaved(refresh), [refresh]);

    return { configured, checked, refresh };
}
