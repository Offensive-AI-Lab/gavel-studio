// Is an OpenAI key set? — the read-only half of the key plumbing.
//
// AI surfaces call this on mount so the "you need a key" note is up BEFORE the
// user types, instead of appearing only after their first message fails. It
// NEVER opens the modal: asking for the key stays tied to something the user
// did (sending a message, pressing Generate, pressing Set API key).
//
// A status call that itself fails (backend still booting, network blip) counts
// as CONFIGURED. A transient error must never accuse the user of a missing key
// — the real 503 contract error still catches it at call time.
//
// The modal broadcasts every successful save, so a note put up by this hook
// clears itself once the key is in place — no reload, no polling.
import { useCallback, useEffect, useRef, useState } from 'react';
import { getOpenAiKeyStatus } from '../api';
import { subscribeOpenAiKeySaved } from '../components/OpenAiKeyModal/openAiKeyPrompt';

export default function useOpenAiKeyStatus() {
    // Optimistic until the first answer lands, so nothing red flashes on mount.
    const [configured, setConfigured] = useState(true);
    const [checked, setChecked] = useState(false);
    const aliveRef = useRef(true);

    const refresh = useCallback(async () => {
        try {
            const res = await getOpenAiKeyStatus();
            if (!aliveRef.current) return;
            // Only an explicit boolean is trusted — a body without the field is
            // as unreliable as a failed call, so it reads as configured.
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

    // A key was just saved somewhere — ask again so the note goes away.
    useEffect(() => subscribeOpenAiKeySaved(refresh), [refresh]);

    return { configured, checked, refresh };
}
