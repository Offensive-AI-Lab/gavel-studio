// LibrarySyncStream — opens a Server-Sent-Events stream to the backend and
// reacts to live freshness push events. No polling.
//
// The backend periodically probes the gavel-rules repository for changes (without
// touching the DB) and pushes `update_available` / `synced` here over SSE. So
// the sidebar surfaces a "click to sync" badge when an update is
// published — and the user applies it on their click (handled in the Sidebar),
// never auto-mutated mid-session. The publisher's own commit doesn't flag them
// (their backend probe comes back "synced").
//
// Replaces the old 90s LibrarySyncPoller. The stream is reconnected on error
// (backend restart / network blip).

import { useEffect } from 'react';
import { useSyncStatus } from '../../contexts/SyncStatusContext';

// Unset = same-origin (the Vite dev proxy forwards /library to the backend);
// keep in sync with src/api.js.
const API_URL = import.meta.env.VITE_API_URL ?? '';
const RECONNECT_MS = 3000;

const LibrarySyncStream = () => {
    const { setStatus } = useSyncStatus();

    useEffect(() => {
        let es = null;
        let stopped = false;
        let retry = null;

        const open = () => {
            if (stopped) return;
            es = new EventSource(`${API_URL}/library/events`);

            es.onmessage = (e) => {
                let data = {};
                try { data = JSON.parse(e.data || '{}'); } catch { return; }
                if (data.event === 'update_available') {
                    setStatus('available');
                } else if (data.event === 'synced') {
                    setStatus('synced');
                } else if (data.event === 'connected') {
                    // Greet carries the current freshness so a tab opened after
                    // the push still shows the badge.
                    setStatus(data.available ? 'available' : 'synced');
                }
            };

            es.onerror = () => {
                // Network blip / backend restart → close and retry; the stream
                // re-greets with a fresh state on reconnect.
                if (es) es.close();
                if (!stopped) retry = setTimeout(open, RECONNECT_MS);
            };
        };

        open();
        return () => {
            stopped = true;
            if (retry) clearTimeout(retry);
            if (es) es.close();
        };
    }, [setStatus]);

    return null;
};

export default LibrarySyncStream;
