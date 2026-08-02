// SyncStatusContext — exposes "is the library up to date?" so the
// sidebar's sync indicator and any other interested component can
// react to it. Single source of truth so the LibrarySyncStream (which
// writes) and the Sidebar (which reads) stay in lockstep without
// prop-drilling.
//
// States:
//
//   'synced'      — local cache matches the registry; nothing to do.
//   'available'   — the registry has content the local cache hasn't pulled.
//                   Pushed instantly by LibrarySyncStream the moment someone
//                   publishes; the user clicks the sidebar indicator to apply
//                   it (we never mutate the DB mid-session on our own).
//   'unknown'     — never connected yet. Renders the same as 'synced'.
//
// `pulling` is a transient flag the sidebar uses to show a spinner
// while a user-triggered pull is in flight.

import { createContext, useCallback, useContext, useMemo, useState } from 'react';

const SyncStatusContext = createContext({
    status: 'unknown',
    pulling: false,
    lastCheckedAt: null,
    setStatus: () => {},
    setPulling: () => {},
});

export const SyncStatusProvider = ({ children }) => {
    const [status, setStatusRaw] = useState('unknown');
    const [pulling, setPulling] = useState(false);
    const [lastCheckedAt, setLastCheckedAt] = useState(null);

    const setStatus = useCallback((next) => {
        setStatusRaw(next);
        setLastCheckedAt(new Date());
    }, []);

    const value = useMemo(() => ({
        status,
        pulling,
        lastCheckedAt,
        setStatus,
        setPulling,
    }), [status, pulling, lastCheckedAt, setStatus]);

    return <SyncStatusContext.Provider value={value}>{children}</SyncStatusContext.Provider>;
};

export const useSyncStatus = () => useContext(SyncStatusContext);

export default SyncStatusContext;
