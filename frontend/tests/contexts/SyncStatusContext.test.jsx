// Tests for SyncStatusContext — the library-sync "is the cache fresh?"
// store the sidebar reads + the LibrarySyncStream writes.
//
// Pins down:
//   * the default (no-provider) context value: status 'unknown',
//     pulling false, lastCheckedAt null, and no-op setters that don't throw
//   * the provider's initial value
//   * setStatus updates status AND stamps lastCheckedAt
//   * setPulling toggles the transient pulling flag without touching status
//   * a consumer component re-renders to reflect updates

import React from 'react';
import { describe, it, expect } from 'vitest';
import { act, render, renderHook, screen } from '@testing-library/react';
import {
    SyncStatusProvider,
    useSyncStatus,
} from '../../src/contexts/SyncStatusContext';

const wrapper = ({ children }) => <SyncStatusProvider>{children}</SyncStatusProvider>;

describe('SyncStatusContext', () => {
    describe('default value (no provider)', () => {
        it('returns the no-op default rather than throwing', () => {
            const { result } = renderHook(() => useSyncStatus());
            expect(result.current.status).toBe('unknown');
            expect(result.current.pulling).toBe(false);
            expect(result.current.lastCheckedAt).toBeNull();
            expect(typeof result.current.setStatus).toBe('function');
            expect(typeof result.current.setPulling).toBe('function');
            // No-op setters: calling them must not throw.
            expect(() => result.current.setStatus('available')).not.toThrow();
            expect(() => result.current.setPulling(true)).not.toThrow();
        });
    });

    describe('provider initial value', () => {
        it('starts unknown / not pulling / no lastCheckedAt', () => {
            const { result } = renderHook(() => useSyncStatus(), { wrapper });
            expect(result.current.status).toBe('unknown');
            expect(result.current.pulling).toBe(false);
            expect(result.current.lastCheckedAt).toBeNull();
        });
    });

    describe('setStatus', () => {
        it('updates status and stamps lastCheckedAt', () => {
            const { result } = renderHook(() => useSyncStatus(), { wrapper });
            expect(result.current.lastCheckedAt).toBeNull();

            act(() => result.current.setStatus('available'));

            expect(result.current.status).toBe('available');
            expect(result.current.lastCheckedAt).toBeInstanceOf(Date);
        });

        it('can move status back to synced', () => {
            const { result } = renderHook(() => useSyncStatus(), { wrapper });
            act(() => result.current.setStatus('available'));
            act(() => result.current.setStatus('synced'));
            expect(result.current.status).toBe('synced');
        });
    });

    describe('setPulling', () => {
        it('toggles the transient pulling flag without touching status', () => {
            const { result } = renderHook(() => useSyncStatus(), { wrapper });
            act(() => result.current.setPulling(true));
            expect(result.current.pulling).toBe(true);
            // status untouched, and the poller-owned lastCheckedAt is not stamped.
            expect(result.current.status).toBe('unknown');
            expect(result.current.lastCheckedAt).toBeNull();

            act(() => result.current.setPulling(false));
            expect(result.current.pulling).toBe(false);
        });
    });

    describe('consumer component', () => {
        const Consumer = () => {
            const { status, pulling, setStatus, setPulling } = useSyncStatus();
            return (
                <div>
                    <span data-testid="status">{status}</span>
                    <span data-testid="pulling">{String(pulling)}</span>
                    <button onClick={() => setStatus('available')}>set-available</button>
                    <button onClick={() => setPulling(true)}>start-pull</button>
                </div>
            );
        };

        it('re-renders to reflect status + pulling updates', () => {
            render(
                <SyncStatusProvider>
                    <Consumer />
                </SyncStatusProvider>,
            );
            expect(screen.getByTestId('status').textContent).toBe('unknown');
            expect(screen.getByTestId('pulling').textContent).toBe('false');

            act(() => screen.getByText('set-available').click());
            expect(screen.getByTestId('status').textContent).toBe('available');

            act(() => screen.getByText('start-pull').click());
            expect(screen.getByTestId('pulling').textContent).toBe('true');
        });
    });
});
