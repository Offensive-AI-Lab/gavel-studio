// TutorialContext drives whether the sidebar Tutorial button shows the
// 5-slide welcome overview or the page-aware help registered by the
// current page. The tests below pin down:
//
//   * useTutorial() outside the provider returns the no-op default
//     (we don't throw — pages that import it without a provider should
//     still degrade gracefully)
//   * show() / showWelcome() / dismiss() set the right (mode, open) pair
//   * useTutorialContent registers content on mount, clears on unmount
//   * REGRESSION: under <StrictMode>, the mount effect→cleanup→effect
//     double-fire used to leave pageContent null because a ref-guard
//     skipped the second setPageContent call after cleanup wiped the
//     state. The fix is in TutorialContext.jsx; this test guards it.
//   * Owner-aware cleanup: if Page A unmounts AFTER Page B has already
//     registered, A's cleanup must NOT clear pageContent (otherwise
//     fast navigations clobber the new page's registration).

import React, { StrictMode } from 'react';
import { describe, it, expect } from 'vitest';
import { act, render, renderHook } from '@testing-library/react';
import { TutorialProvider, useTutorial, useTutorialContent } from '../../src/contexts/TutorialContext';

const sampleContent = {
    title: 'Page A',
    summary: 'sample',
    sections: [{ heading: 'Now', bullets: ['a', 'b'] }],
};

const wrapper = ({ children }) => <TutorialProvider>{children}</TutorialProvider>;
const strictWrapper = ({ children }) => (
    <StrictMode>
        <TutorialProvider>{children}</TutorialProvider>
    </StrictMode>
);

// Probe component: registers `content` AND exposes the latest pageContent
// via a data-testid so the test can read the post-mount state.
const Registrant = ({ content }) => {
    useTutorialContent(content);
    return null;
};

const ContentProbe = () => {
    const { pageContent } = useTutorial();
    return <span data-testid="content">{pageContent ? pageContent.title : 'NULL'}</span>;
};

describe('TutorialContext', () => {
    describe('useTutorial outside provider', () => {
        it('returns the no-op default rather than throwing', () => {
            // Pages can import useTutorial defensively; rendering one
            // outside the provider in a test harness shouldn't crash.
            const { result } = renderHook(() => useTutorial());
            expect(result.current.open).toBe(false);
            expect(result.current.mode).toBe('auto');
            expect(result.current.pageContent).toBeNull();
            expect(typeof result.current.show).toBe('function');
        });
    });

    describe('show / showWelcome / dismiss', () => {
        it('show() opens in auto mode (sidebar Tutorial button path)', () => {
            const { result } = renderHook(() => useTutorial(), { wrapper });
            act(() => result.current.show());
            expect(result.current.open).toBe(true);
            expect(result.current.mode).toBe('auto');
        });

        it('showWelcome() forces welcome mode regardless of registered page content', () => {
            // First-login auto-fire path. Even if a page later registers
            // content, mode='welcome' wins in Tutorial.jsx's resolver.
            const { result } = renderHook(() => useTutorial(), { wrapper });
            act(() => result.current.showWelcome());
            expect(result.current.open).toBe(true);
            expect(result.current.mode).toBe('welcome');
        });

        it('dismiss() closes the modal', () => {
            const { result } = renderHook(() => useTutorial(), { wrapper });
            act(() => result.current.show());
            act(() => result.current.dismiss());
            expect(result.current.open).toBe(false);
        });
    });

    describe('useTutorialContent', () => {
        it('registers content on mount', () => {
            const { result } = renderHook(
                () => {
                    useTutorialContent(sampleContent);
                    return useTutorial();
                },
                { wrapper },
            );
            expect(result.current.pageContent).toEqual(sampleContent);
        });

        // The bug: TutorialContext used to gate setPageContent behind a
        // useRef that tracked the last serialized content. In React 18
        // StrictMode (Vite dev default), every mount fires its effect
        // twice — effect → cleanup → effect — to surface cleanup bugs.
        // The first run set the ref AND pageContent; the cleanup wiped
        // pageContent; the second run saw the ref already matched and
        // skipped the setState, leaving pageContent stuck at null.
        //
        // Symptom in the running app: clicking the sidebar Tutorial
        // button on any page would always open the welcome slides,
        // never the page-aware help. This test catches that exact
        // regression.
        it('survives StrictMode mount double-fire (regression)', () => {
            const { result } = renderHook(
                () => {
                    useTutorialContent(sampleContent);
                    return useTutorial();
                },
                { wrapper: strictWrapper },
            );
            expect(result.current.pageContent).toEqual(sampleContent);
        });

        it('clears registration on unmount', () => {
            const { getByTestId, rerender } = render(
                <TutorialProvider>
                    <Registrant content={sampleContent} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('Page A');
            rerender(
                <TutorialProvider>
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('NULL');
        });

        it('owner-aware cleanup: a stale registrant does not wipe a newer registration', () => {
            // Without the `prev === content` check in the cleanup, a
            // fast navigation Page A → Page B where A unmounts AFTER
            // B's effect has already run would clear B's pageContent
            // back to null. This test pins that down.
            const contentA = { title: 'A', summary: '', sections: [] };
            const contentB = { title: 'B', summary: '', sections: [] };

            const { getByTestId, rerender } = render(
                <TutorialProvider>
                    <Registrant content={contentA} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('A');

            // B mounts alongside A — B's effect runs after A's, so the
            // shared pageContent state ends up holding B's content.
            rerender(
                <TutorialProvider>
                    <Registrant content={contentA} />
                    <Registrant content={contentB} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('B');

            // Now unmount A. A's cleanup compares prev (which is B's
            // content) against its own captured contentA reference;
            // they differ, so cleanup is a no-op. Result: B's
            // registration survives.
            rerender(
                <TutorialProvider>
                    <Registrant content={contentB} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('B');
        });

        it('updates registration when content materially changes', () => {
            // Pages re-render with fresh content objects on every render,
            // but only re-register when the JSON-serialized content
            // actually changes (the [serialized] effect dep handles
            // this). This test verifies a real change DOES propagate.
            const v1 = { title: 'v1', summary: '', sections: [] };
            const v2 = { title: 'v2', summary: 'changed', sections: [] };

            const { getByTestId, rerender } = render(
                <TutorialProvider>
                    <Registrant content={v1} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('v1');

            rerender(
                <TutorialProvider>
                    <Registrant content={v2} />
                    <ContentProbe />
                </TutorialProvider>,
            );
            expect(getByTestId('content').textContent).toBe('v2');
        });
    });
});
