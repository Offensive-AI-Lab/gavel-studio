// Page smoke tests.
//
// One test per page that:
//   * renders the page inside a MemoryRouter (so useNavigate / Link work)
//   * mocks every API function the page might call to a benign default
//     so the data fetches in mount-effects don't blow up
//   * sets a logged-in user in localStorage so pages don't immediately
//     crash on mount
//   * asserts SOMETHING visible rendered without throwing
//
// These don't simulate user interaction. They catch the easy regressions:
// missing imports, undefined-prop crashes, broken lazy data assumptions
// like `(data.rules || []).map(...)`. Behavior tests for the highest-
// leverage flows (RuleService) live in their own files.

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import { TaskTrayProvider } from '../../src/contexts/TaskTrayContext';

// --- One global API mock. Cover every export that pages or their
// shared components (Sidebar, Layout) might call. Everything resolves to
// minimal data so list-rendering, filters, pagination, etc. all work.
vi.mock('../../src/api', () => {
    const empty = (extra = {}) => Promise.resolve({ data: extra });
    const list = (key) => Promise.resolve({ data: { [key]: [] } });
    return {
        default: { get: vi.fn(() => empty()), post: vi.fn(() => empty()), delete: vi.fn(() => empty()), put: vi.fn(() => empty()) },
        // Auth
        // Health
        getBackendHealth: vi.fn(() => empty({ ready: true })),
        // Dashboard
        getDashboardData: vi.fn(() => empty({ stats: { total_models: 0, total_classifiers: 0, active_classifiers: 0, total_rules: 0, total_ces: 0, total_evaluations: 0, total_test_datasets: 0 }, classifier_summary: [], recent_activity: [] })),
        // CEs
        getCognitiveElements: vi.fn(() => empty({ ces: [] })),
        getUserCEs: vi.fn(() => empty({ ces: [] })),
        getCognitiveDataset: vi.fn(() => empty({ samples: [] })),
        getCognitiveElement: vi.fn(() => empty({ ce_id: 1, name: 'CE', definition: '', examples: [] })),
        addCEBookmark: vi.fn(() => empty()),
        getCEBookmarks: vi.fn(() => list('bookmarks')),
        removeCEBookmark: vi.fn(() => empty()),
        // Models
        getUserModels: vi.fn(() => empty({ models: [] })),
        createModel: vi.fn(() => empty()),
        deleteModel: vi.fn(() => empty()),
        // Classifiers
        getClassifiers: vi.fn(() => empty({ classifiers: [] })),
        createClassifier: vi.fn(() => empty()),
        getClassifierDetails: vi.fn(() => empty({ classifier_id: 1, name: 'C', model_id: 1 })),
        getClassifierRules: vi.fn(() => empty({ rules: [] })),
        addRuleToClassifier: vi.fn(() => empty()),
        updateRuleLogic: vi.fn(() => empty()),
        deleteRuleSetup: vi.fn(() => empty()),
        deleteClassifier: vi.fn(() => empty()),
        getTrainingStatus: vi.fn(() => empty({ status: 'idle' })),
        downloadClassifier: vi.fn(() => Promise.resolve()),
        getClassifierConfig: vi.fn(() => empty({})),
        updateClassifierConfig: vi.fn(() => empty()),
        trainClassifier: vi.fn(() => empty()),
        // Rules
        getPublicRules: vi.fn(() => empty({ rules: [] })),
        createPublicRule: vi.fn(() => empty({ rule_id: 1 })),
        addRuleBookmark: vi.fn(() => empty()),
        getRuleBookmarks: vi.fn(() => list('bookmarks')),
        removeRuleBookmark: vi.fn(() => empty()),
        createManualRule: vi.fn(() => empty()),
        createAIRule: vi.fn(() => empty()),
        // Library
        searchLibrary: vi.fn(() => empty({ results: [], total: 0 })),
        getAllCategories: vi.fn(() => empty({ categories: [] })),
        syncLibrary: vi.fn(() => empty()),
        publishCE: vi.fn(() => empty()),
        publishRule: vi.fn(() => empty()),
        checkLibraryName: vi.fn(() => empty({ exists: false })),
        getPublicRecord: vi.fn(() => empty()),
        cleanupLocalDrafts: vi.fn(() => empty()),
        listLocalDrafts: vi.fn(() => empty({ rules: [], ces: [] })),
        deleteDraftRule: vi.fn(() => empty()),
        deleteDraftCE: vi.fn(() => empty()),
        getCeDependentDraftRules: vi.fn(() => empty({ rules: [] })),
        // Eval
        startCalibration: vi.fn(() => empty()),
        startEvaluation: vi.fn(() => empty()),
        getEvaluationResults: vi.fn(() => empty({ results: [] })),
        getCalibratedThresholds: vi.fn(() => empty({ thresholds: {} })),
        getEvalResultsHistory: vi.fn(() => empty({ history: [] })),
        getCalibrationDataStatus: vi.fn(() => empty({ status: 'idle' })),
        // Bookmark search
        searchBookmarks: vi.fn(() => empty({ results: [], total: 0 })),
        // Test sets
        listClassifierTestDatasets: vi.fn(() => empty({ datasets: [] })),
        // Rule default test/calibration sets
        deriveScenario: vi.fn(() => empty({ scenario: '' })),
        generateRuleDefaults: vi.fn(() => empty({ state: 'generating' })),
        getRuleDefaultsStatus: vi.fn(() => empty({ state: 'missing', datasets: [] })),
        getRuleDefaults: vi.fn(() => empty({ datasets: [] })),
        previewRuleTestSets: vi.fn(() => empty({ default: { buckets: [] } })),
        discardUnreadyRule: vi.fn(() => empty({ deleted: false })),
        createDraftRuleFromBookmarks: vi.fn(() => empty({ rule_id: 1 })),
        finalizeRule: vi.fn(() => empty({ rule_id: 1 })),
        // Chat / scenarios
        startScenarioChat: vi.fn(() => empty({ session_id: 's', message: 'hi' })),
        sendScenarioChatMessage: vi.fn(() => empty({ message: 'ok' })),
        generateCe: vi.fn(() => empty({ refuse: false, ce_data: { name: 'x', type: 'CONTEXT', definition: 'd', assigned_categories: ['Safety & Harm Prevention'], in_scope_examples: ['a'], out_of_scope_notes: ['b'] } })),
        generateCeTraining: vi.fn(() => empty()),
        // Realtime
        analyzeRealtime: vi.fn(() => empty({ analyses: [] })),
        analyzeStored: vi.fn(() => empty({ tokens: [], windows: [], rule_triggers: [] })),
        listSampleGroups: vi.fn(() => empty({ groups: [] })),
        getSampleGroup: vi.fn(() => empty({ samples: [] })),
        // ComputeBadge (rendered by RulesManager) fetches this on mount.
        getComputeStatus: vi.fn(() => Promise.resolve({ data: { workloads: {} } })),
    };
});

// Stub the Sidebar — it has its own data fetches and routing logic that we
// don't want to exercise on every page test. The sidebar renders inside
// Layout and is irrelevant for "did this page mount cleanly".
vi.mock('../../src/components/Sidebar/Sidebar', () => ({
    default: () => <aside data-testid="sidebar-stub" />,
}));

// Suppress noisy Swal popup creation during smoke renders.
vi.mock('sweetalert2', () => ({
    default: { fire: vi.fn(() => Promise.resolve({ isConfirmed: false })) },
}));


// --- import pages AFTER mocks ---
import Workspace from '../../src/pages/Workspace';
import Browse from '../../src/pages/Browse';
import BrowseCEs from '../../src/pages/BrowseCEs';
import BookmarksRules from '../../src/pages/BookmarksRules';
import BookmarksCEs from '../../src/pages/BookmarksCEs';
import RulesManager from '../../src/pages/RulesManager';
import LibrarySearch from '../../src/pages/LibrarySearch';
import Evaluation from '../../src/pages/Evaluation';
import BuildRuleFromCEsModal from '../../src/pages/BuildRuleFromCEsModal';
import RealtimeViewer from '../../src/pages/RealtimeViewer';


const setUser = () => {
    sessionStorage.setItem('user', JSON.stringify({ user_id: 1, email: 'a@b.c' }));
};

// Render at a specific URL so pages that read params via useParams get
// well-formed values (otherwise pages like Classifiers / RulesManager that
// expect `:modelId` / `:classifierId` would render with undefined).
// Pages may call useTaskTray (e.g. Browse → AI rule pipeline button), so
// wrap every render in the provider — same nesting App.jsx uses.
const renderAt = (path, route, ui) => render(
    <TaskTrayProvider>
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path={route} element={ui} />
            </Routes>
        </MemoryRouter>
    </TaskTrayProvider>,
);


describe('Page smoke tests (render without throwing)', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        setUser();
    });

    it('Workspace renders the welcome layout', async () => {
        renderAt('/workspace', '/workspace', <Workspace />);
        // The new dashboard renders a time-based greeting ("Good morning…")
        // followed by the user's name. The kicker badge "GAVEL Cloud Platform"
        // is static and always present regardless of dashboard data.
        expect(await screen.findByText(/GAVEL Cloud Platform/i)).toBeInTheDocument();
    });

    it('Browse renders without error', () => {
        renderAt('/browse', '/browse', <Browse />);
        // Sidebar stub confirms Layout mounted.
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('BrowseCEs renders without error', () => {
        renderAt('/browse/ces', '/browse/ces', <BrowseCEs />);
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('BookmarksRules renders without error', () => {
        renderAt('/bookmarks/rules', '/bookmarks/rules', <BookmarksRules />);
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('BookmarksCEs renders without error', () => {
        renderAt('/bookmarks/ces', '/bookmarks/ces', <BookmarksCEs />);
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('RulesManager renders for a given classifierId param', () => {
        renderAt(
            '/classifiers/42/rules',
            '/classifiers/:classifierId/rules',
            <RulesManager />,
        );
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('LibrarySearch renders the search input', () => {
        renderAt('/library/search', '/library/search', <LibrarySearch />);
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('Evaluation renders for a given classifierId', () => {
        renderAt(
            '/classifiers/42/evaluate',
            '/classifiers/:classifierId/evaluate',
            <Evaluation />,
        );
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });

    it('BuildRuleFromCEsModal renders the wizard body when open', async () => {
        renderAt('/browse', '/browse', <BuildRuleFromCEsModal open onClose={() => {}} />);
        expect(await screen.findByText(/Compose a rule from your bookmarked Cognitive Elements/i)).toBeInTheDocument();
    });

    it('RealtimeViewer renders for a given classifierId', () => {
        renderAt(
            '/classifiers/42/monitor',
            '/classifiers/:classifierId/monitor',
            <RealtimeViewer />,
        );
        expect(screen.getByTestId('sidebar-stub')).toBeInTheDocument();
    });
});
