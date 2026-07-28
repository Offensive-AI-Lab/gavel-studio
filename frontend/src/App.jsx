import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { FiLoader } from 'react-icons/fi';
import './App.css';
import RulesManager from './pages/RulesManager';
import Guardrails from './pages/Guardrails';
import Workspace from './pages/Workspace';
import Browse from './pages/Browse';
import BrowseRuleSets from './pages/BrowseRuleSets';
import BrowseCEs from './pages/BrowseCEs';
import Bookmarks from './pages/Bookmarks';
import Evaluation from './pages/Evaluation';
import RulePage from './pages/RulePage';
import RuleSetPage from './pages/RuleSetPage';
import RealtimeViewer from './pages/RealtimeViewer';
import LibrarySearch from './pages/LibrarySearch';
import Profile from './pages/Profile';
import Community from './pages/Community';
import { TaskTrayProvider } from './contexts/TaskTrayContext';
import { TutorialProvider } from './contexts/TutorialContext';
import { SyncStatusProvider } from './contexts/SyncStatusContext';
import TaskTray from './components/TaskTray/TaskTray';
import LibrarySyncStream from './components/LibrarySyncStream/LibrarySyncStream';
import ComparePolicy from './pages/ComparePolicy';
import Tutorial from './components/Tutorial/Tutorial';
import HelpButton from './components/Tutorial/HelpButton';
import { getBackendHealth } from './api';
import { seedLocalUser } from './localUser';

const WarmingUpSplash = ({ status }) => (
  <div style={{
    minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'radial-gradient(ellipse 60% 50% at 12% 0%, rgba(99, 102, 241, 0.30), transparent 60%), radial-gradient(ellipse 70% 50% at 50% 105%, rgba(139, 92, 246, 0.22), transparent 60%), linear-gradient(180deg, #060a1a 0%, #0c1226 100%)',
    fontFamily: 'inherit',
    color: '#e2e8f0',
  }}>
    <div style={{
      width: 'min(440px, 90vw)',
      background: 'rgba(15, 23, 42, 0.85)',
      borderRadius: 18,
      padding: 32, textAlign: 'center',
      border: '1px solid rgba(148, 163, 184, 0.18)',
      backdropFilter: 'blur(20px)',
      boxShadow: '0 24px 48px -12px rgba(2, 6, 23, 0.65), 0 8px 24px -8px rgba(99, 102, 241, 0.30)',
    }}>
      <div style={{
        width: 56, height: 56, borderRadius: '50%',
        background: 'linear-gradient(135deg, #818cf8, #a78bfa)',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        marginBottom: 16, color: '#fff', animation: 'rotate 1.4s linear infinite',
        boxShadow: '0 8px 24px -4px rgba(99, 102, 241, 0.65)',
      }}>
        <FiLoader size={28} />
      </div>
      <h2 style={{ margin: '0 0 8px', color: '#f8fafc', fontSize: '1.3rem' }}>
        Connecting to Gavel
      </h2>
      <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.95rem', lineHeight: 1.55 }}>
        {status === 'unreachable'
          ? 'Backend not responding yet — this can take up to a minute on the first run after a reboot. Retrying...'
          : 'Loading models and warming up. This only takes a few seconds.'}
      </p>
      <style>{`@keyframes rotate { to { transform: rotate(360deg); } }`}</style>
    </div>
  </div>
);

// Seed the single local user at module load — BEFORE any component renders, so
// the pages that read sessionStorage during their first render already find it.
seedLocalUser();

function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [healthStatus, setHealthStatus] = useState('checking'); // 'checking' | 'unreachable' | 'warming' | 'ready'

  useEffect(() => {
    let cancelled = false;
    // The embedding warmup blocks the GIL for ~15s on a cold start, which
    // makes the first few /health calls slow even when the backend is fine.
    // We only flip to 'unreachable' (the loud error message) after a couple
    // of consecutive failures so a user reloading mid-warmup doesn't see it.
    let consecutiveFailures = 0;
    const check = async () => {
      try {
        const res = await getBackendHealth();
        if (cancelled) return;
        consecutiveFailures = 0;
        if (res.data?.ready) {
          setBackendReady(true);
          setHealthStatus('ready');
          return;
        }
        setHealthStatus('warming');
      } catch {
        if (cancelled) return;
        consecutiveFailures += 1;
        setHealthStatus(consecutiveFailures >= 3 ? 'unreachable' : 'warming');
      }
      // Retry while not ready
      if (!cancelled) setTimeout(check, 1500);
    };
    check();

    return () => { cancelled = true; };
  }, []);

  if (!backendReady) return <WarmingUpSplash status={healthStatus} />;

  return (
    <TaskTrayProvider>
      <TutorialProvider>
        <SyncStatusProvider>
          <LibrarySyncStream />
          <Router>
            <TaskTray />
            {/* Tutorial reads navigate() and the open flag, so it has to
                live inside the Router and the TutorialProvider. */}
            <Tutorial />
            {/* Floating "?" that opens this page's contextual help. Uses
                useLocation(), so it also lives inside the Router. */}
            <HelpButton />
            <Routes>
              {/* No login: "/" is the app itself. /login and /register are
                  kept as redirects so old bookmarks don't 404. */}
              <Route path="/" element={<Navigate to="/workspace" replace />} />
              <Route path="/login" element={<Navigate to="/workspace" replace />} />
              <Route path="/register" element={<Navigate to="/workspace" replace />} />
              <Route path="/workspace" element={<Workspace />} />
              {/* Community hub: Rules + CEs (content) and People (contributors)
                  under one section with tabs. /browse* are kept as redirects so
                  old links/bookmarks keep working. */}
              <Route path="/community" element={<Browse />} />
              <Route path="/community/rule-sets" element={<BrowseRuleSets />} />
              <Route path="/community/ces" element={<BrowseCEs />} />
              <Route path="/community/people" element={<Community />} />
              <Route path="/browse" element={<Navigate to="/community" replace />} />
              <Route path="/browse/ces" element={<Navigate to="/community/ces" replace />} />
              <Route path="/browse/rule-sets" element={<Navigate to="/community/rule-sets" replace />} />
              <Route path="/library/search" element={<LibrarySearch />} />
              {/* My Bookmarks: one page with internal Rules/CEs tabs. The
                  /bookmarks/rules and /bookmarks/ces deep links (used by
                  RulesManager's "Add from Bookmarked …" tiles) all resolve
                  here and open the matching tab. */}
              <Route path="/bookmarks" element={<Bookmarks />} />
              <Route path="/bookmarks/rules" element={<Bookmarks />} />
              <Route path="/bookmarks/rule-sets" element={<Bookmarks />} />
              <Route path="/bookmarks/ces" element={<Bookmarks />} />
              {/* Drafts now live inside Your Library (Rules/CEs tabs); keep the
                  old paths working by redirecting them there. */}
              <Route path="/bookmarks/drafts" element={<Navigate to="/bookmarks" replace />} />
              <Route path="/drafts" element={<Navigate to="/bookmarks" replace />} />
              <Route path="/profile/:username" element={<Profile />} />
              {/* Primary private-space flow: guardrails (rule sets) first,
                  model chosen at train time. Models are managed inside the
                  guardrail's Choose-Model flow — there's no standalone page. */}
              <Route path="/guardrails" element={<Guardrails />} />
              <Route path="/classifiers/:classifierId/rules" element={<RulesManager />} />
              <Route path="/classifiers/:classifierId/evaluate" element={<Evaluation />} />
              {/* Side-by-side comparison of guardrails trained on the same policy. */}
              <Route path="/classifiers/:classifierId/compare" element={<ComparePolicy />} />
              {/* Per-rule page (guardrail-independent): CEs + examples and the
                  rule's single auto-generated test + calibration set. */}
              <Route path="/rules/:ruleId" element={<RulePage />} />
              {/* Per-rule-set page (Community): member rules + their CEs, with
                  bookmark + fork-into-my-workspace actions. */}
              <Route path="/rule-sets/:ruleSetPublicId" element={<RuleSetPage />} />
              <Route path="/classifiers/:classifierId/monitor" element={<RealtimeViewer />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Router>
        </SyncStatusProvider>
      </TutorialProvider>
    </TaskTrayProvider>
  );
}

export default App;