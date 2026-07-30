import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FiSliders, FiBarChart2, FiPlay, FiClock, FiCheckCircle, FiAlertTriangle, FiHome, FiShield, FiFileText, FiRefreshCw } from 'react-icons/fi';
import Layout from '../components/Layout/Layout';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import ReactiveButton from '../components/ReactiveButton/ReactiveButton';
import {
    getClassifierDetails,
    getEvaluationResults,
    startCalibration,
    startEvaluation,
    getCalibratedThresholds,
    listClassifierTestDatasets,
    getCalibrationDataStatus,
} from '../api';
import { useTutorialContent } from '../contexts/TutorialContext';
import InlineHelp from '../components/InlineHelp/InlineHelp';
import { evaluateModel } from '../components/InlineHelp/instructorHelp';

const TABS = [
    { key: 'calibration', label: 'Calibration', icon: FiSliders },
    { key: 'evaluation', label: 'Evaluation', icon: FiBarChart2 },
];

const StatusBadge = ({ type }) => {
    const cfg = {
        idle:    { color: '#cbd5e1', bg: 'rgba(148, 163, 184, 0.18)', border: 'rgba(148, 163, 184, 0.30)', label: 'Ready' },
        running: { color: '#fcd34d', bg: 'rgba(245, 158, 11, 0.20)',  border: 'rgba(251, 191, 36, 0.40)', label: 'Running...' },
        done:    { color: '#6ee7b7', bg: 'rgba(16, 185, 129, 0.20)',  border: 'rgba(52, 211, 153, 0.40)', label: 'Complete' },
        error:   { color: '#fca5a5', bg: 'rgba(239, 68, 68, 0.20)',   border: 'rgba(248, 113, 113, 0.40)', label: 'Error' },
    }[type] || { color: '#cbd5e1', bg: 'rgba(148, 163, 184, 0.18)', border: 'rgba(148, 163, 184, 0.30)', label: type };
    return (
        <span style={{
            padding: '4px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600,
            color: cfg.color, background: cfg.bg,
            border: `1px solid ${cfg.border}`,
        }}>{cfg.label}</span>
    );
};

// A live progress line shown under the Run button while a job runs — the
// calibrate/evaluate analogue of training's phase indicator. The text is
// published by the backend on the *_running row (metrics.phase) and refreshed
// by the page's poll, so the user always knows the current stage (fetching
// data, on the cluster, falling back to local, computing metrics, …).
const PhaseLine = ({ phase, fallback }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12, fontSize: 13, color: '#cbd5e1' }}>
        <FiClock size={14} style={{ color: '#fcd34d', flexShrink: 0 }} />
        <span>{phase || fallback}</span>
    </div>
);

export default function Evaluation() {
    const { classifierId } = useParams();
    const navigate = useNavigate();
    const [tab, setTab] = useState('calibration');
    const [guardrail, setGuardrail] = useState(null);
    const [results, setResults] = useState({ calibration: null, evaluation: null });
    const [loading, setLoading] = useState(true);
    const [calibrating, setCalibrating] = useState(false);
    const [evaluating, setEvaluating] = useState(false);
    const [thresholds, setThresholds] = useState(null);
    const [testDatasets, setTestDatasets] = useState([]);
    const [calibStatus, setCalibStatus] = useState(null);
    const [lastCalibStarted, setLastCalibStarted] = useState(0);
    const [lastEvalStarted, setLastEvalStarted] = useState(0);
    const [datasetPollTries, setDatasetPollTries] = useState(0);
    const [loadError, setLoadError] = useState(false);   // initial details fetch failed

    const fetchData = useCallback(async () => {
        try {
            const [detailsRes, resultsRes, datasetsRes, calibRes] = await Promise.all([
                getClassifierDetails(classifierId),
                getEvaluationResults(classifierId),
                listClassifierTestDatasets(classifierId),
                getCalibrationDataStatus(classifierId),
            ]);
            setGuardrail(detailsRes.data);
            setLoadError(false);
            setResults(resultsRes.data);
            setCalibStatus(calibRes.data);
            const ready = (datasetsRes.data?.datasets || []).filter(d => d.status === 'ready');
            setTestDatasets(ready);

            // Resume polling if calibration was running before the page refresh
            const cal = resultsRes.data?.calibration;
            if (cal?.eval_type === 'calibration_running') {
                setCalibrating(true);
                setLastCalibStarted(new Date(cal.created_at).getTime());
            }

            // Same idea for evaluation: a user who navigates away mid-run
            // and comes back must see "still evaluating" — not an idle
            // page that suggests they killed the run by leaving (they
            // didn't; FastAPI BackgroundTasks survive client disconnect).
            const ev = resultsRes.data?.evaluation;
            if (ev?.eval_type === 'evaluation_running') {
                setEvaluating(true);
                setLastEvalStarted(new Date(ev.created_at).getTime());
            }

            // Try loading thresholds
            try {
                const thrRes = await getCalibratedThresholds(classifierId);
                setThresholds(thrRes.data.thresholds);
            } catch { setThresholds(null); }
        } catch (err) {
            console.error('Failed to load evaluation data', err);
            setLoadError(true);
        } finally { setLoading(false); }
    }, [classifierId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    // Per-page tutorial — adapts to which tab is active and what state
    // calibration / evaluation are in for the current guardrail.
    const hasThresholds = !!thresholds;
    const evalReady = results?.evaluation?.eval_type === 'evaluation' && results?.evaluation?.metrics;
    // Once-per-training lock: a SUCCESSFUL calibration/evaluation row (post-train,
    // the backend already filters to the current training) means it's done and
    // can't be re-run until the guardrail is retrained. Errors don't lock —
    // the user keeps retrying until one succeeds.
    const alreadyCalibrated = results?.calibration?.eval_type === 'calibration';
    const alreadyEvaluated = results?.evaluation?.eval_type === 'evaluation';
    // Live phase strings published on the *_running rows (metrics.phase).
    const calibPhase = results?.calibration?.eval_type === 'calibration_running'
        ? (results.calibration.metrics?.phase || null) : null;
    const evalPhase = results?.evaluation?.eval_type === 'evaluation_running'
        ? (results.evaluation.metrics?.phase || null) : null;
    const pageHelp = {
        title: 'Evaluation',
        summary: 'Calibrate per-CE thresholds, then evaluate the trained rule set on each rule\'s test set to measure precision, recall, and F1. Suggested order: calibration first → evaluation second.',
        sections: [
            {
                heading: 'Right now',
                bullets: tab === 'calibration'
                    ? (calibrating
                        ? ['Calibration is running. The page polls every few seconds; stay or come back later — it survives navigation.']
                        : hasThresholds
                            ? ['Thresholds are calibrated. Switch to the Evaluation tab to score against test sets.']
                            : ['No thresholds yet. Click Run Calibration. Calibration uses the per-CE conversations bundled in the library; for fresh CEs you may need to generate calibration data first.'])
                    : (evaluating
                        ? ['Evaluation is running in the background. The status badge will switch to "Done" when it finishes — no need to wait on this page.']
                        : evalReady
                            ? ['Evaluation results are below: per-use-case TPR/FPR/F1, weighted averages, and ROC/PR AUC.']
                            : ['Click Run Evaluation. Every active rule\'s test set (positive + negative) is used automatically, and metrics are reported per rule.']),
            },
            {
                heading: 'How it fits together',
                bullets: [
                    'Calibration tunes per-CE thresholds (using ground-truth conversations) so the rule fires precisely when intended.',
                    'Evaluation runs the trained rule set on every active rule\'s test set in one pass and reports metrics per rule, plus weighted averages across rules.',
                    'AUC < 0.5 in the use-case row is a red flag — usually means the rule predicate is too loose or negatives are too similar to positives.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    // Poll while calibrating or evaluating — compare timestamps to detect
    // NEW results so we don't stop polling on a stale row from a previous
    // run.
    useEffect(() => {
        if (!calibrating && !evaluating) return;
        const interval = setInterval(async () => {
            try {
                const res = await getEvaluationResults(classifierId);
                setResults(res.data);

                const cal = res.data.calibration;
                const ev = res.data.evaluation;
                if (calibrating && cal && ['calibration', 'calibration_error'].includes(cal.eval_type)) {
                    const calTime = new Date(cal.created_at).getTime();
                    if (calTime > lastCalibStarted) {
                        setCalibrating(false);
                        if (cal.eval_type === 'calibration') {
                            try {
                                const thrRes = await getCalibratedThresholds(classifierId);
                                setThresholds(thrRes.data.thresholds);
                            } catch {}
                        }
                    }
                }
                // Stop polling on either a finished result OR a recorded
                // error. Earlier code only checked for the success row,
                // which left the UI stuck in "Evaluating..." forever
                // when the eval failed.
                if (evaluating && ev && ['evaluation', 'evaluation_error'].includes(ev.eval_type)) {
                    const evTime = new Date(ev.created_at).getTime();
                    if (evTime > lastEvalStarted) {
                        setEvaluating(false);
                    }
                }
            } catch {}
        }, 5000);
        return () => clearInterval(interval);
    }, [calibrating, evaluating, classifierId, lastCalibStarted, lastEvalStarted]);

    // Auto-refresh while test sets aren't ready yet. Loading this page lazily
    // pulls the guardrail's test sets from the registry (in
    // getCalibrationDataStatus); re-fetch a few times so the page updates and
    // the Evaluate button unlocks as soon as the download lands — no manual
    // refresh needed. Capped so a guardrail that genuinely has no test sets
    // doesn't poll forever.
    useEffect(() => {
        if (loading || calibrating || evaluating) return;
        if (testDatasets.length > 0) return;
        if (datasetPollTries >= 6) return;  // ~36s of retries
        const t = setTimeout(() => {
            setDatasetPollTries((n) => n + 1);
            fetchData();
        }, 6000);
        return () => clearTimeout(t);
    }, [loading, calibrating, evaluating, testDatasets.length, datasetPollTries, fetchData]);

    const handleCalibrate = async () => {
        setCalibrating(true);
        setLastCalibStarted(Date.now());
        try {
            await startCalibration(classifierId, {});
            setTab('calibration');
        } catch (err) {
            alert(err.response?.data?.detail || 'Calibration failed');
            setCalibrating(false);
        }
    };

    const handleEvaluate = async () => {
        setEvaluating(true);
        setLastEvalStarted(Date.now());
        try {
            // No dataset selection: each rule has exactly one test set, so the
            // backend auto-loads every active rule's test set (positive +
            // negative) and reports per-rule metrics from a single run.
            await startEvaluation(classifierId, {});
            setTab('evaluation');
        } catch (err) {
            alert(err.response?.data?.detail || 'Evaluation failed');
            setEvaluating(false);
        }
    };

    const modelName = guardrail?.model_name || 'Model';

    if (loading) return (
        <Layout>
            <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading...</div>
        </Layout>
    );

    return (
        <Layout currentModel={modelName}>
            <header className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
                <div>
                    <Breadcrumb items={[
                        { label: 'Hub', icon: FiHome, to: '/workspace' },
                        { label: 'Rule Sets', icon: FiShield, to: '/guardrails' },
                        { label: guardrail?.name || 'Rule Set', icon: FiFileText, to: `/classifiers/${classifierId}/rules` },
                        { label: 'Evaluation', icon: FiBarChart2 },
                    ]} />
                    <h1 style={{ margin: '4px 0 0' }}>Evaluate: {guardrail?.name}</h1>
                    <p style={{ margin: 0, color: '#64748b' }}>
                        Calibrate thresholds and evaluate rule set performance
                    </p>
                </div>
                {/* Compare this guardrail's results against others trained on the
                    same policy (same rules) on different base models. */}
                <button
                    onClick={() => navigate(`/classifiers/${classifierId}/compare`)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 10, border: '1px solid rgba(129,140,248,0.4)', background: 'rgba(99,102,241,0.16)', color: '#c7d2fe', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' }}
                    title="Compare rule sets trained on the same rules across different models"
                >
                    <FiBarChart2 size={14} /> Compare models
                </button>
            </header>

            {/* Load-error banner — the guardrail's data couldn't be fetched.
                Keep the tabs below usable but give a clear way forward. */}
            {loadError && (
                <div style={loadErrorBox}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#fecaca', fontWeight: 600 }}>
                        <FiAlertTriangle size={16} /> Couldn’t load this rule set’s evaluation data.
                    </div>
                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                        <button onClick={fetchData} style={loadErrorBtn}>
                            <FiRefreshCw size={14} /> Try again
                        </button>
                        <button onClick={() => navigate('/guardrails')} style={loadErrorBtn}>
                            <FiShield size={14} /> Back to Rule Sets
                        </button>
                        <button onClick={() => navigate('/workspace')} style={loadErrorBtn}>
                            <FiHome size={14} /> Go to Hub
                        </button>
                    </div>
                </div>
            )}

            {/* Tab bar */}
            <div style={tabBarStyle}>
                {TABS.map(t => (
                    <button key={t.key} onClick={() => setTab(t.key)}
                        style={{ ...tabStyle, ...(tab === t.key ? tabActiveStyle : {}) }}>
                        <t.icon size={14} /> {t.label}
                    </button>
                ))}
            </div>

            <InlineHelp content={evaluateModel} />

            {/* Tab content */}
            <div style={{ marginTop: 24 }}>
                {tab === 'calibration' && (
                    <CalibrationTab
                        results={results.calibration}
                        thresholds={thresholds}
                        calibrating={calibrating}
                        onCalibrate={handleCalibrate}
                        calibStatus={calibStatus}
                        phase={calibPhase}
                        alreadyCalibrated={alreadyCalibrated}
                    />
                )}
                {tab === 'evaluation' && (
                    <EvaluationTab
                        results={results.evaluation}
                        evaluating={evaluating}
                        calibrating={calibrating}
                        onEvaluate={handleEvaluate}
                        hasThresholds={!!thresholds}
                        testDatasets={testDatasets}
                        phase={evalPhase}
                        alreadyEvaluated={alreadyEvaluated}
                    />
                )}
            </div>
        </Layout>
    );
}


// ---------------------------------------------------------------------------
// Calibration Tab
// ---------------------------------------------------------------------------
function CalibrationTab({ results, thresholds, calibrating, onCalibrate, calibStatus, phase, alreadyCalibrated }) {
    const hasResults = results && results.eval_type === 'calibration' && results.thresholds;
    const hasError = results && results.eval_type === 'calibration_error';
    const allReady = calibStatus?.all_ready;
    const ces = calibStatus?.ces || [];

    return (
        <div>
            <div style={sectionStyle}>
                <h3 style={sectionTitle}>Threshold Calibration</h3>
                <p style={sectionDesc}>
                    Picks a per-CE threshold by maximising Youden's J (J = TPR - FPR),
                    scored against the rules' calibration dialogues using each rule's
                    CE groups and firing condition as the ground truth.
                </p>

                {ces.length > 0 && (
                    <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {ces.map(ce => (
                            <span key={ce.ce_id} style={{
                                padding: '3px 10px', borderRadius: 10, fontSize: 12, fontWeight: 600,
                                color: ce.has_calibration ? '#059669' : '#d97706',
                                background: ce.has_calibration ? '#d1fae5' : '#fef3c7',
                            }}>
                                {ce.name}: {ce.has_calibration ? 'Ready' : 'Missing'}
                            </span>
                        ))}
                    </div>
                )}

                {!allReady && ces.length > 0 && (
                    <div style={{ marginTop: 8, padding: '8px 12px', borderRadius: 8, background: 'rgba(245, 158, 11, 0.18)', border: '1px solid rgba(251, 191, 36, 0.40)', fontSize: 13, color: '#fcd34d' }}>
                        Some CEs are missing calibration data. Generate training data for those CEs first — calibration data is created automatically alongside it.
                    </div>
                )}

                <div style={{ display: 'flex', gap: 12, marginTop: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                    <ReactiveButton
                        label={calibrating ? 'Calibrating...' : alreadyCalibrated ? 'Calibrated' : 'Run Calibration'}
                        onClick={onCalibrate}
                        Icon={calibrating ? FiClock : alreadyCalibrated ? FiCheckCircle : FiPlay}
                        disabled={calibrating || !allReady || alreadyCalibrated}
                    />
                    <StatusBadge type={calibrating ? 'running' : hasError ? 'error' : hasResults ? 'done' : 'idle'} />
                    {alreadyCalibrated && (
                        <span style={{ color: '#34d399', fontSize: 13 }}>
                            <FiCheckCircle /> Calibrated for this training — retrain to recalibrate.
                        </span>
                    )}
                </div>
                {calibrating && <PhaseLine phase={phase} fallback="Starting calibration…" />}
                {hasError && results.metrics?.error && (
                    <div style={errorBox}><FiAlertTriangle /> {results.metrics.error}</div>
                )}
            </div>

            {hasResults && thresholds && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>Calibrated Thresholds</h3>
                    <div style={tableWrap}>
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>CE / Topic</th>
                                    <th style={thStyle}>Threshold</th>
                                    <th style={thStyle}>Patience</th>
                                    <th style={thStyle}>Youden's J</th>
                                    <th style={thStyle}>TPR</th>
                                    <th style={thStyle}>FPR</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(thresholds).map(([topic, params]) => (
                                    <tr key={topic}>
                                        <td style={tdStyle}>{topic}</td>
                                        <td style={tdStyle}>{params.threshold?.toFixed(3)}</td>
                                        <td style={tdStyle}>{params.patience}</td>
                                        <td style={tdStyle}>{params.youden_j?.toFixed(4)}</td>
                                        <td style={{ ...tdStyle, color: params.tpr_at_optimal >= 0.85 ? '#059669' : '#d97706' }}>
                                            {params.tpr_at_optimal?.toFixed(3)}
                                        </td>
                                        <td style={{ ...tdStyle, color: params.fpr_at_optimal <= 0.1 ? '#059669' : '#dc2626' }}>
                                            {params.fpr_at_optimal?.toFixed(3)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {hasResults && results.plots?.mosaic && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>Calibration Curves</h3>
                    <img
                        src={`data:image/png;base64,${results.plots.mosaic}`}
                        alt="Calibration curves"
                        style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid #e5e7eb' }}
                    />
                </div>
            )}
        </div>
    );
}


// ---------------------------------------------------------------------------
// Evaluation Tab
// ---------------------------------------------------------------------------
function EvaluationTab({ results, evaluating, calibrating, onEvaluate, hasThresholds, testDatasets, phase, alreadyEvaluated }) {
    const hasResults = results && results.eval_type === 'evaluation' && results.metrics;
    const hasError = results && results.eval_type === 'evaluation_error';
    const metrics = hasResults ? results.metrics : null;

    // The evaluator reports two kinds of use-case rows: the real rules, and the
    // two neutral pseudo-use-cases ("conversational"/"instructive") that score
    // the global neutral corpus (false-positive rate against benign content).
    // Show them in SEPARATE tables: rules in "Per Use-Case Metrics", neutral in
    // its own "Neutral Use Cases" table (TPR/FPR/Acc + support).
    const isNeutralUsecase = (u) =>
        ['conversational', 'instructive'].includes(
            String(u || '').toLowerCase().replace(/^neutral_/, '')
        );
    const allMetricRows = metrics?.metrics || [];
    const usecaseRows = allMetricRows.filter(
        r => !isNeutralUsecase(r.Usecase) && ((r.Support_Pos || 0) > 0 || (r.Support_Neg || 0) > 0)
    );
    const neutralRows = allMetricRows.filter(
        r => isNeutralUsecase(r.Usecase) && ((r.Support_Pos || 0) > 0 || (r.Support_Neg || 0) > 0)
    );
    // AUC needs both classes; neutral has no positives, so drop those rows.
    const aucRows = (metrics?.auc || []).filter(r => !isNeutralUsecase(r.Usecase));

    return (
        <div>
            <div style={sectionStyle}>
                <h3 style={sectionTitle}>Run Evaluation</h3>
                <p style={sectionDesc}>
                    Evaluates the rule set on every active rule's test set using the
                    calibrated thresholds.
                    {!hasThresholds && ' You must calibrate this rule set before you can evaluate it.'}
                </p>
                <div style={{ display: 'flex', gap: 12, marginTop: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                    <ReactiveButton
                        label={evaluating ? 'Evaluating...' : alreadyEvaluated ? 'Evaluated' : calibrating ? 'Calibration running…' : !hasThresholds ? 'Calibrate first' : 'Run Evaluation'}
                        onClick={onEvaluate}
                        Icon={evaluating || calibrating ? FiClock : alreadyEvaluated ? FiCheckCircle : FiPlay}
                        disabled={evaluating || calibrating || alreadyEvaluated || !hasThresholds || testDatasets.length === 0}
                    />
                    <StatusBadge type={evaluating ? 'running' : hasError ? 'error' : hasResults ? 'done' : 'idle'} />
                    {alreadyEvaluated && (
                        <span style={{ color: '#34d399', fontSize: 13 }}>
                            <FiCheckCircle /> Evaluated for this training — retrain to re-evaluate.
                        </span>
                    )}
                    {!alreadyEvaluated && hasThresholds && <span style={{ color: '#34d399', fontSize: 13 }}><FiCheckCircle /> Using calibrated thresholds</span>}
                    {!alreadyEvaluated && !hasThresholds && !calibrating && (
                        <span style={{ color: '#fbbf24', fontSize: 13 }}>
                            <FiAlertTriangle /> Not calibrated yet — run calibration first.
                        </span>
                    )}
                    {calibrating && (
                        <span style={{ color: '#fbbf24', fontSize: 13 }}>
                            <FiAlertTriangle /> Calibration is still running — evaluation unlocks when it finishes.
                        </span>
                    )}
                    {!calibrating && hasThresholds && testDatasets.length === 0 && (
                        <span style={{ color: '#fbbf24', fontSize: 13 }}>
                            <FiAlertTriangle /> No test sets ready yet for this rule set's rules.
                        </span>
                    )}
                </div>
                {evaluating && <PhaseLine phase={phase} fallback="Starting evaluation…" />}
                {hasError && results.metrics?.error && (
                    <div style={errorBox}><FiAlertTriangle /> {results.metrics.error}</div>
                )}
            </div>

            {metrics?.weighted_averages && Object.keys(metrics.weighted_averages).length > 0 && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>Weighted Averages</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                        {Object.entries(metrics.weighted_averages).map(([key, val]) => (
                            <div key={key} style={metricCard}>
                                <div style={{ fontSize: 12, color: '#94a3b8' }}>{key}</div>
                                <div style={{ fontSize: 24, fontWeight: 700, color: '#f1f5f9' }}>{(val * 100).toFixed(1)}%</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {usecaseRows.length > 0 && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>Per Use-Case Metrics</h3>
                    <div style={tableWrap}>
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>Use Case</th>
                                    <th style={thStyle}>TPR</th>
                                    <th style={thStyle}>FPR</th>
                                    <th style={thStyle}>Accuracy</th>
                                    <th style={thStyle}>F1</th>
                                    <th style={thStyle}>Pos Support</th>
                                    <th style={thStyle}>Neg Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                {usecaseRows.map((row, i) => (
                                    <tr key={i}>
                                        <td style={tdStyle}>{row.Usecase}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.TPR, true) }}>{row.TPR?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.FPR, false) }}>{row.FPR?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.Accuracy, true) }}>{row.Accuracy?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.F1, true) }}>{row.F1?.toFixed(3)}</td>
                                        <td style={tdStyle}>{row.Support_Pos}</td>
                                        <td style={tdStyle}>{row.Support_Neg}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {neutralRows.length > 0 && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>Neutral Use Cases</h3>
                    <p style={sectionDesc}>
                        How often a rule fires on the global neutral corpus —
                        everyday benign content nothing should trigger on. FPR is
                        the false-alarm rate; there are no positives here (TPR is
                        always 0). Support is shown as positives/negatives.
                    </p>
                    <div style={tableWrap}>
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>Use Case</th>
                                    <th style={thStyle}>TPR</th>
                                    <th style={thStyle}>FPR</th>
                                    <th style={thStyle}>Accuracy</th>
                                    <th style={thStyle}>Support (P/N)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {neutralRows.map((row, i) => (
                                    <tr key={i}>
                                        <td style={tdStyle}>{row.Usecase}</td>
                                        <td style={tdStyle}>{row.TPR?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.FPR, false) }}>{row.FPR?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.Accuracy, true) }}>{row.Accuracy?.toFixed(3)}</td>
                                        <td style={tdStyle}>{(row.Support_Pos || 0)}/{(row.Support_Neg || 0)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {aucRows.length > 0 && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>AUC Scores</h3>
                    <div style={tableWrap}>
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>Use Case</th>
                                    <th style={thStyle}>ROC AUC</th>
                                    <th style={thStyle}>PR AUC</th>
                                </tr>
                            </thead>
                            <tbody>
                                {aucRows.map((row, i) => (
                                    <tr key={i}>
                                        <td style={tdStyle}>{row.Usecase}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.ROC_AUC, true) }}>{row.ROC_AUC?.toFixed(3)}</td>
                                        <td style={{ ...tdStyle, ...colorVal(row.PR_AUC, true) }}>{row.PR_AUC?.toFixed(3)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Per-CE metrics intentionally NOT shown here. the reference
              * regular evaluate.py is per-usecase + overall only — exactly
              * what this tab mirrors. */}

            {results?.plots?.roc_all_usecases && (
                <div style={sectionStyle}>
                    <h3 style={sectionTitle}>ROC Curves</h3>
                    <img
                        src={`data:image/png;base64,${results.plots.roc_all_usecases}`}
                        alt="ROC curves"
                        style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid #e5e7eb' }}
                    />
                </div>
            )}
        </div>
    );
}


// ---------------------------------------------------------------------------
// Helpers & styles
// ---------------------------------------------------------------------------

function colorVal(val, higherIsBetter) {
    if (val == null) return {};
    if (higherIsBetter) {
        if (val >= 0.85) return { color: '#34d399', fontWeight: 600 };
        if (val >= 0.65) return { color: '#fbbf24' };
        return { color: '#f87171' };
    }
    // Lower is better (FPR)
    if (val <= 0.1) return { color: '#34d399', fontWeight: 600 };
    if (val <= 0.3) return { color: '#fbbf24' };
    return { color: '#f87171' };
}

const tabBarStyle = {
    display: 'flex', gap: 4, background: 'rgba(2, 6, 23, 0.55)', padding: 4,
    borderRadius: 10, width: 'fit-content', border: '1px solid rgba(148, 163, 184, 0.14)',
};
const tabStyle = {
    display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px',
    border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13,
    fontWeight: 500, background: 'transparent', color: '#94a3b8', transition: 'all 0.2s',
};
const tabActiveStyle = {
    background: 'linear-gradient(135deg, #818cf8 0%, #3b82f6 100%)', color: '#ffffff',
    boxShadow: '0 4px 12px -2px rgba(99, 102, 241, 0.55)',
};
const sectionStyle = {
    background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.62) 0%, rgba(15, 23, 42, 0.55) 100%)',
    borderRadius: 12, padding: 24, color: '#e2e8f0',
    border: '1px solid rgba(148, 163, 184, 0.14)', marginBottom: 16,
    backdropFilter: 'blur(12px)',
    boxShadow: '0 4px 12px rgba(2, 6, 23, 0.30)',
};
const sectionTitle = { margin: '0 0 4px', fontSize: 16, fontWeight: 600, color: '#f1f5f9' };
const sectionDesc = { margin: 0, fontSize: 13, color: '#94a3b8', lineHeight: 1.5 };
const errorBox = {
    marginTop: 12, padding: '10px 14px', borderRadius: 8,
    background: 'rgba(239, 68, 68, 0.18)', color: '#fecaca', fontSize: 13,
    border: '1px solid rgba(248, 113, 113, 0.40)',
    display: 'flex', alignItems: 'center', gap: 8,
};
const loadErrorBox = {
    display: 'flex', flexDirection: 'column', gap: 12,
    background: 'rgba(239, 68, 68, 0.14)', border: '1px solid rgba(248, 113, 113, 0.40)',
    borderRadius: 10, padding: '14px 16px', margin: '16px 0',
};
const loadErrorBtn = {
    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 14px',
    borderRadius: 10, border: '1px solid rgba(129,140,248,0.4)', background: 'rgba(99,102,241,0.16)',
    color: '#c7d2fe', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer',
};
const tableWrap = { overflowX: 'auto', marginTop: 12 };
const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13, color: '#e2e8f0' };
const thStyle = {
    textAlign: 'left', padding: '10px 12px', borderBottom: '2px solid rgba(148, 163, 184, 0.18)',
    color: '#94a3b8', fontWeight: 600, fontSize: 12, textTransform: 'uppercase',
    background: 'rgba(99, 102, 241, 0.10)',
};
const tdStyle = { padding: '10px 12px', borderBottom: '1px solid rgba(148, 163, 184, 0.10)', color: '#cbd5e1' };
const metricCard = {
    background: 'rgba(15, 23, 42, 0.55)', borderRadius: 10, padding: 16,
    border: '1px solid rgba(148, 163, 184, 0.14)', textAlign: 'center',
    color: '#f1f5f9',
};
