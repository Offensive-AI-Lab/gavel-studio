// Step 2A — Rule Generation.
//
// Reads scenario.description from step 1's data, calls
// /ai/generate-pipeline (gpt-5.2 thinking model), shows the rule +
// new CEs the model proposed. Accepting links rule_id
// onto the pipeline_run row so downstream steps can find them.
import { useState, useMemo } from 'react';
import { FiCpu, FiCheckCircle, FiAlertTriangle, FiPlay } from 'react-icons/fi';
import { generateGavelPipeline, updatePipelineLinks, discardPipelineResources } from '../../api';
import InlineHelp from '../../components/InlineHelp/InlineHelp';
import { step2aRuleGeneration } from '../../components/InlineHelp/instructorHelp';
import {
    getStepState, startStep, completeStep, errorStep,
    card, primaryBtn, secondaryBtn, errorBanner, successBanner, muted,
} from './wizardShared';


// Same pattern as RulesManager + every other page that needs user_id —
// localStorage is the canonical session store (auth middleware reads
// the token from it; the row's just the user-shaped JSON it returned).
function readUser() {
    try {
        return JSON.parse(sessionStorage.getItem('user') || 'null');
    } catch {
        return null;
    }
}


export default function Step2ARule({ run, classifierId, onPatchStep, setRun, onAdvance }) {
    const user = readUser();
    const step1 = (run?.steps?.['1'] && run.steps['1'].data) || {};
    const state = getStepState(run, '2A');
    const data = state.data || {};

    const [generating, setGenerating] = useState(false);
    const [proposal, setProposal] = useState(data.proposal || null);
    const [error, setError] = useState(null);
    const [approving, setApproving] = useState(false);

    // Approve → hand off to the wizard's onFinish, which runs CE training,
    // CE calibration, the test/calibration set and finalization as ONE
    // background task and navigates to Drafts. Disabled after the first click.
    const handleApprove = async () => {
        setApproving(true);
        try {
            await onAdvance?.();
        } catch (err) {
            setApproving(false);
            setError(err?.message || 'Could not start the build.');
        }
    };

    const canGenerate = useMemo(
        () => Boolean(step1.description?.trim()),
        [step1.description],
    );

    const generate = async () => {
        if (!user?.user_id) {
            setError('Missing user context — try reloading.');
            return;
        }
        if (!canGenerate) {
            setError('Step 1 has no scenario description. Go back and finish step 1 first.');
            return;
        }
        setGenerating(true);
        setError(null);
        await startStep(onPatchStep, '2A', { ...data });

        try {
            // classifierId is null in Pipeline A (rule generation is
            // guardrail-agnostic). The backend accepts null and creates
            // the rule + CEs as library artifacts, unattached.
            const parsedCid = classifierId ? parseInt(classifierId) : null;
            const res = await generateGavelPipeline(step1.description, user.user_id, parsedCid);
            const r = res.data;
            setProposal(r);

            // Link the rule onto the run so 2B/2C/3* can find it.
            try {
                const linkRes = await updatePipelineLinks(run.run_id, {
                    ruleId: r.rule_id || null,
                });
                if (setRun && linkRes?.data) setRun(linkRes.data);
            } catch (linkErr) {
                console.warn('Link update failed:', linkErr);
            }

            await completeStep(onPatchStep, '2A', {
                proposal: r,
                rule_id: r.rule_id,
                ce_ids: (r.new_ces || []).map(ce => ce.ce_id),
                new_ces: r.new_ces || [],
            });
        } catch (err) {
            const msg = err?.response?.data?.detail || err.message;
            setError(msg);
            await errorStep(onPatchStep, '2A', msg, { ...data });
        } finally {
            setGenerating(false);
        }
    };

    const discard = async () => {
        if (!proposal) return;
        try {
            await discardPipelineResources(
                (proposal.new_ces || []).map(ce => ce.ce_id),
                proposal.rule_id || null,
            );
        } catch {}
        setProposal(null);
        await onPatchStep('2A', { status: 'pending', data: {} });
    };

    if (!canGenerate) {
        return (
            <div style={card}>
                <div style={errorBanner}>
                    <FiAlertTriangle /> Step 1 hasn't produced a scenario yet — finish the chat first.
                </div>
            </div>
        );
    }

    return (
        <div>
            <InlineHelp content={step2aRuleGeneration} />
            <div style={card}>
                <div style={{ marginBottom: 10 }}><strong>Scenario from step 1</strong></div>
                <div style={{ ...muted, fontSize: 13, fontStyle: 'italic' }}>
                    {step1.description}
                </div>
            </div>

            {!proposal && (
                <div style={card}>
                    <p style={{ marginTop: 0, color: '#cbd5e1' }}>
                        The reasoning model will analyze your scenario, propose a rule
                        (named CE groups + a firing condition over them), and identify
                        any new CEs that need to be created.
                    </p>
                    <button onClick={generate} disabled={generating} style={primaryBtn}>
                        {generating ? <><FiCpu /> Generating…</> : <><FiPlay /> Generate Rule</>}
                    </button>
                </div>
            )}

            {error && <div style={card}><div style={errorBanner}><FiAlertTriangle /> {error}</div></div>}

            {proposal && (
                <>
                    <div style={card}>
                        <div style={successBanner}>
                            <FiCheckCircle /> Rule proposal generated. Review it below, then <strong>Approve &amp; Build</strong> — the CE training, calibration and test sets all generate automatically in the background (watch the task tray, top-right).
                        </div>
                    </div>

                    <div style={card}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                            <h3 style={{ margin: 0 }}>{proposal.name}</h3>
                            {proposal.categories?.length > 0 && (
                                <div style={{ display: 'flex', gap: 6 }}>
                                    {proposal.categories.map(c => (
                                        <span key={c} style={categoryPill}>{c}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                        {proposal.description && (
                            <>
                                <div style={{ marginTop: 10, ...muted, fontSize: 12 }}>What this rule detects</div>
                                <div style={{ marginTop: 4, fontSize: 13, color: '#f1f5f9', lineHeight: 1.55 }}>
                                    {proposal.description}
                                </div>
                            </>
                        )}

                        {(proposal.groups && Object.keys(proposal.groups).length > 0) ? (
                            <>
                                <div style={{ marginTop: 12, ...muted, fontSize: 12 }}>CE Groups</div>
                                <div style={{ display: 'grid', gap: 6 }}>
                                    {Object.entries(proposal.groups).map(([gname, members]) => (
                                        <div key={gname}>
                                            <strong style={{ fontFamily: 'monospace' }}>{gname}:</strong>{' '}
                                            {(members || []).join(', ')}
                                        </div>
                                    ))}
                                </div>
                                <div style={{ marginTop: 10, ...muted, fontSize: 12 }}>Firing condition</div>
                                <pre style={preStyle}>{proposal.condition || proposal.predicate}</pre>
                            </>
                        ) : (
                            <>
                                {/* Fallback for older responses without groups */}
                                <div style={{ marginTop: 10, ...muted, fontSize: 12 }}>Predicate</div>
                                <pre style={preStyle}>{proposal.predicate}</pre>
                            </>
                        )}
                    </div>

                    {(proposal.new_ces || []).length > 0 && (
                        <div style={card}>
                            <strong>New Cognitive Elements ({proposal.new_ces.length})</strong>
                            <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
                                {proposal.new_ces.map(ce => (
                                    <div key={ce.ce_id} style={ceCardStyle}>
                                        <div style={{ fontWeight: 600 }}>{ce.ce_name}</div>
                                        <div style={{ ...muted, fontSize: 13, marginTop: 2 }}>{ce.definition}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                        <button onClick={discard} disabled={approving} style={secondaryBtn}>
                            Discard + Re-generate
                        </button>
                        <div style={{ flex: 1 }} />
                        <button onClick={handleApprove} disabled={approving} style={primaryBtn}>
                            {approving ? <><FiCpu /> Starting…</> : <><FiCheckCircle /> Approve &amp; Build</>}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

const categoryPill = {
    fontSize: 11, padding: '2px 8px',
    background: 'rgba(99, 102, 241, 0.18)',
    color: '#c7d2fe',
    borderRadius: 999,
    border: '1px solid rgba(129, 140, 248, 0.35)',
};

const preStyle = {
    margin: '4px 0 0',
    background: 'rgba(15, 23, 42, 0.55)',
    padding: 10,
    borderRadius: 8,
    fontSize: 13,
    color: '#e2e8f0',
    whiteSpace: 'pre-wrap',
    border: '1px solid rgba(148, 163, 184, 0.18)',
};

const ceCardStyle = {
    background: 'rgba(15, 23, 42, 0.55)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    borderRadius: 8, padding: 10,
};
