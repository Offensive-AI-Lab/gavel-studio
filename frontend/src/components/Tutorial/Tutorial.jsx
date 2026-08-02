// Tutorial — first-login onboarding modal.
//
// Shape: 5 slides, each with a CSS/SVG mini-visualization of one core
// concept. Auto-fires on the user's first /workspace mount via the
// TutorialContext; the sidebar's "Tutorial" item replays it on demand.
//
// Why CSS/SVG instead of product screenshots: every screenshot becomes
// a maintenance burden the moment we move a button or rename a label.
// Abstract illustrations capture the SHAPE of each concept and don't
// drift with the UI.
//
// On finish or "I'll explore on my own", we PUT /user/tutorial-seen
// and update the cached localStorage user blob in lockstep so the auto-
// fire doesn't re-trigger after a hard refresh.

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    FiShield, FiBox, FiPenTool, FiLayers, FiArrowRight,
    FiBookOpen, FiZap, FiEdit3, FiCpu, FiTarget, FiCheckCircle,
    FiActivity, FiUploadCloud,
} from 'react-icons/fi';
import { useTutorial } from '../../contexts/TutorialContext';
import { markTutorialSeen } from '../../api';
import './Tutorial.css';

// --- Mini-visualizations ----------------------------------------------------

const VizWelcome = () => (
    <div className="viz-welcome">
        <div className="viz-welcome-orbit o1" />
        <div className="viz-welcome-orbit o2" />
        <div className="viz-welcome-orbit o3" />
        <div className="viz-welcome-core"><FiShield size={28} /></div>
    </div>
);

const VizHierarchy = () => (
    <div className="viz-hierarchy">
        <div className="viz-h-row">
            <span className="viz-box viz-box-tone-blue"><FiBox /> Model</span>
        </div>
        <span className="viz-h-arrow">▼</span>
        <div className="viz-h-row">
            <span className="viz-box viz-box-tone-indigo"><FiCpu /> Rule Set</span>
        </div>
        <span className="viz-h-arrow">▼</span>
        <div className="viz-h-row">
            <span className="viz-box viz-box-tone-violet"><FiPenTool /> Rule</span>
            <span className="viz-h-arrow">→</span>
            <div className="viz-ce-cluster">
                <span className="viz-ce-chip" />
                <span className="viz-ce-chip" />
                <span className="viz-ce-chip" />
            </div>
        </div>
    </div>
);

const VizSources = () => (
    <div className="viz-sources">
        <div className="viz-source-card">
            <span className="viz-source-icon tone-blue"><FiBookOpen size={18} /></span>
            <span className="viz-source-label">Library</span>
            <span className="viz-source-sub">Browse + bookmark vetted rules &amp; CEs</span>
        </div>
        <div className="viz-source-card">
            <span className="viz-source-icon tone-violet"><FiZap size={18} /></span>
            <span className="viz-source-label">AI</span>
            <span className="viz-source-sub">Describe a scenario, get a rule</span>
        </div>
        <div className="viz-source-card">
            <span className="viz-source-icon tone-emerald"><FiPenTool size={18} /></span>
            <span className="viz-source-label">Manual</span>
            <span className="viz-source-sub">Build from your bookmarked CEs</span>
        </div>
    </div>
);

const VizPipeline = () => (
    <div className="viz-pipeline">
        <div className="viz-step">
            <span className="viz-step-icon"><FiEdit3 size={18} /></span>
            <span className="viz-step-label">Edit</span>
        </div>
        <span className="viz-pipeline-arrow">→</span>
        <div className="viz-step">
            <span className="viz-step-icon"><FiCpu size={18} /></span>
            <span className="viz-step-label">Train</span>
        </div>
        <span className="viz-pipeline-arrow">→</span>
        <div className="viz-step">
            <span className="viz-step-icon"><FiTarget size={18} /></span>
            <span className="viz-step-label">Calibrate</span>
        </div>
        <span className="viz-pipeline-arrow">→</span>
        <div className="viz-step">
            <span className="viz-step-icon"><FiCheckCircle size={18} /></span>
            <span className="viz-step-label">Evaluate</span>
        </div>
    </div>
);

const VizShip = () => (
    <div className="viz-ship">
        <div className="viz-ship-panel">
            <span className="viz-ship-title">Real-time</span>
            <div className="viz-conv-line"><span className="viz-conv-dot" />user message…</div>
            <div className="viz-conv-line"><span className="viz-conv-dot active" />assistant flagged</div>
            <div className="viz-conv-line"><span className="viz-conv-dot" />user message…</div>
            <div className="viz-conv-line"><span className="viz-conv-dot active" />CE activated</div>
        </div>
        <span className="viz-ship-divider">·</span>
        <div className="viz-ship-publish">
            <FiUploadCloud size={22} />
            Contribute via PR
        </div>
    </div>
);


// --- Slide content ----------------------------------------------------------

// Each slide carries: kicker (the small chip), title, body copy, and a
// mini-viz component. The final slide swaps body for two CTA links.
const SLIDES = [
    {
        kicker: 'Welcome',
        title: 'Build AI safety rule sets tailored to your risks.',
        body: (
            <>
                Generic rule sets are too broad — Gavel lets you define
                exactly what <strong>"bad"</strong> means for your product. You
                describe behaviors in natural language, the platform turns
                them into trained rule sets you can ship.
            </>
        ),
        viz: <VizWelcome />,
    },
    {
        kicker: 'The vocabulary',
        title: 'Four moving parts.',
        body: (
            <>
                A <strong>Model</strong> is the LLM you want to protect. A{' '}
                <strong>Rule Set</strong> is your safety detector trained
                on top of it. <strong>Rules</strong> make up the rule set —
                combinations of <strong>Cognitive Elements (CEs)</strong>,
                atomic concepts the rule set learns to recognize.
            </>
        ),
        viz: <VizHierarchy />,
    },
    {
        kicker: 'Where rules come from',
        title: 'Three ways in.',
        body: (
            <>
                <strong>Browse the public library</strong> for vetted rules
                and CEs from other users — bookmark to save them.{' '}
                <strong>Generate via AI</strong>: describe a scenario and let
                it propose a rule. Or <strong>build manually</strong> from
                your bookmarked CEs. Anything you create lands in{' '}
                <strong>My Drafts</strong> — local to this Studio.
            </>
        ),
        viz: <VizSources />,
    },
    {
        kicker: 'The training cycle',
        title: 'Train, calibrate, evaluate.',
        body: (
            <>
                Once you have rules, the rule set needs to learn from them.{' '}
                <strong>Generate test sets</strong> for ground truth,{' '}
                <strong>calibrate</strong> per-CE thresholds, then{' '}
                <strong>evaluate</strong> to see precision and recall. Edit
                and re-train until the numbers look right.
            </>
        ),
        viz: <VizPipeline />,
    },
    {
        kicker: 'You\'re ready',
        title: 'Two ways to ship.',
        body: (
            <>
                <strong>Real-time monitoring</strong> runs your trained
                rule set on live conversations and shows you which CEs
                activate. <strong>Contribute your rules</strong> to the shared
                library via a gavel-rules pull request so other users can
                reuse them. Then start the next scenario.
            </>
        ),
        viz: <VizShip />,
        // Final slide — show CTAs in addition to the body.
        ctas: true,
    },
];


// --- Page-mode component ----------------------------------------------------

// Renders the per-page contextual help: a title + summary + N
// sections of state-derived bullet points, plus optional CTA links.
// This is what the sidebar Tutorial button shows on most pages; the
// 5-slide welcome only fires automatically on first /workspace mount.
const PageModeTutorial = ({ content, onClose, navigate }) => {
    return (
        <div
            className="tutorial-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="tutorial-title"
            onClick={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div className="tutorial-card">
                <div className="tutorial-slide">
                    <span className="tutorial-kicker">
                        Help · This page
                    </span>
                    <h2 id="tutorial-title" className="tutorial-title">
                        {content.title}
                    </h2>
                    {content.summary && (
                        <p className="tutorial-body">{content.summary}</p>
                    )}

                    {(content.sections || []).map((section, i) => (
                        <div key={i} className="tutorial-section">
                            <div className="tutorial-section-heading">{section.heading}</div>
                            <ul className="tutorial-section-list">
                                {(section.bullets || []).map((b, bi) => (
                                    <li key={bi}>{b}</li>
                                ))}
                            </ul>
                        </div>
                    ))}

                    {(content.ctas || []).length > 0 && (
                        <div className="tutorial-ctas">
                            {content.ctas.map((cta, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    className={`tutorial-cta ${cta.primary ? 'tutorial-cta-primary' : ''}`}
                                    onClick={() => {
                                        onClose();
                                        if (cta.to) navigate(cta.to);
                                        if (typeof cta.onClick === 'function') cta.onClick();
                                    }}
                                >
                                    {cta.label}
                                    <FiArrowRight size={18} />
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="tutorial-footer">
                    <div /> {/* spacer for layout symmetry */}
                    <div className="tutorial-actions">
                        <button
                            type="button"
                            className="tutorial-btn tutorial-btn-primary"
                            onClick={onClose}
                        >
                            Got it
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};


// --- Component --------------------------------------------------------------

const Tutorial = () => {
    const { open, mode, pageContent, dismiss } = useTutorial();
    const navigate = useNavigate();
    const [index, setIndex] = useState(0);

    // Reset to slide 1 every time the modal re-opens, so a user who
    // closed at slide 4 doesn't pick up at slide 4 next time.
    useEffect(() => {
        if (open) setIndex(0);
    }, [open]);

    if (!open) return null;

    // Page mode: a page has registered content via useTutorialContent.
    // Render that instead of the welcome slides — single page, no
    // slide navigation, just sections of state-derived guidance.
    //
    // mode='welcome' forces the 5-slide overview even if a page has
    // registered content (the first-login auto-fire path uses this).
    // mode='auto' (sidebar Tutorial button) prefers page content when
    // present, falls back to welcome.
    const showPage = mode !== 'welcome' && pageContent;
    if (showPage) {
        return (
            <PageModeTutorial
                content={pageContent}
                onClose={dismiss}
                navigate={navigate}
            />
        );
    }

    const slide = SLIDES[index];
    const isFirst = index === 0;
    const isLast = index === SLIDES.length - 1;

    // Persist + close. Called by both the "Done" button on the final
    // slide and the "I'll explore on my own" skip button. Idempotent
    // server-side; the localStorage update keeps the auto-fire silent
    // on subsequent /workspace mounts.
    const finish = async (navigateTo = null) => {
        try {
            await markTutorialSeen();
        } catch {
            // Best-effort — even if the PUT fails, dismiss the modal
            // for this session so the user isn't stuck. The next /me
            // poll will sync the flag.
        }
        try {
            const stored = JSON.parse(sessionStorage.getItem('user') || 'null');
            if (stored && !stored.tutorial_seen) {
                stored.tutorial_seen = true;
                sessionStorage.setItem('user', JSON.stringify(stored));
            }
        } catch {
            // localStorage corrupted? Nothing to do — modal still closes.
        }
        dismiss();
        if (navigateTo) navigate(navigateTo);
    };

    return (
        <div
            className="tutorial-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="tutorial-title"
            onClick={(e) => {
                // Click on backdrop (not the card itself) just dismisses
                // for this session — does NOT mark seen, so the user
                // gets the tutorial again next time. Treats backdrop-
                // dismiss as "not now" rather than "I'm done."
                if (e.target === e.currentTarget) dismiss();
            }}
        >
            <div className="tutorial-card">
                <div className="tutorial-slide">
                    <span className="tutorial-kicker">
                        {slide.kicker}
                    </span>
                    <h2 id="tutorial-title" className="tutorial-title">
                        {slide.title}
                    </h2>
                    <p className="tutorial-body">{slide.body}</p>

                    <div className="tutorial-viz-wrap">{slide.viz}</div>

                    {slide.ctas && (
                        <div className="tutorial-ctas">
                            <button
                                type="button"
                                className="tutorial-cta tutorial-cta-primary"
                                onClick={() => finish('/browse')}
                            >
                                Browse the library
                                <FiArrowRight size={18} />
                            </button>
                            <button
                                type="button"
                                className="tutorial-cta"
                                onClick={() => finish('/workspace')}
                            >
                                Open my workspace
                                <FiArrowRight size={18} />
                            </button>
                        </div>
                    )}
                </div>

                <div className="tutorial-footer">
                    <div className="tutorial-progress" aria-label={`Slide ${index + 1} of ${SLIDES.length}`}>
                        {SLIDES.map((_, i) => (
                            <span
                                key={i}
                                className={`tutorial-dot ${i === index ? 'active' : ''}`}
                            />
                        ))}
                    </div>
                    <div className="tutorial-actions">
                        {!isLast && (
                            <button
                                type="button"
                                className="tutorial-btn tutorial-btn-skip"
                                onClick={() => finish()}
                            >
                                I'll explore on my own
                            </button>
                        )}
                        {!isFirst && (
                            <button
                                type="button"
                                className="tutorial-btn tutorial-btn-ghost"
                                onClick={() => setIndex(index - 1)}
                            >
                                Back
                            </button>
                        )}
                        {!isLast ? (
                            <button
                                type="button"
                                className="tutorial-btn tutorial-btn-primary"
                                onClick={() => setIndex(index + 1)}
                            >
                                Next
                            </button>
                        ) : (
                            <button
                                type="button"
                                className="tutorial-btn tutorial-btn-primary"
                                onClick={() => finish()}
                            >
                                Got it
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Tutorial;
