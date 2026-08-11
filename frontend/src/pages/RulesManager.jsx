import { useState, useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

// Components
import Layout from '../components/Layout/Layout';
import ReactiveButton from '../components/ReactiveButton/ReactiveButton';
import GlassModal from '../components/GlassModal/GlassModal';
import RuleCard from '../components/RuleCard/RuleCard';
import ExportClassifierModal from '../components/ExportClassifierModal/ExportClassifierModal';
import ComputeBadge from '../components/ComputeBadge/ComputeBadge';
import GlassSelect from '../components/GlassSelect/GlassSelect';
import AddModelModal from '../components/AddModelModal/AddModelModal';
import CreateChooserModal from '../components/CreateChooserModal/CreateChooserModal';
import Breadcrumb from '../components/Breadcrumb/Breadcrumb';
import GroupConditionEditor from '../components/GroupConditionEditor/GroupConditionEditor';

// Services & API
import {
    getClassifierRules,
    deleteRuleSetup,
    addRuleToClassifier,
    getClassifierDetails,
    getRuleBookmarks,
    getCEBookmarks,
    saveEditedRule,
    checkRuleDuplicate,
    trainClassifier,
    getComputeTargets,
    getTrainingStatus,
    cancelTraining,
    downloadClassifier,
    listLocalDrafts,
    getUserModels,
    attachModel,
    cloneClassifierToModel,
    updateModelLayers,
} from '../api';
import { useLibraryRefresh } from '../hooks/useLibraryRefresh';
import useLibrarySearch from '../hooks/useLibrarySearch';
import { useTutorialContent } from '../contexts/TutorialContext';
import { recordRecent } from '../utils/recents';
import { extractLogic, normalizeGroups, validateEditorState } from '../utils/ruleLogic';
import { wasTrainingCancelled } from '../utils/errorText';

// Icons & Utils
import { showAlertDialog, showConfirmDialog } from '../components/ConfirmDialog/confirmDialog';
import { FiPlus, FiGlobe, FiZap, FiRefreshCw, FiArrowLeft, FiInbox, FiDownload, FiUploadCloud, FiCheckCircle, FiSettings, FiBarChart2, FiRadio, FiAlertTriangle, FiCpu, FiCopy, FiBookmark, FiLayers, FiHome, FiShield, FiChevronRight, FiFileText, FiSearch, FiXCircle } from 'react-icons/fi';

import '../css/RulesManager.css';

// Translate a raw training failure (from training_phase_detail) into a clear,
// actionable message. Known causes get a plain-English explanation; anything
// else falls through to the raw text so nothing is hidden.
function friendlyTrainingError(detail) {
    if (!detail) return null;
    const d = String(detail);
    if (/chat[_\s-]?template/i.test(d)) {
        return "This model can't be trained: its tokenizer has no chat template, so the "
            + "conversations can't be formatted. Train this rule set on a chat / instruct model "
            + "(one whose tokenizer defines a chat template) instead.";
    }
    return d;
}

// How many library matches the "Add a Rule" picker asks for per request. The
// modal list is a short scrollable strip: it shows one page at a time and asks
// for the next one as the user scrolls towards the bottom.
const ADD_SEARCH_PAGE_SIZE = 20;
// How deep the picker can page. The backend collects at most `candidate_limit`
// rows and paginates within them, so this is the ceiling on how far the list
// can be scrolled — with the hook's default of 80 the browse list would stop
// four pages in. 200 is the route's maximum (candidate_limit: le=200).
const ADD_SEARCH_CANDIDATE_LIMIT = 200;
// How close to the bottom of the list (in px) counts as "scrolled to the end"
// and triggers the next page. A small cushion so the fetch starts just before
// the last row is reached instead of after it.
const ADD_SEARCH_SCROLL_SLACK = 80;

const RulesManager = () => {
    const { classifierId } = useParams();
    const navigate = useNavigate();
    
    // --- State Management ---
    const [rules, setRules] = useState([]);
    const [rulesLoadError, setRulesLoadError] = useState(false);   // getClassifierRules failed
    // An empty array means two different things — "not fetched yet" and
    // "fetched, genuinely nothing" — and the empty states used to render on the
    // bare length check, so a slow or hanging backend showed "No rules in this
    // rule set" and "No models yet" as if they were answers (#11). These flags
    // start TRUE so the very first paint is a loading state, never an empty one.
    const [rulesLoading, setRulesLoading] = useState(true);
    const [expandedRule, setExpandedRule] = useState(null);
    
    // Modal Config
    const [modalConfig, setModalConfig] = useState({ isOpen: false, type: null });
    // Guardrail bundle export modal (tier picker + publish-before-export).
    const [exportOpen, setExportOpen] = useState(false);
    // Multi-select: the set of rule_ids checked in the "Add a Rule" modal.
    const [selectedRuleIds, setSelectedRuleIds] = useState(() => new Set());

    const [ruleBookmarks, setRuleBookmarks] = useState([]);
    // The user's own unpublished draft rules. Surfaced in the "Add a Rule"
    // picker alongside bookmarks so freshly-built rules (which have no
    // public_id and therefore can't be bookmarked) can still be added to a
    // guardrail.
    const [ruleDrafts, setRuleDrafts] = useState([]);
    // rule_ids whose explanation is expanded in the "Add a Rule" picker.
    const [expandedAddDescIds, setExpandedAddDescIds] = useState(new Set());

    // "Add a Rule" picker search. Bookmarks and drafts alone meant a community
    // rule could only be added by leaving for the Community page, bookmarking
    // it and coming back; the picker searches the public library itself now.
    const [addSearchQuery, setAddSearchQuery] = useState('');
    // Stable identities so the search hook's deps don't churn on re-render.
    const addSearchAssetTypes = useMemo(() => ['rule'], []);
    const addSearchCategories = useMemo(() => [], []);
    const addModalOpen = modalConfig.isOpen && modalConfig.type === 'add_bookmarked_rule';
    // Which page of library matches has been asked for. Bumped by scrolling to
    // the bottom of the picker list.
    const [addSearchPage, setAddSearchPage] = useState(1);
    const {
        results: addSearchResults,
        totalResults: addSearchTotal,
        loading: addSearchLoading,
        error: addSearchError,
    } = useLibrarySearch({
        // Blanking the query while the modal is closed keeps a leftover search
        // from firing on an unrelated re-render.
        query: addModalOpen ? addSearchQuery : '',
        categories: addSearchCategories,
        page: addSearchPage,
        pageSize: ADD_SEARCH_PAGE_SIZE,
        assetTypes: addSearchAssetTypes,
        candidateLimit: ADD_SEARCH_CANDIDATE_LIMIT,
        allowEmptyQuery: false,
        // Open the picker and the library is already listed, so the user can
        // scroll from their bookmarks and drafts straight into it. Tied to the
        // modal being open: with it closed there is nothing to browse for.
        browseWhenEmpty: addModalOpen,
    });
    // useLibrarySearch REPLACES its results on every page, so the picker keeps
    // the pages it has already been given: { [pageNumber]: results }. Scrolling
    // to page 2 would otherwise swap the first 20 hits out from under the user —
    // and take any ticked row with them. Keyed by page rather than concatenated
    // so a response landing twice can only overwrite its own slot.
    const [addSearchPages, setAddSearchPages] = useState({});
    // Highest page claimed. Kept in a ref as well as state because a burst of
    // scroll events all run against the same render, and each of them has to see
    // the page the one before it already claimed.
    const addPageRef = useRef(1);
    // The response last filed away, by identity. Bumping the page re-runs the
    // effect below while the hook still holds the PREVIOUS page's results; this
    // is what tells the two apart.
    const addFiledResultsRef = useRef(null);

    // Drop every collected page and go back to page 1.
    const resetAddLibraryPaging = () => {
        addPageRef.current = 1;
        addFiledResultsRef.current = null;
        setAddSearchPage(1);
        setAddSearchPages({});
    };

    // A new query means new results, and reopening the modal starts a fresh
    // pick — neither may inherit the pages collected before it.
    useEffect(() => {
        resetAddLibraryPaging();
    }, [addSearchQuery, addModalOpen]);

    // A failed search leaves nothing trustworthy to page from: clear what was
    // collected and start again from page 1.
    useEffect(() => {
        if (addSearchError) resetAddLibraryPaging();
    }, [addSearchError]);

    // File each response under the page it was asked for.
    useEffect(() => {
        if (addSearchLoading || addSearchError) return;
        // Same results object as last time — the page was just bumped and its
        // response is still on the way.
        if (addFiledResultsRef.current === addSearchResults) return;
        addFiledResultsRef.current = addSearchResults;
        setAddSearchPages((prev) => ({ ...prev, [addSearchPage]: addSearchResults || [] }));
    }, [addSearchResults, addSearchLoading, addSearchError, addSearchPage]);

    // Every page collected so far, in the order they were asked for, deduped by
    // id — pages can overlap when the library shifts between requests.
    const addLibraryRows = useMemo(() => {
        const seen = new Set();
        const rows = [];
        Object.keys(addSearchPages).map(Number).sort((a, b) => a - b).forEach((p) => {
            (addSearchPages[p] || []).forEach((item) => {
                const id = item.id ?? item.rule_id;
                if (id == null || seen.has(String(id))) return;
                seen.add(String(id));
                rows.push(item);
            });
        });
        return rows;
    }, [addSearchPages]);

    // Whether the library has matches beyond the ones already collected. Counted
    // on the raw hits, duplicates included: counting the deduped rows would never
    // reach the total when pages overlap, and the picker would ask for page after
    // page forever. An empty page also means the end, whatever the total says.
    const addLibraryHasMore = useMemo(() => {
        const pages = Object.keys(addSearchPages).map(Number).sort((a, b) => a - b);
        if (pages.length === 0) return false;
        if ((addSearchPages[pages[pages.length - 1]] || []).length === 0) return false;
        const fetched = pages.reduce((n, p) => n + (addSearchPages[p] || []).length, 0);
        return fetched < (addSearchTotal || 0);
    }, [addSearchPages, addSearchTotal]);

    // A page has been asked for and its results are not in yet.
    const addPageInFlight = !addSearchError && addSearchPages[addSearchPage] === undefined;

    // Ask for the next page once the user reaches the bottom of the list. Held
    // back while a request is out, and once every match is on screen.
    const handleAddListScroll = (e) => {
        const el = e.currentTarget;
        if (!el || !addLibraryHasMore) return;
        if (addSearchLoading || addPageInFlight) return;
        // A bump claimed earlier in this same burst of scroll events.
        if (addPageRef.current !== addSearchPage) return;
        const distanceToEnd = el.scrollHeight - el.scrollTop - el.clientHeight;
        if (distanceToEnd > ADD_SEARCH_SCROLL_SLACK) return;
        addPageRef.current = addSearchPage + 1;
        setAddSearchPage(addPageRef.current);
    };

    const [sidebarContext, setSidebarContext] = useState({ modelName: 'Loading...', classifierName: 'Loading...' });
    // Seed from sessionStorage so navigating back to this page mid-training
    // shows the banner instantly instead of flickering empty for ~1-2s
    // while the status API runs its SSH cycle. The API response will
    // confirm/correct this seed on resolve.
    const _cachedStatus = sessionStorage.getItem(`trainStatus_${classifierId}`);
    const _cachedPhase = sessionStorage.getItem(`trainPhase_${classifierId}`);
    const _cachedDetail = sessionStorage.getItem(`trainDetail_${classifierId}`);
    const [trainingStatus, setTrainingStatus] = useState(_cachedStatus || null); // null | 'untrained' | 'training' | 'active' | 'error' | 'needs_retraining'
    // Snapshot from the moment the guardrail was last trained.
    //   * `trainedSetupIds`  — volatile local PKs of the rule_setup rows
    //     active at training time. Kept around for places that want a
    //     PK-based query (download, etc).
    //   * `trainedRuleNames` — durable identity used for drift detection.
    //     A rule deleted and re-added with the same name should NOT
    //     register as drift, even though setup_id changes. Comparing by
    //     setup_id was the source of two reproducible bugs (re-add cycle,
    //     post-retrain banner staleness when setup_ids churn).
    const [trainedSetupIds, setTrainedSetupIds] = useState([]);
    const [trainedRuleNames, setTrainedRuleNames] = useState([]);
    // Live phase signal from the trainer's progress callback (e.g.
    // "Extracting embeddings", "Training RNN" with a per-epoch detail).
    // Only meaningful while trainingStatus === 'training'; the backend
    // forces these to null when status flips back, so a stale banner
    // can't linger past completion.
    const [trainingPhase, setTrainingPhase] = useState(_cachedPhase || null);
    const [trainingPhaseDetail, setTrainingPhaseDetail] = useState(_cachedDetail || null);
    // The row's `training_log`. Only read for one thing: a stopped run says in
    // there whether it was cancelled by the user or lost/failed, and the two
    // get different banners.
    const [trainingLog, setTrainingLog] = useState(null);
    // The post-training chain — "Calibrating" then "Evaluating" — which runs
    // AFTER status flips to 'active'. Tracked separately from trainingPhase
    // because the training banner is gated on trainingStatus === 'training';
    // without this the page went silent while calibration ran for minutes.
    const [chainPhase, setChainPhase] = useState(null);
    const [chainPhaseDetail, setChainPhaseDetail] = useState(null);
    // Consecutive post-training polls that reported no chain stage.
    const postTrainIdleRef = useRef(0);
    // True while we're awaiting the trainClassifier API call — the remote
    // submission (payload staging + job submit) can take 10-20s, and without
    // this flag the UI sits silent until the "Training started" dialog pops.
    const [submitting, setSubmitting] = useState(false);
    // True from the moment the user asks to cancel until the run's new state is
    // on screen — locks the Cancel button so the run can't be cancelled twice.
    // Mirrored in a ref because the click handler has to see the lock before
    // React re-renders (two clicks in one tick).
    const [cancelling, setCancelling] = useState(false);
    const cancellingRef = useRef(false);
    // Model-last flow: a guardrail may have no model until the user picks one
    // (at train time). `models` backs both the attach picker and the
    // "Apply to another model" clone picker.
    const [models, setModels] = useState([]);
    const [modelsLoading, setModelsLoading] = useState(true);   // see rulesLoading
    const [attachOpen, setAttachOpen] = useState(false);
    const [attachTargetModelId, setAttachTargetModelId] = useState('');
    // Per-model LLM layer editor inside the Choose-Model modal.
    const [attachLayerStart, setAttachLayerStart] = useState(null);
    const [attachLayerEnd, setAttachLayerEnd] = useState(null);
    const [layerSaving, setLayerSaving] = useState(false);
    const [attachBusy, setAttachBusy] = useState(false);
    const [cloneOpen, setCloneOpen] = useState(false);
    const [cloneTargetModelId, setCloneTargetModelId] = useState('');
    const [cloneBusy, setCloneBusy] = useState(false);
    const [addModelOpen, setAddModelOpen] = useState(false);   // inline "add a model" (no Models page)
    const [createOpen, setCreateOpen] = useState(false);       // "Create a New Rule" → shared chooser
    // In-place rule-logic editor (groups + condition) for one setup row.
    const [logicEdit, setLogicEdit] = useState(null);          // { rule, groupList, condition, pool } | null
    const [logicNewName, setLogicNewName] = useState('');
    const [logicSaving, setLogicSaving] = useState(false);
    const [machineOpen, setMachineOpen] = useState(false);     // "choose a machine" picker (>1 target)
    const [machineTargets, setMachineTargets] = useState([]);
    const user = JSON.parse(sessionStorage.getItem('user'));

    // --- Init ---
    useEffect(() => {
        if (user && classifierId) {
            // Re-seed training state from the per-guardrail cache on EVERY
            // classifierId change. Navigating between guardrails via the sidebar
            // reuses this component (no remount), so the useState initializer
            // doesn't re-run — without this, the PREVIOUS guardrail's status
            // (e.g. "untrained") lingers and shows a clickable "Train Guardrail"
            // until the slow status API resolves. Seed = the cached 'training'
            // (instant banner) or null (shows "Checking status…" until confirmed).
            setTrainingStatus(sessionStorage.getItem(`trainStatus_${classifierId}`) || null);
            setTrainingPhase(sessionStorage.getItem(`trainPhase_${classifierId}`) || null);
            setTrainingPhaseDetail(sessionStorage.getItem(`trainDetail_${classifierId}`) || null);
            refreshData();
            fetchSidebarContext();
            fetchBookmarks();
            fetchTrainingStatus();
            fetchModels();
        }
    }, [classifierId]);

    // Auto-refresh on any library mutation: someone adding/removing a
    // rule from this guardrail in another tab, an AI pipeline finishing
    // and dropping a draft, the user toggling bookmarks elsewhere, an
    // library sync pulling fresh data, etc. — all keep this page current
    // without a manual reload.
    useLibraryRefresh(() => {
        if (user && classifierId) {
            refreshData();
            fetchBookmarks();
        }
    });

    // Poll while training. When the run finishes (status flips out of
    // 'training'), re-fetch the guardrail details so trainedSetupIds
    // gets the fresh snapshot — without that refetch, the drift banner
    // can't detect rules the user added DURING training, because the
    // local trainedSetupIds stays stuck at the page-mount value.
    useEffect(() => {
        // Keep polling past the end of training while the calibration →
        // evaluation chain is still running, so its progress keeps ticking.
        if (trainingStatus !== 'training' && !chainPhase) return;
        const interval = setInterval(async () => {
            try {
                const res = await getTrainingStatus(classifierId);
                const newStatus = res.data.status;
                if (newStatus !== trainingStatus) {
                    setTrainingStatus(newStatus);
                }
                setChainPhase(res.data.post_training_phase || null);
                setChainPhaseDetail(res.data.post_training_phase_detail || null);
                if (res.data.post_training_phase) postTrainIdleRef.current = 0;
                // Pick up the live phase + detail on every poll so the
                // banner ticks forward as the trainer crosses stage
                // boundaries. Backend forces these to null off-status,
                // so we just mirror what the route returns.
                setTrainingPhase(res.data.training_phase || null);
                setTrainingPhaseDetail(res.data.training_phase_detail || null);
                setTrainingLog(res.data.training_log ?? null);
                if (newStatus === 'training') {
                    sessionStorage.setItem(`trainStatus_${classifierId}`, newStatus);
                    sessionStorage.setItem(`trainPhase_${classifierId}`, res.data.training_phase || '');
                    sessionStorage.setItem(`trainDetail_${classifierId}`, res.data.training_phase_detail || '');
                } else {
                    sessionStorage.removeItem(`trainStatus_${classifierId}`);
                    sessionStorage.removeItem(`trainPhase_${classifierId}`);
                    sessionStorage.removeItem(`trainDetail_${classifierId}`);
                }
                // The chain's first marker row lands a moment after training
                // flips to 'active', so one empty poll doesn't mean "done" —
                // stopping there would miss calibration entirely. Give it two.
                if (!res.data.is_training && !res.data.post_training_phase) {
                    postTrainIdleRef.current += 1;
                    if (postTrainIdleRef.current < 2) return;
                    clearInterval(interval);
                    // Training just completed (success or error). Refresh
                    // the guardrail-details payload so the snapshot we
                    // compare drift against reflects what was actually
                    // trained on, not what the user had selected on
                    // mount.
                    fetchSidebarContext();
                }
            } catch {
                return;
            }
        }, 5000);
        return () => clearInterval(interval);
        // chainPhase is a dep so the poll RESTARTS if training ends and the
        // chain picks up; it only toggles at stage boundaries, so this doesn't
        // churn the interval.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [trainingStatus, chainPhase]);

    const fetchSidebarContext = async () => {
        try {
            const res = await getClassifierDetails(classifierId);
            setSidebarContext({ modelName: res.data.model_name, classifierName: res.data.name, modelId: res.data.model_id });
            recordRecent('guardrail', { id: classifierId, name: res.data.name, path: `/classifiers/${classifierId}/rules` });
            // The same payload also carries the post-training snapshot we
            // need to detect rule drift — store it now to avoid a second
            // round trip from the drift banner.
            setTrainedSetupIds(Array.isArray(res.data.trained_rule_setup_ids) ? res.data.trained_rule_setup_ids : []);
            setTrainedRuleNames(Array.isArray(res.data.trained_rule_names) ? res.data.trained_rule_names : []);
        } catch { /* sidebar context is non-critical */ }
    };

    const fetchTrainingStatus = async () => {
        try {
            const res = await getTrainingStatus(classifierId);
            setTrainingStatus(res.data.status);
            setTrainingPhase(res.data.training_phase || null);
            setTrainingPhaseDetail(res.data.training_phase_detail || null);
            setTrainingLog(res.data.training_log ?? null);
            // Picked up on mount too, so landing on the page mid-calibration
            // shows the banner (and starts the poll) instead of looking idle.
            setChainPhase(res.data.post_training_phase || null);
            setChainPhaseDetail(res.data.post_training_phase_detail || null);
            // Cache so navigating away and back shows the banner instantly.
            if (res.data.status === 'training') {
                sessionStorage.setItem(`trainStatus_${classifierId}`, res.data.status);
                sessionStorage.setItem(`trainPhase_${classifierId}`, res.data.training_phase || '');
                sessionStorage.setItem(`trainDetail_${classifierId}`, res.data.training_phase_detail || '');
            } else {
                sessionStorage.removeItem(`trainStatus_${classifierId}`);
                sessionStorage.removeItem(`trainPhase_${classifierId}`);
                sessionStorage.removeItem(`trainDetail_${classifierId}`);
            }
        } catch { /* non-critical */ }
    };

    // Three-way state machine comparing the user's current rule selection
    // to the snapshot the guardrail was last trained against:
    //
    //   'aligned'  — the two sets are identical (or there's no snapshot yet
    //                and no current rules). Train button is "Up to Date" /
    //                "Train Guardrail" depending on whether the guardrail
    //                has ever been trained. No banner.
    //   'empty'    — the user has zero rules selected. Can't train without
    //                rules, regardless of what the guardrail was trained
    //                on. Train button disabled. Banner explains why.
    //   'drifted'  — current selection is non-empty AND differs from the
    //                trained snapshot. Train button becomes "Retrain
    //                Guardrail". Banner tells the user evaluation will
    //                keep using the snapshot until they retrain.
    //
    // The comparison is on rule NAMES, not setup_ids. setup_id is volatile —
    // deleting and re-adding a rule mints a new id even though the user
    // sees "the same rule". Names are durable. Set equality means reverting
    // to the trained selection (remove-then-readd, edit-then-revert) takes
    // the banner away — a simple "rules edited at all" flag would leave it
    // stuck.
    const policyState = useMemo(() => {
        const currentNames = Array.isArray(rules)
            ? rules.map(r => r.custom_name).filter(n => typeof n === 'string' && n.length > 0)
            : [];

        if (currentNames.length === 0) {
            return 'empty';
        }

        // No prior training → not "drifted", just the user's first selection.
        if (!Array.isArray(trainedRuleNames) || trainedRuleNames.length === 0) {
            return 'aligned';
        }

        const sameSize = currentNames.length === trainedRuleNames.length;
        if (!sameSize) return 'drifted';

        const trainedSet = new Set(trainedRuleNames);
        const isSubset = currentNames.every(n => trainedSet.has(n));
        return isSubset ? 'aligned' : 'drifted';
    }, [rules, trainedRuleNames]);

    // Per-page tutorial — adapts to the train-state machine and the
    // user's current rule set. Same vocabulary the train button uses
    // so the help reads as "what the buttons in front of me mean".
    const pageHelp = {
        title: 'Rule Set Logic Manager',
        summary: 'Add rules to this rule set and train it. Each rule groups its CEs (Cognitive Elements) into named groups and fires on a condition written over those groups. You pick the model the rule set runs on when you click Train — it then trains only on that one model.',
        sections: [
            {
                heading: 'Right now',
                bullets:
                    rulesLoading
                        ? ['Loading this rule set…']
                        : rules.length === 0
                        ? [
                            'No rules yet. Use "Add an Existing Rule" to drop a bookmarked public rule or one of your drafts into this rule set.',
                            'To author a new rule, use "Create a New Rule" — it opens the Create menu (Rule with AI, Build Rule from CEs, or a new CE); the finished rule lands in Your Library → Rules and you add it here.',
                        ]
                        : trainingStatus === 'training'
                            ? ['Training is running — the banner above shows the live phase. Don\'t close the tab; the run keeps going on the server even if you navigate away.']
                            : trainingStatus === 'untrained' || trainedRuleNames.length === 0
                                ? [`${rules.length} rule${rules.length === 1 ? '' : 's'} attached. Click Train (top right). If no model is attached yet, you'll pick one first — the rule set then trains on that model only.`]
                                : policyState === 'drifted'
                                    ? [`Your rule selection differs from what the rule set was trained on (${trainedRuleNames.length} rules). Evaluation and real-time use the OLD selection until you click Retrain.`]
                                    : [`Trained and aligned with ${rules.length} rule${rules.length === 1 ? '' : 's'}. Evaluate, Monitor, Download, Export, and "Apply to another model" are now available.`],
            },
            {
                heading: 'Working with a rule',
                bullets: [
                    'Click a rule\'s chevron to expand it — its explanation, firing logic (CE groups + condition), and the CEs it combines.',
                    'Open "Rule page" for the full details: every CE\'s definition and examples, plus the rule\'s test set.',
                    'The trash icon removes a rule from this rule set; the rule itself stays in the library.',
                ],
            },
            {
                heading: 'When you\'re trained',
                bullets: [
                    'Evaluate → measure precision/recall/F1 on each rule\'s test set.',
                    'Monitor → run the rule set on live conversations in real time.',
                    'Apply to another model → copy this rule set onto a second model (independent copy, retrained there).',
                    'Download → grab the raw trained model files as a .zip.',
                    'Export → package the rule set as a shareable bundle (model, optionally calibration + evaluation). Every rule and CE in the policy must already be in the public library — drafts can\'t be bundled (contribute them via a gavel-rules pull request first). Builds in the background — you can close the dialog and it keeps going. Export only shows when the rule set matches what the model was trained on; if you changed the rules, retrain first.',
                ],
            },
        ],
    };
    useTutorialContent(pageHelp);

    const fetchModels = async () => {
        try {
            const res = await getUserModels(user.user_id);
            setModels(res.data.models || []);
        } catch {
            setModels([]);
        } finally {
            setModelsLoading(false);
        }
    };

    // ---- In-place rule-logic editor (groups + condition) -----------------
    // Opens on a setup row; edits go to POST /rules/setup/{id}/save-edited
    // ({groups: {name: [ce_id]}, condition, new_name?}). The backend patches
    // the user's own draft in place, or forks under new_name when the source
    // rule is public / setup-only; its 400/409 detail is surfaced verbatim.
    const openLogicEditor = async (rule) => {
        const memberCes = (rule.active_ces || []).filter((c) => c && c.ce_id != null);
        // Pool = the rule's current CEs + the user's bookmarked CEs (so new
        // elements can be pulled into a group).
        let pool = memberCes.map((c) => ({ ce_id: c.ce_id, name: c.name }));
        try {
            const res = await getCEBookmarks(user.user_id);
            (res.data?.bookmarks || []).forEach((b) => {
                if (b.ce_id != null && !pool.some((p) => p.ce_id === b.ce_id)) {
                    pool.push({ ce_id: b.ce_id, name: b.name });
                }
            });
        } catch { /* bookmarks are optional here */ }

        const logic = extractLogic(rule);
        let groupList = Object.entries(normalizeGroups(logic.groups || {})).map(([name, members]) => ({
            name,
            ceIds: members.map((m) => m.ce_id).filter((id) => id != null),
        }));
        let condition = logic.condition;
        if (groupList.length === 0) {
            // Legacy row without groups — seed one group with the membership.
            groupList = [{ name: 'required', ceIds: memberCes.map((c) => c.ce_id) }];
            if (!condition) condition = 'all of required';
        }
        setLogicNewName('');
        setLogicEdit({ rule, groupList, condition, pool });
    };

    const saveLogicEdit = async () => {
        if (!logicEdit || logicSaving) return;
        const { rule, groupList, condition } = logicEdit;
        const err = validateEditorState(groupList, condition);
        if (err) return showAlertDialog({ title: 'Fix the logic first', message: err, variant: 'info' });
        const groups = {};
        groupList.forEach((g) => { groups[g.name.trim()] = g.ceIds; });

        setLogicSaving(true);
        try {
            // Dedup probe first: an identical structure elsewhere means the
            // user is about to retrain a copy — point them at the original.
            try {
                const dup = await checkRuleDuplicate({
                    groups,
                    condition: condition.trim(),
                    classifier_id: parseInt(classifierId, 10),
                    exclude_setup_id: rule.setup_id,
                });
                if (dup.data?.exists) {
                    setLogicSaving(false);
                    return showAlertDialog({
                        title: 'Duplicate rule',
                        message: `This logic is structurally identical to "${dup.data.name}". Reuse that rule instead, or change the groups/condition.`,
                        variant: 'warning',
                    });
                }
            } catch { /* dedup probe is best-effort; the save re-validates */ }

            await saveEditedRule(rule.setup_id, {
                groups,
                condition: condition.trim(),
                new_name: logicNewName.trim() || null,
            });
            setLogicEdit(null);
            refreshData();
        } catch (e) {
            showAlertDialog({
                title: 'Could not save',
                message: e.response?.data?.detail || e.message || 'Failed to save the edited logic.',
                variant: 'error',
            });
        } finally {
            setLogicSaving(false);
        }
    };

    // Train entry point. Model choice is now a SEPARATE step (the "Choose
    // Model" button) and the Train button is disabled until a model is
    // attached — so by the time Train is clickable a model is guaranteed.
    const handleTrain = async () => {
        if (submitting) return;
        if (!sidebarContext.modelId) return;   // guarded: Train is disabled without a model
        // If more than one compute target is configured (e.g. local + remote GPU /
        // remote), let the user pick the machine first; otherwise train directly.
        let targets = [];
        try {
            const res = await getComputeTargets('training');
            targets = res.data?.targets || [];
        } catch { /* fall back to auto-resolution on the backend */ }
        if (targets.length > 1) {
            setMachineTargets(targets);
            setMachineOpen(true);
            return;
        }
        // 0 or 1 target → no prompt; let the backend auto-resolve.
        doTrain(null);
    };

    const pickMachine = (name) => {
        setMachineOpen(false);
        doTrain(name);
    };

    // Choose / change the model from the Model configure card. Attaches it
    // immediately; allowed until the guardrail is trained.
    const handleSelectModel = async (modelId) => {
        if (!modelId) return;
        try {
            const res = await attachModel(classifierId, Number(modelId));
            const newId = res?.data?.classifier?.model_id ?? Number(modelId);
            const modelName = models.find(m => String(m.model_id) === String(modelId))?.name;
            setSidebarContext(prev => ({ ...prev, modelId: newId, modelName: modelName || prev.modelName }));
        } catch (err) {
            await showAlertDialog({ title: 'Error', message: err.response?.data?.detail || 'Failed to set this model.', variant: 'error' });
        }
    };

    // Seed the layer editor from a model (its saved range, or a sensible
    // default). When the model's layer count is unknown we cap at 100; any
    // chosen layer beyond the model's real count is ignored at train time.
    const LAYER_FALLBACK_MAX = 100;
    const seedLayers = (modelId) => {
        const m = models.find(x => String(x.model_id) === String(modelId));
        if (!m) { setAttachLayerStart(null); setAttachLayerEnd(null); return; }
        const max = m.num_layers || LAYER_FALLBACK_MAX;
        const sel = Array.isArray(m.selected_layers) && m.selected_layers.length === 2
            ? m.selected_layers
            : (m.num_layers ? [Math.round(max * 0.4), Math.round(max * 0.84)] : [13, 27]);
        setAttachLayerStart(sel[0]); setAttachLayerEnd(sel[1]);
    };

    // Re-seed the layer editor whenever the attached model changes.
    const seededRef = useRef(null);
    useEffect(() => {
        if (sidebarContext.modelId && models.length > 0 && seededRef.current !== sidebarContext.modelId) {
            seedLayers(sidebarContext.modelId);
            seededRef.current = sidebarContext.modelId;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sidebarContext.modelId, models]);

    const saveLayers = async () => {
        const m = models.find(x => String(x.model_id) === String(sidebarContext.modelId));
        if (!m) return;
        const max = m.num_layers || LAYER_FALLBACK_MAX;
        const start = Math.max(0, Math.min(attachLayerStart, max - 1));
        const end = Math.max(start + 1, Math.min(attachLayerEnd, max));
        setLayerSaving(true);
        try {
            await updateModelLayers(m.model_id, [start, end]);
            await fetchModels();
            await showAlertDialog({ title: 'Saved', message: `Layers set to ${start}–${end} for ${m.name}.`, variant: 'success' });
        } catch (err) {
            await showAlertDialog({ title: 'Error', message: err?.response?.data?.detail || 'Could not save layers.', variant: 'error' });
        } finally {
            setLayerSaving(false);
        }
    };

    // After a model is added inline, refresh the list and auto-select the new
    // (newest) model so it's configured immediately.
    const handleModelAdded = async () => {
        const res = await getUserModels(user.user_id);
        const list = res.data.models || [];
        setModels(list);
        const newest = list[0];   // get_user_models orders newest-first
        if (newest) {
            try {
                await attachModel(classifierId, newest.model_id);
                setSidebarContext(prev => ({ ...prev, modelId: newest.model_id, modelName: newest.name }));
            } catch { /* the user can still pick it from the dropdown */ }
        }
    };

    const handleClone = async () => {
        if (!cloneTargetModelId) return;
        setCloneBusy(true);
        try {
            const res = await cloneClassifierToModel(classifierId, Number(cloneTargetModelId));
            setCloneOpen(false);
            const newId = res?.data?.classifier?.classifier_id;
            const targetName = models.find(m => String(m.model_id) === String(cloneTargetModelId))?.name || 'the model';
            const goOpen = await showConfirmDialog({
                title: 'Copied to another model',
                message: `A copy was created on “${targetName}”. It's untrained — train it to run on that model. Open it now?`,
                confirmText: 'Open copy',
                cancelText: 'Stay here',
                variant: 'info',
            });
            if (goOpen && newId) navigate(`/classifiers/${newId}/rules`);
        } catch (err) {
            await showAlertDialog({
                title: 'Error',
                message: err.response?.data?.detail || 'Failed to copy this rule set.',
                variant: 'error',
            });
        } finally {
            setCloneBusy(false);
        }
    };

    const openClone = () => {
        if (models.length === 0) {
            showAlertDialog({
                title: 'No models yet',
                message: 'Add an LLM under Models first — "Apply to another model" copies this rule set onto one of your models.',
                variant: 'info',
            });
            return;
        }
        const firstOther = models.find(m => m.model_id !== sidebarContext.modelId) || models[0];
        setCloneTargetModelId(String(firstOther.model_id));
        setCloneOpen(true);
    };

    const doTrain = async (target = null) => {
        // Retrain is destructive: the trainer wipes the existing folder
        // before writing fresh artifacts (see trainer.py:289), so once
        // the user confirms there is no way back to the previous model.
        // Use showConfirmDialog so the dialog chrome matches the rest of
        // the app's polished modals (the same look as the "Training
        // started" success dialog) instead of raw Swal default styling.
        if (submitting) return;   // ignore double-clicks while a submit is in flight
        const isRetrain = trainedRuleNames.length > 0;
        // Lock the button + show the progress banner from the INSTANT of the
        // click (before the confirm dialog), so it can't be double-submitted and
        // the feedback is immediate. Rolled back below if they cancel the confirm.
        setSubmitting(true);
        const modelLabel = sidebarContext.modelName ? ` on <strong>${sidebarContext.modelName}</strong>` : '';
        const confirmed = await showConfirmDialog({
            title: isRetrain ? 'Retrain rule set?' : 'Train rule set?',
            messageHtml: isRetrain
                ? `Starting a new training run will <strong>delete the currently trained model</strong>. Once the new run starts, the previous trained file is gone — download it first if you want to keep a copy.<br/><br/><span style="font-size:0.85rem;color:#6b7280">Training runs in the background and may take several minutes.</span>`
                : `This will train the rule set${modelLabel} using the current rules and CE excitation datasets.<br/><br/><span style="font-size:0.85rem;color:#6b7280">Training runs in the background. Status will update automatically.</span>`,
            confirmText: isRetrain ? 'Yes, retrain (deletes current)' : 'Start Training',
            cancelText: 'Cancel',
            variant: isRetrain ? 'danger' : 'info',
        });
        if (!confirmed) { setSubmitting(false); return; }   // cancelled → unlock
        // Clear any error/phase left over from a PREVIOUS failed run so its
        // "Training failed" banner disappears the instant a new run starts —
        // otherwise it lingers next to the new "Submitting…" indicator until
        // the first poll. (The error banner gates on trainingPhaseDetail, so
        // nulling it hides it immediately without faking the status.)
        setTrainingPhase(null);
        setTrainingPhaseDetail(null);
        try {
            await trainClassifier(classifierId, target);
            setTrainingStatus('training');
            // No "Training started" success popup here. The progress banner
            // already shows the live training phase, and if the user has
            // navigated away by the time submission finishes (which takes
            // 10-20s while the worker allocates a GPU), a modal that
            // demands a click on whatever page they're on now is just
            // noise. Errors still get a popup because they're actionable.
        } catch (err) {
            const detail = err.response?.data?.detail || 'Failed to start training';
            await showAlertDialog({ title: 'Error', message: detail, variant: 'error' });
        } finally {
            setSubmitting(false);
        }
    };

    // Stop the run that's in progress. Confirm first — cancelling throws away
    // everything the run has done so far. A 409 says the run already ended
    // (finished on its own, or was cancelled elsewhere); that's not a failure,
    // so we skip the error dialog and just re-read the status. Re-reading is
    // also what stops the poll: it flips trainingStatus off 'training'.
    const handleCancelTraining = async () => {
        // Locked BEFORE the dialog opens — two clicks in the same tick both
        // read the pre-render value of `cancelling`, so only the ref can stop
        // the second one from opening a second dialog. Declining unlocks it.
        if (cancellingRef.current) return;
        cancellingRef.current = true;
        setCancelling(true);
        const confirmed = await showConfirmDialog({
            title: 'Cancel training?',
            messageHtml: 'The run stops now and the work it has done so far is lost.'
                + '<br/><br/><span style="font-size:0.85rem;color:#6b7280">This rule set will need training again before you can use it.</span>',
            confirmText: 'Cancel training',
            cancelText: 'Keep training',
            variant: 'danger',
        });
        if (!confirmed) {
            cancellingRef.current = false;
            setCancelling(false);
            return;
        }
        try {
            await cancelTraining(classifierId);
        } catch (err) {
            if (err?.response?.status !== 409) {
                cancellingRef.current = false;
                setCancelling(false);
                await showAlertDialog({
                    title: 'Error',
                    message: err?.response?.data?.detail || 'Could not cancel this training run.',
                    variant: 'error',
                });
                return;
            }
        }
        await fetchTrainingStatus();
        fetchSidebarContext();
        cancellingRef.current = false;
        setCancelling(false);
    };

    const refreshData = async () => {
        try {
            const res = await getClassifierRules(classifierId);
            setRules(Array.isArray(res.data.rules || res.data) ? (res.data.rules || res.data) : []);
            setRulesLoadError(false);
        } catch {
            setRulesLoadError(true);
            showAlertDialog({ title: 'Error', message: 'Failed to load rules', variant: 'error' });
        } finally {
            setRulesLoading(false);
        }
    };

    const fetchBookmarks = async () => {
        try {
            const [rulesRes, draftsRes] = await Promise.all([
                getRuleBookmarks(user.user_id),
                listLocalDrafts().catch(() => ({ data: { rules: [] } })),
            ]);
            setRuleBookmarks(rulesRes.data?.bookmarks || []);
            setRuleDrafts(draftsRes.data?.rules || []);
        } catch {
            setRuleBookmarks([]);
            setRuleDrafts([]);
        }
    };

    // --- Modal Logic ---
    const openAddFromBookmarks = () => {
        setSelectedRuleIds(new Set());
        setAddSearchQuery('');
        // Refetch so a rule the user just finished building (which became ready
        // in the background while they were on this page) shows up immediately,
        // instead of only after leaving and re-entering the page.
        fetchBookmarks();
        setModalConfig({ isOpen: true, type: 'add_bookmarked_rule' });
    };

    // Toggle one rule's checkbox in the multi-select modal.
    const toggleSelectedRule = (ruleId) => {
        const key = String(ruleId);
        setSelectedRuleIds((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key); else next.add(key);
            return next;
        });
    };

    const handleAddBookmarkedRule = async () => {
        const ids = Array.from(selectedRuleIds);
        if (ids.length === 0) {
            return showAlertDialog({ title: 'Select a rule', message: 'Choose at least one rule to add.', variant: 'info' });
        }
        // Add each selected rule; keep going if one fails and report the count.
        const results = await Promise.allSettled(ids.map((id) => addRuleToClassifier(classifierId, id)));
        const failed = results.filter((r) => r.status === 'rejected').length;
        setModalConfig({ isOpen: false, type: null });
        setSelectedRuleIds(new Set());
        refreshData();
        if (failed > 0) {
            showAlertDialog({
                title: 'Some rules not added',
                message: `${ids.length - failed} added, ${failed} failed.`,
                variant: 'warning',
            });
        }
    };

    const handleDeleteRule = async (setupId) => {
        const ok = await showConfirmDialog({
            title: 'Remove rule?',
            message: 'This will detach the rule from this rule set.',
            confirmText: 'Remove',
            cancelText: 'Cancel',
            variant: 'danger',
        });
        if (ok) {
            await deleteRuleSetup(setupId);
            setRules(prev => prev.filter(r => r.setup_id !== setupId));
        }
    };


    if (!user) return null;

    return (
        <Layout 
            currentModel={sidebarContext.modelName} 
            currentClassifier={sidebarContext.classifierName} 
            classifierId={classifierId}
        >
            
            {/* 1. Header (Uses classes from RulesManager.css) */}
            <header className="rules-header">
                <div>
                    <Breadcrumb items={[
                        { label: 'Hub', icon: FiHome, to: '/workspace' },
                        { label: 'Rule Sets', icon: FiShield, to: '/guardrails' },
                        { label: sidebarContext.classifierName, icon: FiFileText },
                    ]} />
                    <h1 style={{color: '#f1f5f9', margin: '5px 0 0 0', fontWeight: 700}}>{sidebarContext.classifierName}</h1>
                    {/* "Training on …" — where a Train run will execute (left side). */}
                    <div style={{ marginTop: '10px' }}>
                        <ComputeBadge workload="training" prefix="Training on" />
                    </div>
                </div>

                {/*
                  * Train button — snapshot-driven.
                  *
                  * The single source of truth is `trained_rule_setup_ids`
                  * (the snapshot taken at the start of the last training
                  * run) compared to the user's current rule selection.
                  * trainingStatus only contributes one thing: detecting
                  * "training is in flight right now" so the button shows
                  * a spinner.
                  *
                  * Decision tree:
                  *   training in progress       → "Training..." (disabled)
                  *   no rules selected          → "Train Guardrail" (grey, disabled)
                  *   never trained (no snapshot)→ "Train Guardrail" (blue)
                  *   snapshot == current        → "Up to Date" (green, disabled)
                  *   snapshot != current        → "Retrain Guardrail" (orange)
                  *
                  * needs_retraining and error from the legacy status field
                  * are no longer separately checked — the snapshot
                  * comparison fully covers "policy changed since last
                  * train" in either of those scenarios.
                  */}
                <div style={{ flexShrink: 0 }}>
                    <ReactiveButton
                        label={
                            // trainingStatus === null => status not loaded yet. The
                            // status API can round-trip to the worker (~1-2s), and when
                            // you switch guardrails via the sidebar there's no cached
                            // status to seed from — so show a neutral "Checking…"
                            // instead of flashing a clickable "Train Guardrail" on a
                            // guardrail that may already be training.
                            submitting ? 'Submitting...' :
                            trainingStatus === null ? 'Checking status…' :
                            trainingStatus === 'training' ? 'Training...' :
                            !sidebarContext.modelId ? 'Train Rule Set' :
                            policyState === 'empty' ? 'Train Rule Set' :
                            trainedRuleNames.length === 0 ? 'Train Rule Set' :
                            policyState === 'aligned' ? 'Up to Date' :
                            'Retrain Rule Set'
                        }
                        onClick={
                            submitting ? undefined :
                            trainingStatus === null ? undefined :
                            trainingStatus === 'training' ? undefined :
                            !sidebarContext.modelId ? undefined :
                            policyState === 'empty' ? undefined :
                            (trainedRuleNames.length > 0 && policyState === 'aligned') ? undefined :
                            handleTrain
                        }
                        Icon={
                            submitting ? FiRefreshCw :
                            trainingStatus === null ? FiRefreshCw :
                            trainingStatus === 'training' ? FiRefreshCw :
                            (trainedRuleNames.length > 0 && policyState === 'aligned') ? FiCheckCircle :
                            FiZap
                        }
                        disabled={
                            submitting ||
                            trainingStatus === null ||
                            trainingStatus === 'training' ||
                            !sidebarContext.modelId ||
                            policyState === 'empty' ||
                            (trainedRuleNames.length > 0 && policyState === 'aligned')
                        }
                        title={!sidebarContext.modelId ? 'Choose a model first' : undefined}
                        style={{
                            backgroundColor:
                                trainingStatus === null ? '#6b7280' :
                                !sidebarContext.modelId ? '#9ca3af' :
                                policyState === 'empty' ? '#9ca3af' :
                                (trainedRuleNames.length > 0 && policyState === 'aligned') ? '#059669' :
                                (trainedRuleNames.length > 0 && policyState === 'drifted') ? '#c2410c' :
                                '#2563eb',
                            opacity:
                                trainingStatus === null ? 0.7 :
                                trainingStatus === 'training' ? 0.7 :
                                !sidebarContext.modelId ? 0.7 :
                                policyState === 'empty' ? 0.7 :
                                (trainedRuleNames.length > 0 && policyState === 'aligned') ? 0.7 : 1,
                            cursor:
                                trainingStatus === null ? 'default' :
                                trainingStatus === 'training' ? 'default' :
                                !sidebarContext.modelId ? 'not-allowed' :
                                policyState === 'empty' ? 'not-allowed' :
                                (trainedRuleNames.length > 0 && policyState === 'aligned') ? 'default' : 'pointer',
                        }}
                    />
                </div>
            </header>

            {/* Rules-load-error banner — the guardrail's rule list couldn't be
              * fetched. Offer a retry plus explicit ways back so the page is
              * never a dead end. */}
            {rulesLoadError && (
                <div
                    role="alert"
                    style={{
                        background: 'rgba(239, 68, 68, 0.16)',
                        border: '1px solid rgba(248, 113, 113, 0.45)',
                        color: '#fecaca',
                        borderRadius: 8,
                        padding: '12px 16px',
                        margin: '12px 0',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 10,
                    }}
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 600 }}>
                        <FiAlertTriangle size={16} /> Couldn’t load this rule set’s rules.
                    </span>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <button onClick={refreshData} style={{ ...actionBtnStyle, background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', borderColor: 'rgba(96, 165, 250, 0.45)' }}>
                            <FiRefreshCw size={14} /> Try again
                        </button>
                        <button onClick={() => navigate('/guardrails')} style={{ ...actionBtnStyle, background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', borderColor: 'rgba(96, 165, 250, 0.45)' }}>
                            <FiShield size={14} /> Back to Rule Sets
                        </button>
                        <button onClick={() => navigate('/workspace')} style={{ ...actionBtnStyle, background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', borderColor: 'rgba(96, 165, 250, 0.45)' }}>
                            <FiHome size={14} /> Go to Hub
                        </button>
                    </div>
                </div>
            )}

            {/*
              * In-progress phase banner. Shown only while training is
              * actively running; the trainer's progress_callback writes
              * `training_phase` + `training_phase_detail` at every stage
              * boundary so this ticks forward through "Loading language
              * model" → "Extracting embeddings" → "Training RNN —
              * Epoch 3 of 10" without the user having to babysit logs.
              * Calm indigo palette (matches the active-tab pills) — this
              * is informational, not a warning like the policy banners.
              */}
            {(trainingStatus === 'training' || submitting || chainPhase) && (
                <div
                    role="status"
                    style={{
                        background: 'rgba(99, 102, 241, 0.18)',
                        border: '1px solid rgba(129, 140, 248, 0.45)',
                        color: '#c7d2fe',
                        borderRadius: 8,
                        padding: '12px 16px',
                        margin: '12px 0',
                        fontSize: '0.92rem',
                        lineHeight: 1.5,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                    }}
                >
                    <FiRefreshCw
                        size={16}
                        style={{ animation: 'spin 1.4s linear infinite', flexShrink: 0 }}
                    />
                    <div style={{ minWidth: 0, flex: 1 }}>
                        {/* Same banner, three sources: the submit round-trip, the
                            training run, and — once training is done — the
                            calibration → evaluation chain. Calibration and
                            evaluation keep their own tabs; this only says which
                            stage is currently running. */}
                        <strong>
                            {submitting
                                ? 'Looking for a GPU'
                                : (trainingStatus === 'training'
                                    ? (trainingPhase || 'Training in progress')
                                    : chainPhase)}
                        </strong>
                        <span style={{ marginLeft: 8, color: '#a5b4fc', fontWeight: 500 }}>
                            — {submitting
                                ? 'Uploading the job and requesting a GPU...'
                                : (trainingStatus === 'training'
                                    ? (trainingPhaseDetail || 'Status updating shortly')
                                    : (chainPhaseDetail || 'Status updating shortly'))}
                        </span>
                    </div>
                    {/* Only while the training run itself is in flight: the
                        submit round-trip has nothing to stop yet, and
                        calibration / evaluation aren't cancellable here. */}
                    {trainingStatus === 'training' && !submitting && (
                        <button
                            type="button"
                            onClick={handleCancelTraining}
                            disabled={cancelling}
                            style={cancelTrainingBtnStyle}
                            title="Stop this training run"
                        >
                            <FiXCircle size={13} /> {cancelling ? 'Cancelling...' : 'Cancel training'}
                        </button>
                    )}
                </div>
            )}

            {/*
              * Cancelled-run banner. A run the user stopped is written off like
              * any other unfinished run — status 'error', phase columns cleared,
              * reason in training_log — but it did not fail, so it gets its own
              * plain line instead of the failure banner below. (The two can't
              * both show: a cancellation clears training_phase_detail, which is
              * what the failure banner needs.)
              */}
            {trainingStatus === 'error' && !trainingPhaseDetail && !submitting
                && wasTrainingCancelled(trainingLog) && (
                <div role="status" style={cancelledBannerStyle}>
                    <FiXCircle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
                    <div style={{ minWidth: 0 }}>
                        <strong>Training was cancelled.</strong>{' '}
                        Train this rule set again whenever you're ready.
                    </div>
                </div>
            )}

            {/*
              * Training-failed banner. When a run errors (locally or on the
              * worker) the backend stores the failure in training_phase_detail
              * and flips status to 'error'. Surface it clearly — and translate
              * known causes (e.g. a model with no chat template) into something
              * the user can act on — instead of silently dropping the banner.
              */}
            {trainingStatus === 'error' && trainingPhaseDetail && !submitting && (
                <div
                    role="alert"
                    style={{
                        background: 'rgba(239, 68, 68, 0.16)',
                        border: '1px solid rgba(248, 113, 113, 0.45)',
                        color: '#fecaca',
                        borderRadius: 8,
                        padding: '12px 16px',
                        margin: '12px 0',
                        fontSize: '0.92rem',
                        lineHeight: 1.5,
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 12,
                    }}
                >
                    <FiAlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
                    <div style={{ minWidth: 0 }}>
                        <strong>Training failed.</strong>{' '}
                        {friendlyTrainingError(trainingPhaseDetail)}
                        {friendlyTrainingError(trainingPhaseDetail) !== trainingPhaseDetail && (
                            <div style={{ marginTop: 6, fontSize: '0.78rem', color: '#fca5a5', opacity: 0.85, wordBreak: 'break-word' }}>
                                Details: {trainingPhaseDetail}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/*
              * Two banners share the same look but cover different cases:
              *
              *   * 'empty'    — user has no rules selected. Can't train at
              *                  all; train button is grey/disabled. We
              *                  surface this regardless of trained state
              *                  because either way, training is blocked.
              *
              *   * 'drifted'  — current selection differs from what the
              *                  guardrail was trained on. Evaluation /
              *                  realtime guardrail keep using the trained
              *                  snapshot, not the user's current edits.
              *                  No counts ("N rules added") — set equality
              *                  decides whether a banner shows at all, and
              *                  diff counts encourage the user to compare
              *                  numbers when what they should compare is
              *                  "is this what I trained the model on".
              */}
            {policyState === 'empty' && (
                <div
                    role="status"
                    style={{
                        background: 'rgba(245, 158, 11, 0.18)',
                        border: '1px solid rgba(251, 191, 36, 0.45)',
                        color: '#fde68a',
                        borderRadius: 8,
                        padding: '12px 16px',
                        margin: '12px 0',
                        fontSize: '0.92rem',
                        lineHeight: 1.5,
                    }}
                >
                    <strong>No rules in this rule set.</strong>{' '}
                    {trainedRuleNames.length > 0
                        ? "You've removed every rule this rule set was trained on. Add at least one rule before you can retrain."
                        : "Add at least one rule before you can train this rule set."}
                </div>
            )}

            {policyState === 'drifted' && (
                <div
                    role="status"
                    style={{
                        background: 'rgba(249, 115, 22, 0.18)',
                        border: '1px solid rgba(251, 146, 60, 0.45)',
                        color: '#fed7aa',
                        borderRadius: 8,
                        padding: '12px 16px',
                        margin: '12px 0',
                        fontSize: '0.92rem',
                        lineHeight: 1.5,
                    }}
                >
                    <strong>Rule set differs from the trained model.</strong>{' '}
                    Your current rule selection isn't the same as what this rule set was last trained on. Evaluation and the realtime rule set will keep using the previously-trained rule set until you retrain.
                </div>
            )}

            {/* Action buttons bar.
              *
              * Rule generation moved off RulesManager in Phase 7 — it's
              * library-scoped now, lives on the Browse page. Test set
              * generation moved to the per-rule "Run Test Pipeline"
              * button on each RuleCard (Pipeline B). What's left here
              * are the guardrail-level read views: Evaluation results
              * and Realtime monitoring. */}
            <div style={actionBarStyle}>
                {/* "Apply to another model": copy this rule set onto a second
                  * model (independent, retrained there). Only meaningful once a
                  * model is attached — an unattached guardrail has nothing to
                  * apply "to another" yet. Works whether or not it's trained. */}
                {sidebarContext.modelId && (
                    <button
                        onClick={openClone}
                        style={{ ...actionBtnStyle, background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', borderColor: 'rgba(96, 165, 250, 0.45)' }}
                        title="Copy this rule set onto another model"
                    >
                        <FiCopy size={14} /> Apply to another model
                    </button>
                )}
                {(trainingStatus === 'active' || trainingStatus === 'needs_retraining') && (
                    <>
                        <button onClick={() => navigate(`/classifiers/${classifierId}/evaluate`)} style={{ ...actionBtnStyle, background: 'rgba(245, 158, 11, 0.18)', color: '#fcd34d', borderColor: 'rgba(251, 191, 36, 0.45)' }}>
                            <FiBarChart2 size={14} /> Evaluate
                        </button>
                        <button onClick={() => navigate(`/classifiers/${classifierId}/monitor`)} style={{ ...actionBtnStyle, background: 'rgba(139, 92, 246, 0.18)', color: '#ddd6fe', borderColor: 'rgba(167, 139, 250, 0.45)' }}>
                            <FiRadio size={14} /> Monitor
                        </button>
                        <button
                            onClick={() => downloadClassifier(classifierId, sidebarContext.classifierName).catch(() => showAlertDialog({ title: 'Error', message: 'Failed to download rule set.', variant: 'error' }))}
                            style={{ ...actionBtnStyle, background: 'rgba(16, 185, 129, 0.18)', color: '#6ee7b7', borderColor: 'rgba(16, 185, 129, 0.45)' }}
                            title={
                                trainingStatus === 'needs_retraining'
                                    ? 'Download the last trained model (you have unsaved rule set changes)'
                                    : 'Download the trained rule set as a .zip'
                            }
                        >
                            <FiDownload size={14} /> Download
                        </button>
                        {/* Export is only offered when the live policy still
                          * matches what the model was trained on — a drifted
                          * guardrail hides it until retrained. */}
                        {trainingStatus === 'active' && (
                            <button
                                onClick={() => setExportOpen(true)}
                                style={{ ...actionBtnStyle, background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', borderColor: 'rgba(96, 165, 250, 0.45)' }}
                                title="Export this rule set as a shareable bundle (model, calibration, evaluation)"
                            >
                                <FiUploadCloud size={14} /> Export
                            </button>
                        )}
                    </>
                )}
            </div>

            <ExportClassifierModal
                isOpen={exportOpen}
                classifierId={classifierId}
                classifierName={sidebarContext.classifierName}
                onClose={() => setExportOpen(false)}
            />


            <AddModelModal
                isOpen={addModelOpen}
                onClose={() => setAddModelOpen(false)}
                userId={user?.user_id}
                onAdded={handleModelAdded}
            />

            {/* Choose a machine — shown only when more than one compute target is
              * configured (local and/or remote GPU). */}
            <GlassModal isOpen={machineOpen} onClose={() => setMachineOpen(false)} title="Choose a machine to train on">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '0 0 4px' }}>
                        Where should this training run execute?
                    </p>
                    {machineTargets.map(t => (
                        <button
                            key={t.name}
                            onClick={() => pickMachine(t.name)}
                            style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', textAlign: 'left', padding: '12px 14px', borderRadius: '12px', border: '1px solid rgba(148, 163, 184, 0.22)', background: 'rgba(2, 6, 23, 0.45)', color: '#e2e8f0', cursor: 'pointer', fontSize: '0.92rem', fontWeight: 600 }}
                        >
                            <FiCpu size={16} />
                            <span>{t.label}{t.accelerator && t.accelerator !== 'REMOTE' ? ` · ${t.accelerator}` : ''}</span>
                        </button>
                    ))}
                </div>
            </GlassModal>

            {/* "Apply to another model": deep-copy this rule set onto another model. */}
            <GlassModal isOpen={cloneOpen} onClose={() => setCloneOpen(false)} title="Apply to another model">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                        Copy <strong style={{ color: '#e2e8f0' }}>{sidebarContext.classifierName}</strong>'s rule set onto another model.
                        The copy is independent and starts untrained — you'll train it for that model.
                    </p>
                    <label style={{ fontSize: '0.85rem', color: '#cbd5e1', marginBottom: '-6px' }}>Target model</label>
                    <GlassSelect
                        value={cloneTargetModelId}
                        onChange={setCloneTargetModelId}
                        placeholder="Select a model"
                        options={models.map(m => ({
                            value: m.model_id,
                            label: `${m.name}${m.model_id === sidebarContext.modelId ? ' (current)' : ''}`,
                        }))}
                    />
                    <ReactiveButton
                        label={cloneBusy ? 'Copying...' : 'Copy rule set'}
                        onClick={handleClone}
                        Icon={FiCopy}
                        style={{ justifyContent: 'center', width: '100%' }}
                    />
                </div>
            </GlassModal>

            {/* Model configuration — pick the model + its LLM layers. Editable
              * until trained, then locked. Training is gated on this being set. */}
            <div style={modelConfigCardStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FiCpu size={16} style={{ color: '#93c5fd' }} />
                    <h3 style={{ margin: 0, color: '#e2e8f0', fontSize: '1rem', fontWeight: 700 }}>Model</h3>
                    {trainedRuleNames.length > 0 && <span style={lockedBadgeStyle}>locked · trained</span>}
                    {trainedRuleNames.length === 0 && !sidebarContext.modelId && <span style={{ marginLeft: 'auto', fontSize: '0.78rem', color: '#fca5a5' }}>required before training</span>}
                </div>

                {trainedRuleNames.length > 0 ? (
                    <div style={{ color: '#cbd5e1', fontSize: '0.9rem', marginTop: 8 }}>
                        Runs on <strong style={{ color: '#e2e8f0' }}>{sidebarContext.modelName}</strong>
                        {(() => {
                            const m = models.find(x => String(x.model_id) === String(sidebarContext.modelId));
                            return m?.num_layers && Array.isArray(m.selected_layers)
                                ? ` · layers ${m.selected_layers[0]}–${m.selected_layers[1]} of ${m.num_layers}` : '';
                        })()}
                        <span style={{ color: '#64748b', marginLeft: 8 }}>(locked once trained — use “Apply to another model” to try a different one)</span>
                    </div>
                ) : modelsLoading ? (
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '8px 0 0' }} role="status">Loading models…</p>
                ) : models.length === 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
                        <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>No models yet — add one to configure this rule set.</p>
                        <button onClick={() => setAddModelOpen(true)} style={configUploadBtnStyle}><FiUploadCloud size={14} /> Add a model</button>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
                        <div>
                            <label style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'block', marginBottom: 6 }}>Model</label>
                            <GlassSelect
                                value={sidebarContext.modelId || ''}
                                onChange={handleSelectModel}
                                placeholder="Choose a model"
                                options={models.map(m => ({ value: m.model_id, label: m.name }))}
                            />
                        </div>

                        {/* Per-model LLM layer range (saved on the model; drives training).
                          * Always shown once a model is picked. When the model's layer
                          * count is unknown we cap at 100 — anything beyond the model's
                          * real count is ignored at train time. */}
                        {sidebarContext.modelId && attachLayerStart != null && (() => {
                            const m = models.find(x => String(x.model_id) === String(sidebarContext.modelId));
                            const known = !!m?.num_layers;
                            const max = m?.num_layers || 100;
                            return (
                                <div>
                                    <label style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                                        <FiLayers size={13} /> LLM layers <span style={{ color: '#64748b', fontWeight: 400 }}>· activations used to train</span>
                                    </label>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                                        <input type="number" className="glass-input" style={{ marginBottom: 0, width: 84 }} min={0} max={max - 1} value={attachLayerStart} onChange={e => setAttachLayerStart(Math.max(0, Math.min(Number(e.target.value), max - 1)))} />
                                        <span style={{ color: '#94a3b8' }}>to</span>
                                        <input type="number" className="glass-input" style={{ marginBottom: 0, width: 84 }} min={1} max={max} value={attachLayerEnd} onChange={e => setAttachLayerEnd(Math.max(1, Math.min(Number(e.target.value), max)))} />
                                        <span style={{ fontSize: '0.78rem', color: '#64748b' }}>{known ? `of ${max}` : 'max 100 (extra layers ignored at train)'}</span>
                                        <button onClick={saveLayers} disabled={layerSaving} style={{ marginLeft: 'auto', background: 'rgba(59, 130, 246, 0.18)', border: '1px solid rgba(96, 165, 250, 0.45)', color: '#bfdbfe', borderRadius: 8, padding: '6px 12px', cursor: layerSaving ? 'default' : 'pointer', fontSize: '0.8rem', fontWeight: 600 }}>{layerSaving ? 'Saving…' : 'Save layers'}</button>
                                    </div>
                                </div>
                            );
                        })()}

                        <button onClick={() => setAddModelOpen(true)} style={configUploadBtnStyle}><FiUploadCloud size={14} /> Add a new model</button>
                    </div>
                )}
            </div>

            {/* 2. Action Cards (Clean, no inline styles) */}
            <div className="action-cards-container">
                <div className="add-rule-card" onClick={openAddFromBookmarks}>
                    <FiBookmark size={28} />
                    <span>Add an Existing Rule</span>
                    <span>Pick from your Library — bookmarked rules or your own drafts.</span>
                </div>

                <div className="add-rule-card" onClick={() => setCreateOpen(true)}>
                    <FiPlus size={28} />
                    <span>Create a New Rule</span>
                    <span>Build one with AI, from your bookmarked CEs, or a new CE — then add it here.</span>
                </div>
            </div>

            <CreateChooserModal isOpen={createOpen} onClose={() => setCreateOpen(false)} />

            {/* 3. Rules List */}
            {rulesLoading ? (
                // Still fetching: say so. Rendering "No Rules Defined" here told
                // the user this rule set was empty when we simply didn't know yet.
                <div className="empty-state" role="status">
                    <FiRefreshCw size={40} style={{ color: '#64748b', marginBottom: '16px', animation: 'spin 1.4s linear infinite' }} />
                    <p style={{ color: '#94a3b8' }}>Loading rules…</p>
                </div>
            ) : rules.length === 0 ? (
                <div className="empty-state">
                    <FiInbox size={64} style={{ color: '#64748b', marginBottom: '20px' }} />
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '10px', color: '#cbd5e1' }}>No Rules Defined</h2>
                    <p style={{marginBottom: '20px', color: '#94a3b8'}}>Create a rule to start filtering content.</p>
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                        <ReactiveButton label="Add an Existing Rule" onClick={openAddFromBookmarks} Icon={FiBookmark} />
                        <ReactiveButton label="Create a New Rule" onClick={() => setCreateOpen(true)} Icon={FiPlus} />
                    </div>
                </div>
            ) : (
                <div className="rules-list">
                    {rules.map((rule) => (
                        <RuleCard 
                            key={rule.setup_id}
                            rule={rule}
                            isExpanded={expandedRule === rule.setup_id}
                            onToggle={() => setExpandedRule(expandedRule === rule.setup_id ? null : rule.setup_id)}
                            onDelete={handleDeleteRule}
                            onEditLogic={openLogicEditor}
                            onGenerateTestSets={(r) => {
                                // Phase 7 entry point for Pipeline B
                                // (Test + Evaluation). The wizard scopes to
                                // a single guardrail + rule pair, walks
                                // through 3A-3D, then runs calibration and
                                // evaluation against the generated sets.
                                // `source_rule_id` is the FK into the
                                // global rules table; `rule_id` is the
                                // legacy field name some payloads carry.
                                const rid = r.source_rule_id || r.rule_id;
                                if (!rid) {
                                    alert('This rule has no global id — refresh the page and try again.');
                                    return;
                                }
                                navigate(`/rules/${rid}`);
                            }}
                        />
                    ))}
                </div>
            )}

            {/* 4. Glass Modal */}
            <GlassModal
                isOpen={modalConfig.isOpen}
                onClose={() => setModalConfig({ ...modalConfig, isOpen: false })}
                title="Add a Rule"
            >
                {modalConfig.type === 'add_bookmarked_rule' && (() => {
                    // Merge bookmarked (public) rules with the user's own draft
                    // rules so freshly-built / AI-generated rules — which have no
                    // public_id and can't be bookmarked — can still be added to a
                    // guardrail. The public library follows them — browsed with an
                    // empty box, searched once something is typed (which also
                    // narrows the local rows) — so a community rule can be
                    // attached here instead of via the Community page.
                    // Hide any rule already attached, then dedup by id.
                    const attached = new Set((rules || []).map((r) => String(r.source_rule_id)));
                    const q = addSearchQuery.trim().toLowerCase();
                    // Mirrors the hook's 2-character minimum: below it nothing is
                    // requested, so the local list must not narrow either.
                    const searching = q.length >= 2;
                    // With the box empty the picker browses the library instead
                    // of searching it, so there are library rows to scroll into
                    // from the very first paint. The gap in between — one typed
                    // character — asks for nothing (see useLibrarySearch), so
                    // the list is the local rows alone for that keystroke.
                    const libraryLive = searching || q.length === 0;
                    const matchesQuery = (c) => !searching
                        || [c.name, c.description, c.predicate].some((f) => String(f || '').toLowerCase().includes(q));
                    const seen = new Set();
                    const keepFirst = (c) => {
                        const k = String(c.rule_id);
                        if (seen.has(k)) return false;
                        seen.add(k);
                        return true;
                    };
                    const local = [
                        ...ruleBookmarks.map((b) => ({ rule_id: b.rule_id, name: b.name, predicate: b.predicate, description: b.description, source: 'bookmark' })),
                        ...ruleDrafts.map((d) => ({ rule_id: d.rule_id, name: d.name, predicate: d.predicate, description: d.description, source: 'draft' })),
                    ].filter((c) => !attached.has(String(c.rule_id))).filter(matchesQuery).filter(keepFirst);
                    // Library hits come after the local rows and are deduped
                    // against them, so a rule the user already bookmarked stays a
                    // single row carrying its own badge. Every page collected so
                    // far is listed, oldest first, so scrolling past the
                    // bookmarks and drafts keeps going into the library.
                    const libraryRows = !libraryLive ? [] : (addLibraryRows || [])
                        .filter((item) => (item.asset_type || item.type) === 'rule')
                        .map((item) => ({
                            rule_id: item.id ?? item.rule_id,
                            name: item.name,
                            predicate: item.content || item.predicate,
                            description: item.description,
                            is_local_draft: item.is_local_draft,
                            source: 'library',
                        }))
                        .filter((c) => c.rule_id != null && !attached.has(String(c.rule_id)))
                        .filter(keepFirst);
                    const list = [...local, ...libraryRows];
                    // A request is out (or about to be) for the rows this list
                    // is still missing.
                    const fetching = libraryLive && (addSearchLoading || addPageInFlight);
                    // Footer state, in the order it can occur: the next page is
                    // on its way / there is nothing left to fetch.
                    const loadingMore = fetching && list.length > 0;
                    const atEnd = libraryLive && !loadingMore && !addLibraryHasMore && libraryRows.length > 0;
                    return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>
                                Add rules to this rule set — from your bookmarks, your unpublished drafts, or the public library. Pick as many as you like.
                            </p>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', background: 'rgba(2, 6, 23, 0.55)', border: '1.5px solid rgba(148, 163, 184, 0.22)', borderRadius: '12px' }}>
                                <FiSearch style={{ color: '#94a3b8', flexShrink: 0 }} />
                                <input
                                    type="text"
                                    value={addSearchQuery}
                                    onChange={(e) => {
                                        // A new query changes which rows are visible;
                                        // carrying ticks on now-hidden rows would attach
                                        // rules the user can no longer see.
                                        setSelectedRuleIds(new Set());
                                        setAddSearchQuery(e.target.value);
                                    }}
                                    placeholder="Search the public library…"
                                    aria-label="Search rules"
                                    style={{ flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none', color: '#f1f5f9', fontSize: '0.95rem', fontFamily: 'inherit' }}
                                />
                            </div>
                            {addSearchError && (
                                <div style={{ fontSize: '0.82rem', color: '#fca5a5' }}>{addSearchError}</div>
                            )}
                            <div
                                data-testid="add-rule-list"
                                onScroll={handleAddListScroll}
                                style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '300px', overflowY: 'auto' }}
                            >
                                {list.length === 0 ? (
                                    <div style={{ textAlign: 'center', padding: '24px', color: '#94a3b8', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
                                        {fetching ? (
                                            <span>{searching ? 'Searching…' : 'Loading the library…'}</span>
                                        ) : searching ? (
                                            <span>No rules match your search.</span>
                                        ) : (
                                            <>
                                                {/* The public library is already listed here, so
                                                    an empty list means there is nothing anywhere. */}
                                                <span>No rules to add yet — nothing in your bookmarks or drafts, and nothing in the public library.</span>
                                                <button
                                                    onClick={() => navigate('/community')}
                                                    style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '10px', border: '1px solid rgba(96, 165, 250, 0.45)', background: 'rgba(59, 130, 246, 0.18)', color: '#bfdbfe', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}
                                                >
                                                    <FiGlobe size={14} /> Browse Community rules
                                                </button>
                                            </>
                                        )}
                                    </div>
                                ) : list.map((r) => {
                                    const selected = selectedRuleIds.has(String(r.rule_id));
                                    // A search hit's badge comes from the row itself, not from
                                    // the list it arrived in — a bookmarked rule stays BOOKMARK.
                                    const isDraft = r.source === 'library' ? !!r.is_local_draft : r.source === 'draft';
                                    const isLibrary = r.source === 'library' && !isDraft;
                                    return (
                                        <div
                                            key={`${r.source}-${r.rule_id}`}
                                            role="checkbox"
                                            aria-checked={selected}
                                            onClick={() => toggleSelectedRule(r.rule_id)}
                                            style={{
                                                padding: '14px 16px', borderRadius: '12px', cursor: 'pointer',
                                                border: selected ? '2px solid #a78bfa' : '1px solid rgba(148, 163, 184, 0.18)',
                                                background: selected ? 'rgba(139, 92, 246, 0.18)' : 'rgba(15, 23, 42, 0.55)',
                                                transition: 'all 0.15s',
                                            }}
                                        >
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <input
                                                    type="checkbox"
                                                    checked={selected}
                                                    readOnly
                                                    style={{ accentColor: '#a78bfa', width: 16, height: 16, flexShrink: 0 }}
                                                />
                                                {/* flex:1 + min-width:0 lets a long name take the row's width and
                                                    wrap to at most 2 lines (clamped) instead of overflowing and
                                                    shoving the badge off the row. */}
                                                <span style={{
                                                    fontWeight: 600, color: '#f1f5f9', fontSize: '0.9rem',
                                                    flex: 1, minWidth: 0, overflowWrap: 'anywhere',
                                                    display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                                                }} title={r.name}>{r.name}</span>
                                                <span style={{
                                                    flexShrink: 0, fontSize: '0.66rem', fontWeight: 700, letterSpacing: '0.04em',
                                                    padding: '2px 8px', borderRadius: 999,
                                                    color: isDraft ? '#fcd34d' : (isLibrary ? '#6ee7b7' : '#93c5fd'),
                                                    background: isDraft ? 'rgba(245, 158, 11, 0.18)' : (isLibrary ? 'rgba(16, 185, 129, 0.18)' : 'rgba(59, 130, 246, 0.18)'),
                                                    border: `1px solid ${isDraft ? 'rgba(251, 191, 36, 0.40)' : (isLibrary ? 'rgba(52, 211, 153, 0.40)' : 'rgba(96, 165, 250, 0.40)')}`,
                                                }}>{isDraft ? 'DRAFT' : (isLibrary ? 'LIBRARY' : 'BOOKMARK')}</span>
                                            </div>
                                            {/* Prefer the rule's plain-English explanation over the raw
                                                predicate; clamp long ones with an inline Show more/less. Falls
                                                back to the predicate when a rule has no explanation yet. */}
                                            {(() => {
                                                const desc = (r.description || '').trim();
                                                if (desc) {
                                                    const expanded = expandedAddDescIds.has(String(r.rule_id));
                                                    const long = desc.length > 130;
                                                    return (
                                                        <div style={{ marginTop: '4px', marginLeft: 26 }}>
                                                            <div style={{
                                                                fontSize: '0.78rem', color: '#cbd5e1', lineHeight: 1.5, overflowWrap: 'anywhere',
                                                                ...(long && !expanded ? { display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' } : {}),
                                                            }}>{desc}</div>
                                                            {long && (
                                                                <button
                                                                    type="button"
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        setExpandedAddDescIds((prev) => {
                                                                            const next = new Set(prev);
                                                                            const k = String(r.rule_id);
                                                                            next.has(k) ? next.delete(k) : next.add(k);
                                                                            return next;
                                                                        });
                                                                    }}
                                                                    style={{ marginTop: 2, padding: 0, background: 'none', border: 'none', color: '#a5b4fc', fontSize: '0.74rem', fontWeight: 600, cursor: 'pointer' }}
                                                                >
                                                                    {expanded ? 'Show less' : 'Show more'}
                                                                </button>
                                                            )}
                                                        </div>
                                                    );
                                                }
                                                return r.predicate ? (
                                                    <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '4px', marginLeft: 26, fontFamily: 'monospace', overflowWrap: 'anywhere' }}>{r.predicate.slice(0, 80)}{r.predicate.length > 80 ? '...' : ''}</div>
                                                ) : null;
                                            })()}
                                        </div>
                                    );
                                })}
                                {loadingMore && (
                                    <div style={{ padding: '10px 4px', textAlign: 'center', fontSize: '0.78rem', color: '#94a3b8' }}>
                                        Loading more rules…
                                    </div>
                                )}
                                {atEnd && (
                                    <div style={{ padding: '10px 4px', textAlign: 'center', fontSize: '0.78rem', color: '#64748b' }}>
                                        No more rules to show.
                                    </div>
                                )}
                            </div>
                            <ReactiveButton
                                label={selectedRuleIds.size > 1 ? `Add ${selectedRuleIds.size} Rules to Rule Set` : 'Add to Rule Set'}
                                onClick={handleAddBookmarkedRule}
                                Icon={FiGlobe}
                                disabled={selectedRuleIds.size === 0}
                            />
                        </div>
                    );
                })()}
            </GlassModal>

            {/* 5. In-place rule-logic editor (groups + condition) */}
            <GlassModal
                isOpen={!!logicEdit}
                onClose={() => setLogicEdit(null)}
                title={logicEdit ? `Edit logic — ${logicEdit.rule.custom_name}` : 'Edit logic'}
            >
                {logicEdit && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        <GroupConditionEditor
                            pool={logicEdit.pool}
                            groupList={logicEdit.groupList}
                            condition={logicEdit.condition}
                            onChange={({ groupList, condition }) =>
                                setLogicEdit((prev) => (prev ? { ...prev, groupList, condition } : prev))}
                        />
                        <div>
                            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>
                                Save as (new name)
                            </label>
                            <input
                                value={logicNewName}
                                onChange={(e) => setLogicNewName(e.target.value)}
                                placeholder="Leave empty to update in place (your own drafts only)"
                                maxLength={255}
                                style={{
                                    width: '100%', boxSizing: 'border-box', padding: '10px 12px',
                                    borderRadius: 10, border: '1px solid rgba(148, 163, 184, 0.25)',
                                    background: 'rgba(2, 6, 23, 0.55)', color: '#f1f5f9', outline: 'none',
                                }}
                            />
                            <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: '#64748b' }}>
                                Editing a library rule (or a setup with no backing draft) forks it into a new
                                draft — a new name is required then. Your own drafts update in place.
                            </p>
                        </div>
                        <ReactiveButton
                            label={logicSaving ? 'Saving…' : 'Save logic'}
                            onClick={saveLogicEdit}
                            Icon={FiCheckCircle}
                            disabled={logicSaving}
                        />
                    </div>
                )}
            </GlassModal>

        </Layout>
    );
};

const actionBarStyle = {
    display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20, padding: '12px 0',
    borderBottom: '1px solid rgba(148, 163, 184, 0.14)',
};
const actionBtnStyle = {
    display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px',
    borderRadius: 8, border: '1px solid rgba(148, 163, 184, 0.22)', background: 'rgba(15, 23, 42, 0.55)',
    color: '#cbd5e1', fontSize: 13, fontWeight: 500, cursor: 'pointer',
    transition: 'all 0.15s',
};
// The stopped-by-the-user banner. Same shape as the training-failed banner, in
// neutral slate: nothing went wrong, the user asked for this.
const cancelledBannerStyle = {
    background: 'rgba(148, 163, 184, 0.14)', border: '1px solid rgba(148, 163, 184, 0.35)',
    color: '#cbd5e1', borderRadius: 8, padding: '12px 16px', margin: '12px 0',
    fontSize: '0.92rem', lineHeight: 1.5, display: 'flex', alignItems: 'flex-start', gap: 12,
};
// "Cancel training" inside the in-progress banner — red so it reads as the one
// destructive control there, but sized down: it sits next to live status text.
const cancelTrainingBtnStyle = {
    display: 'inline-flex', alignItems: 'center', gap: 6, flexShrink: 0, padding: '6px 12px',
    borderRadius: 8, border: '1px solid rgba(248, 113, 113, 0.45)', background: 'rgba(239, 68, 68, 0.18)',
    color: '#fecaca', fontSize: '0.82rem', fontWeight: 600, cursor: 'pointer',
    transition: 'all 0.15s',
};
// Under-Train stack (model tag + Change link + Apply-to-another-model).
// Model configuration card.
const modelConfigCardStyle = { background: 'rgba(15, 23, 42, 0.55)', border: '1px solid rgba(148, 163, 184, 0.16)', borderRadius: 14, padding: '16px 18px', marginBottom: 20 };
const lockedBadgeStyle = { marginLeft: 8, fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#94a3b8', background: 'rgba(148, 163, 184, 0.16)', border: '1px solid rgba(148, 163, 184, 0.3)', borderRadius: 6, padding: '2px 8px' };
const configUploadBtnStyle = { alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: 6, background: 'none', border: '1px solid rgba(96, 165, 250, 0.45)', color: '#93c5fd', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, borderRadius: 8, padding: '6px 12px' };

export default RulesManager;