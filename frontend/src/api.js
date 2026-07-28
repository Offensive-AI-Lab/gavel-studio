// frontend/src/api.js
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_URL,
    headers: { 'Content-Type': 'application/json' },
});

// No auth header: there is no login. The backend attributes every request to
// the single local user (backend/utils/auth.py: LOCAL_USER_ID).


// ---------------------------------------------------------------------------
// Live-sync event plumbing
// ---------------------------------------------------------------------------
//
// Pages that display library content (models, guardrails, rules, CEs,
// bookmarks, drafts) listen for `gavel:libraryChanged` and re-fetch when
// it fires. The Sidebar and most pages register that listener via the
// `useLibraryRefresh` hook (see hooks/useLibraryRefresh.js).
//
// We wrap every MUTATION API call below with `withNotify(...)` so the
// event fires automatically on success — nobody at the call site needs
// to remember to dispatch. Reads are NOT wrapped (no point).
//
// Adding a new mutation? Wrap it at the api.js level and you get every
// listening page refreshed for free. The previous approach of dispatching
// inline at every component callsite was easy to forget and led to stale
// sidebars (the bug that triggered this refactor).

const notifyLibrary = () => {
    window.dispatchEvent(new Event('gavel:libraryChanged'));
};

const withNotify = async (promise) => {
    const res = await promise;
    notifyLibrary();
    return res;
};

// --- Health ---
// Long timeout because the first /health on a cold start can sit behind the
// embedding-model warmup thread (it briefly blocks the GIL). Anything shorter
// gets killed by axios and the splash flips to "Backend not responding yet"
// even though the backend is just busy loading weights.
export const getBackendHealth = async () => api.get('/health', { timeout: 30000 });

// --- Onboarding ---
// Flip the `tutorial_seen` flag to TRUE so the first-run onboarding modal
// doesn't re-fire on the next /workspace mount. The caller updates the cached
// sessionStorage user object in lockstep — see Tutorial.jsx.
export const markTutorialSeen = async () => api.put('/user/tutorial-seen');

// --- Contributor profiles ---
// Assembled by the backend from the LOCALLY-SYNCED library: any author stamped
// on a synced rule/CE has a profile, even though they never logged in here
// (nobody can — there is no login). The Profile page renders these; the
// "by [username]" link on rule/CE cards uses getUserProfile to check the
// target exists.
export const getUserProfile = async (username) =>
    api.get(`/user/profile/${encodeURIComponent(username)}`);

export const getUserContributions = async (username, type = 'rule', page = 1, pageSize = 20) =>
    api.get(`/user/profile/${encodeURIComponent(username)}/contributions`, {
        params: { type, page, page_size: pageSize },
    });

// Edit own profile. Only display_name + bio are mutable. Wrapped with
// withNotify so the sidebar / any open profile views refresh.
export const updateMyProfile = async ({ display_name, bio }) =>
    withNotify(api.patch('/user/me', { display_name, bio }));

// --- Contributor discovery ---
// Both return the same ArtistListResponse shape:
//   { page, page_size, total, items: [ArtistSummary, ...] }
// Both list only contributors whose published work is in the LOCAL synced
// library, so someone appears once Sync pulls their items in. Ratings were
// removed with accounts, so the leaderboard ranks by contribution volume and
// `by` / `minRatings` are accepted but no longer change the order.

export const searchArtists = async (q = '', page = 1, pageSize = 20) =>
    api.get('/user/search', { params: { q, page, page_size: pageSize } });

export const getLeaderboard = async (by = 'avg_rating', page = 1, pageSize = 20, minRatings = 0) =>
    api.get('/user/leaderboard', { params: { by, page, page_size: pageSize, min_ratings: minRatings } });

// --- Dashboard ---
export const getDashboardData = async (uid) => api.get(`/dashboard/${uid}`);

// --- Cognitive Elements ---
export const getUserCEs = async (uid) => api.get(`/cognitive/${uid}`);

// --- Models ---
export const getUserModels = async (uid) => api.get(`/models/${uid}`);
export const createModel = async (uid, name, storage_path, hfToken = null, numLayers = null, selectedLayers = null) =>
    withNotify(api.post(`/models/create`, {
        user_id: uid, name, storage_path, hf_token: hfToken || null,
        num_layers: numLayers, selected_layers: selectedLayers,
    }));
// Update which LLM layers a model uses ([start, end)). Saved per-model.
export const updateModelLayers = async (modelId, selectedLayers) =>
    withNotify(api.patch(`/models/${modelId}/layers`, { selected_layers: selectedLayers }));

// --- Classifiers (UI: "Guardrails") ---
export const getClassifiers = async (modelId) => api.get(`/classifiers/${modelId}`);
// Per-model create (secondary flow: a model is chosen up front).
export const createClassifier = async (modelId, name) =>
    withNotify(api.post(`/classifiers/create`, { model_id: modelId, name }));
// Primary flow: create a guardrail with just a name (model picked later, at train time).
export const createGuardrail = async (name) =>
    withNotify(api.post(`/classifiers/create`, { name }));
// Every guardrail the user owns, across all models AND the unattached ones.
export const getUserGuardrails = async () => api.get(`/classifiers/details/all`);

// --- Guardrail folders (the manual "library" grouping on the Guardrails page) ---
// Membership rides on each guardrail's folder_id in getUserGuardrails.
export const getGuardrailFolders = async () => api.get(`/guardrail-folders`);
export const createGuardrailFolder = async (name) => api.post(`/guardrail-folders`, { name });
export const renameGuardrailFolder = async (folderId, name) =>
    api.patch(`/guardrail-folders/${folderId}`, { name });
export const deleteGuardrailFolder = async (folderId) => api.delete(`/guardrail-folders/${folderId}`);
// Move a guardrail into a folder, or out of one (folderId null = ungroup).
export const assignGuardrailFolder = async (classifierId, folderId) =>
    api.post(`/guardrail-folders/assign`, { classifier_id: classifierId, folder_id: folderId });
// Bind a model-less guardrail to a model (the model-last step before training).
export const attachModel = async (classifierId, modelId) =>
    withNotify(api.post(`/classifiers/details/${classifierId}/attach-model`, { model_id: modelId }));
// "Apply to another model": deep-copy this guardrail's rule set into a new,
// untrained guardrail attached to the chosen model.
export const cloneClassifierToModel = async (classifierId, targetModelId, name = null) =>
    withNotify(api.post(`/classifiers/details/${classifierId}/clone`, { target_model_id: targetModelId, name }));
export const getClassifierDetails = async (classifierId) => api.get(`/classifiers/details/${classifierId}`);
// Compare guardrails trained on the SAME policy (rules/CEs) across different
// base models — returns each one's latest post-training evaluation metrics.
export const getPolicyComparison = async (classifierId, mode = 'same_policy') =>
    api.get(`/classifiers/${classifierId}/policy-comparison`, { params: { mode } });


// 1. Get specific logic rules for a guardrail
export const getClassifierRules = async (classifierId) => api.get(`/classifiers/${classifierId}/rules`);
// Rule-scoped detail (name, predicate, CEs with definition + examples) — for the rule page.
export const getRuleDetail = async (ruleId) => api.get(`/rules/${ruleId}/detail`);

// 2. Fork a public rule into a guardrail
export const addRuleToClassifier = async (classifierId, publicRuleId) =>
    withNotify(api.post(`/classifiers/${classifierId}/rules/add`, { rule_id: publicRuleId }));

// 3. Update the boolean logic/CEs of a specific rule instance (role-aware)
// Accepts an options object: { predicate, active_ces, ce_links }
export const updateRuleLogic = async (setupId, userId, options = {}) => {
    const payload = { user_id: userId, ...options };
    return withNotify(api.put(`/rules/setup/${setupId}`, payload));
};

// 4. Delete a rule from the guardrail
export const deleteRuleSetup = async (setupId) =>
    withNotify(api.delete(`/rules/setup/${setupId}`));

// 5. Public Library (for the "Add Rule" dropdown)
export const getPublicRules = async () => api.get(`/rules/public/library`);
export const createPublicRule = async (name, predicate, { necessary = [], fallback = [], sufficient = [], ceNames = [], categories = [] } = {}, userId, definition = "") =>
    withNotify(api.post(`/rules/public/create`, {
        name,
        predicate,
        user_id: userId,
        necessary,
        fallback,
        sufficient,
        ce_names: ceNames, // legacy for backward compatibility
        definition,
        categories,
    }));
export const addRuleBookmark = async (userId, ruleId) =>
    withNotify(api.post(`/rules/public/bookmark`, { user_id: userId, rule_id: ruleId }));
export const getRuleBookmarks = async (userId) => api.get(`/rules/public/bookmarks/${userId}`);
export const removeRuleBookmark = async (userId, ruleId) =>
    withNotify(api.delete(`/rules/public/bookmark/${userId}/${ruleId}`));

// --- Public Rule Sets (Community) ---
// A rule set is a model-agnostic, shareable collection of published rules.
// Reads are unwrapped; mutations go through withNotify so the sidebar / open
// Community pages refresh on gavel:libraryChanged.
export const getPublicRuleSets = async () => api.get(`/rules/public/rule-sets`);
export const getRuleSetDetail = async (publicId) => api.get(`/rules/public/rule-set/${publicId}/detail`);
export const addRuleSetBookmark = async (userId, ruleSetId) =>
    withNotify(api.post(`/rules/public/rule-set/bookmark`, { user_id: userId, rule_set_id: ruleSetId }));
export const getRuleSetBookmarks = async (userId) => api.get(`/rules/public/rule-set/bookmarks/${userId}`);
export const removeRuleSetBookmark = async (userId, ruleSetId) =>
    withNotify(api.delete(`/rules/public/rule-set/bookmark/${userId}/${ruleSetId}`));
// Publish a private rule set (model-less classifier) to the community.
export const publishRuleSet = async (classifierId) =>
    withNotify(api.post(`/library/publish/rule-set/${classifierId}`));
// Fork a public rule set into a new private, model-less rule set.
export const forkPublicRuleSet = async (ruleSetPublicId, name = null) =>
    withNotify(api.post(`/classifiers/from-rule-set`, { rule_set_public_id: ruleSetPublicId, name }));

// --- Library Search ---
export const searchLibrary = async ({ q, categories, asset_types, author, page=1, page_size=10, top_k, candidate_limit = 100 }) => {
    // If old code still passes top_k, map it to page_size for backward compatibility
    const size = page_size || top_k || 10;
    const params = { q };
    if (categories) params.categories = categories;
    if (asset_types) params.asset_types = asset_types;
    // Phase 4: filter results to a single author (case-insensitive).
    if (author) params.author = author;
    // Map params to new pagination API
    params.page = page;
    params.page_size = size;
    params.candidate_limit = candidate_limit;
    return api.get(`/library/search`, { params });
};
export const getAllCategories = async () => api.get(`/library/categories`);

// --- Library Sync + Publish ---
// Sync pulls deltas from the HF registry into the local DB. Idempotent and
// cheap when nothing has changed. Wire this into the login flow so the
// user always lands on the latest library.
// syncLibrary intentionally does NOT use withNotify. Called from two
// paths today: (1) login fire-and-forget for a fresh start, and (2)
// the manual "Sync now" fallback. Neither path wants
// gavel:libraryChanged dispatched on a no-op pull, so we gate the
// dispatch on actual deltas. Ongoing live updates do NOT come through
// here — they arrive via the SSE stream (LibrarySyncStream), which the
// backend pushes when its registry update poll spots a change.
export const syncLibrary = async ({ force = false } = {}) => {
    const res = await api.get(`/library/sync`, { params: force ? { force: true } : {} });
    const data = res?.data || {};
    const changed = (data.rules_added || 0) > 0
                 || (data.ces_added || 0) > 0
                 || (data.categories_synced || 0) > 0;
    if (changed) notifyLibrary();
    return res;
};

// Cheap probe: "does HF have content the local DB hasn't seen?"
// Returns { available: bool, checked: bool, reason: string|null }.
// Retained as a manual fallback; the live path is now the SSE stream,
// not a timer. Doesn't mutate anything; never fires gavel:libraryChanged.
export const checkLibraryUpdates = async () =>
    api.get(`/library/check-updates`);

// Which compute backend would run each workload right now (training / inference
// / realtime) + the accelerator. Public/unauthenticated — powers the "which GPU
// am I on" badge. Never throws secrets; safe to poll.
export const getComputeStatus = async () =>
    api.get(`/compute/status`);
// Selectable compute targets for a workload (the machine picker at train time).
export const getComputeTargets = async (workload = 'training') =>
    api.get(`/compute/targets`, { params: { workload } });

// Probe whether a given CE/role/fallback shape collides with any rule
// the user could observe (their guardrail setups + the global rules
// table). Used by the rule editor on Save to surface "this duplicates
// rule X" before the user invests in retraining a copy. See backend
// route POST /rules/check-duplicate.
export const checkRuleDuplicate = async ({ ce_links, classifier_id = null, exclude_setup_id = null }) =>
    api.post(`/rules/check-duplicate`, {
        ce_links,
        classifier_id,
        exclude_setup_id,
    });

// Save user-edited rule logic. The backend decides between an in-place
// patch (when editing the user's own draft) and a fork (when the
// source is a public rule or a manual setup with no backing rule yet);
// in the fork case it creates a new draft rule under `new_name` and
// optionally bookmarks it for cross-guardrail reuse. Wrapped with
// withNotify so My Drafts / My Bookmarks / sidebar all refresh on
// success without callers needing to know.
export const saveEditedRule = async (setupId, { user_id, ce_links, new_name = null, add_bookmark = false }) =>
    withNotify(api.post(`/rules/setup/${setupId}/save-edited`, {
        user_id,
        ce_links,
        new_name,
        add_bookmark,
    }));

// Publish pushes a local draft to the HF registry. Atomic: succeeds with a
// public_id, returns CONFLICT if the name is already taken, RACE if another
// user pushed between our sync and our push, or ERROR (and the local draft
// is removed) on hard failure. See backend/services/hf_publish.py.
export const publishCE = async (ceId) => withNotify(api.post(`/library/publish/ce/${ceId}`));
export const publishRule = async (ruleId) => withNotify(api.post(`/library/publish/rule/${ruleId}`));

// Replace a local DRAFT CE with the existing PUBLIC CE it name-clashed with
// (in place). Used by the rule-publish "adopt existing CE" resolution so a rule
// can point at a shared CE without a rule editor. See routes/library.py.
export const adoptPublicCE = async (ceId, publicId) =>
    api.post(`/library/ce/${ceId}/adopt-public`, { public_id: publicId });

// Probe whether a rule/CE name is already taken in the registry. Used by
// the AI pipeline's early-conflict modal and the rename input. Returns
// { exists, public_id?, summary? }.
export const checkLibraryName = async ({ kind, name }) =>
    api.get(`/library/check-name`, { params: { kind, name } });

// Fetch a single public record's summary by public_id (from local cache).
// Used by the conflict-modal preview pane.
export const getPublicRecord = async (kind, publicId) =>
    api.get(`/library/record/${kind}/${encodeURIComponent(publicId)}`);

// Cleanup orphan local drafts left over from interrupted AI pipelines /
// cancels / crashes. Should be called AFTER /library/sync so the manifest
// cache is fresh and ghost-published rows have already been healed.
// Returns { rules_deleted, ces_deleted, kept_for_conflict }.
export const cleanupLocalDrafts = async () => withNotify(api.post(`/library/cleanup-local-drafts`));

// List every local-draft rule and CE — powers the "My Drafts" page where
// the user reviews everything still private and decides what to publish.
// Returns { rules: [...], ces: [...] }.
export const listLocalDrafts = async () => api.get(`/library/drafts`);

// Permanently remove a single draft rule / CE from the local DB. The
// backend refuses if the row is published (is_local_draft=FALSE), so
// these are safe to call from the Drafts page without an extra check.
// Deleting a CE cascade-deletes any draft rule that referenced it; the
// response includes the deleted_rules list so the UI can update.
export const deleteDraftRule = async (ruleId) => withNotify(api.delete(`/library/drafts/rule/${ruleId}`));
export const deleteDraftCE = async (ceId) => withNotify(api.delete(`/library/drafts/ce/${ceId}`));

// List the draft rules that depend on this CE. Used to populate the
// confirm-delete dialog so the user knows which rules will be cascade-
// deleted along with the CE.
export const getCeDependentDraftRules = async (ceId) =>
    api.get(`/library/drafts/ce/${ceId}/dependent-rules`);

// --- Cognitive Elements ---
export const getCognitiveElements = async (uid) => api.get(`/cognitive/${uid}`);
export const getCognitiveDataset = async (ceId) => api.get(`/ai/ce-training/${ceId}`);
// A single CE's detail (definition + curated examples) — for the rule page.
export const getCognitiveElement = async (ceId) => api.get(`/cognitive/element/${ceId}`);
export const addCEBookmark = async (userId, ceId) =>
    withNotify(api.post(`/cognitive/bookmark`, { user_id: userId, ce_id: ceId }));
export const getCEBookmarks = async (userId) => api.get(`/cognitive/bookmarks/${userId}`);
export const removeCEBookmark = async (userId, ceId) =>
    withNotify(api.delete(`/cognitive/bookmark/${userId}/${ceId}`));
export const startScenarioChat = async () => api.post(`/ai/scenario-chat/start`);
export const sendScenarioChatMessage = async (sessionId, message) => api.post(`/ai/scenario-chat/message`, { session_id: sessionId, message });

// Rule generation pipeline. Returns the full RuleGenerationResponse —
// rule_id, predicate, new_ces[], conflict info.
export const generateGavelPipeline = async (scenario, userId, classifierId = null) =>
    api.post(`/ai/generate-pipeline`, {
        scenario,
        user_id: userId,
        classifier_id: classifierId,
    });

// Embed + finalize a rule after its CE training data is in place.
// Flips is_ready=TRUE on the rule and its CEs, computes embeddings, and
// kicks off generation of the rule's DEFAULT test/calibration set from the
// ideation `scenario` (if omitted, the backend derives one from the CEs).
export const embedResources = async ({ ruleId, ceIds = [], userId, classifierId = null, ruleCategories = [], assistantRoles = {}, scenario = null }) =>
    api.post(`/ai/embed-resources`, {
        rule_id: ruleId,
        ce_ids: ceIds,
        user_id: userId,
        classifier_id: classifierId,
        rule_categories: ruleCategories,
        assistant_role_assignments: assistantRoles,
        scenario,
    });

// Discard a half-finished pipeline run — removes the local-draft rule
// + draft CEs. The wizard's "Skip" path on step 2A uses this when the
// user rejects the AI's proposed rule.
export const discardPipelineResources = async (ceIds, ruleId = null) =>
    api.post(`/ai/discard-pipeline-resources`, { ce_ids: ceIds, rule_id: ruleId });

// CE calibration generation (reference 2C two-phase recipe — see Phase 1).
export const generateCeCalibration = async (ceId, targetCount = 30) =>
    api.post(`/ai/ce-calibration/generate`, { ce_id: ceId, target_count: targetCount });
export const getCeCalibration = async (ceId) =>
    api.get(`/ai/ce-calibration/${ceId}`);
export const getCeTraining = async (ceId) =>
    api.get(`/ai/ce-training/${ceId}`);
// CE generator with optional clarification flow. Returns
// { success, needs_clarification, clarification_question?, refuse, refuse_reason?, ce_data? }.
// `history` is the running list of prior {question, answer} clarifications.
export const generateCe = async (description, preferType = null, history = []) =>
    withNotify(api.post(`/ai/ce-generate`, { description, prefer_type: preferType, history }));
export const generateCeTraining = async ({ ce_id, ce_name, definition, category = 'CONTEXT', categories = [], examples = [], target_samples = 500, related_ce_names = [], defer_ready = false }) =>
    withNotify(api.post(`/ai/ce-training/generate`, { ce_id, ce_name, definition, category, categories, examples, target_samples, related_ce_names, defer_ready }));


// 6. Create Manual Rule (Private/Local Only)
export const createManualRule = async (classifierId, name) =>
    withNotify(api.post(`/classifiers/${classifierId}/rules/manual`, { name }));

// 7. Create AI Rule (Global -> Local)
export const createAIRule = async (classifierId, name, predicate, activeCes, userId) =>
    withNotify(api.post(`/classifiers/${classifierId}/rules/ai`, {
        name,
        predicate,
        active_ces: activeCes,
        user_id: userId
    }));

// 8. Delete Classifier
export const deleteClassifier = (classifierId) => {
    return withNotify(api.delete(`/classifiers/${classifierId}`));
};

// 9. Delete Model
export const deleteModel = (modelId) => {
    return withNotify(api.delete(`/models/${modelId}`));
};

// 10. Train Classifier (starts background training job)
export const trainClassifier = (classifierId, target = null) =>
    api.post(`/classifiers/${classifierId}/train`, null, target ? { params: { target } } : undefined);

// 10b. Classifier Training Config
export const getClassifierConfig = (classifierId) =>
    api.get(`/classifiers/${classifierId}/config`);
export const updateClassifierConfig = (classifierId, config) =>
    api.put(`/classifiers/${classifierId}/config`, config);

// 11. Get Classifier Training Status
export const getTrainingStatus = (classifierId) =>
    api.get(`/classifiers/${classifierId}/training-status`);

// 12. Download trained classifier as zip
export const downloadClassifier = async (classifierId, classifierName) => {
    const response = await api.get(`/classifiers/${classifierId}/download`, { responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    const a = document.createElement('a');
    a.href = url;
    a.download = `classifier_${classifierId}_${classifierName}.zip`;
    a.click();
    URL.revokeObjectURL(url);
};

// --- Classifier bundle export / import (server-side background jobs) ---
// Preflight: can this classifier be exported, which tiers, what's unpublished.
export const getExportPreflight = (classifierId) =>
    api.get(`/classifiers/${classifierId}/export/preflight`);

// Kick off an export job (publishes drafts if needed, then builds the bundle).
// tier: 'model' | 'model+calibration' | 'full'. Returns { job_id }.
export const startExport = (classifierId, tier) =>
    api.post(`/classifiers/${classifierId}/export/start`, null, { params: { tier } });

// Latest running/ready export job for a classifier (for resume on reopen).
export const getExportActiveJob = (classifierId) =>
    api.get(`/classifiers/${classifierId}/export/active-job`);

// Kick off an import job. Returns { job_id }.
export const startImport = (file) => {
    const form = new FormData();
    form.append('file', file);
    return api.post(`/classifiers/import/start`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
};

// Poll a bundle job (export or import).
export const getBundleJob = (jobId) =>
    api.get(`/classifiers/bundle-jobs/${jobId}`);

// Download a finished export bundle (triggers a browser download).
export const downloadBundleJob = async (jobId, fallbackName) => {
    const response = await api.get(`/classifiers/bundle-jobs/${jobId}/download`, { responseType: 'blob' });
    let filename = fallbackName || `bundle_${jobId}.gavel.zip`;
    const cd = response.headers?.['content-disposition'];
    const m = cd && /filename="?([^"]+)"?/.exec(cd);
    if (m) filename = m[1];
    const url = URL.createObjectURL(response.data);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
};

// --- Evaluation ---
export const startCalibration = (classifierId, body) =>
    api.post(`/evaluation/${classifierId}/calibrate`, body);
export const startEvaluation = (classifierId, body) =>
    api.post(`/evaluation/${classifierId}/evaluate`, body);
export const getEvaluationResults = (classifierId) =>
    api.get(`/evaluation/${classifierId}/results`);
export const getCalibratedThresholds = (classifierId) =>
    api.get(`/evaluation/${classifierId}/thresholds`);
export const getEvalResultsHistory = (classifierId, limit = 10) =>
    api.get(`/evaluation/${classifierId}/results/history`, { params: { limit } });
export const getCalibrationDataStatus = (classifierId) =>
    api.get(`/evaluation/${classifierId}/calibration-status`);

// --- Bookmark Search ---
export const searchBookmarks = async ({ user_id, q, categories, asset_types, page=1, page_size=10, top_k, candidate_limit = 100 }) => {
    // If old code still passes top_k, map it to page_size for backward compatibility
    const size = page_size || top_k || 10;
    const params = { user_id, q };
    if (categories) params.categories = categories;
    if (asset_types) params.asset_types = asset_types;
    // Map params to new pagination API
    params.page = page;
    params.page_size = size;
    params.candidate_limit = candidate_limit;
    return api.get(`/library/bookmarks/search`, { params });
};

// --- Test Sets ---
// Lists every test set usable to evaluate a classifier (across its rules).
export const listClassifierTestDatasets = (classifierId) =>
    api.get(`/ai/test-sets/by-classifier/${classifierId}`);

// --- Rule default test/calibration sets (schema v9) ---
// Every rule carries a default set, generated at rule-creation time and
// published to HF when the rule goes public.
export const deriveScenario = (ruleId) =>
    api.post(`/ai/derive-scenario`, { rule_id: ruleId });
export const generateRuleDefaults = (ruleId, scenarioInstructions, opts = {}) =>
    api.post(`/ai/rules/${ruleId}/generate-defaults`, {
        scenario_instructions: scenarioInstructions,
        // Match the reference seeded rules: 100 positive + 100 negative,
        // 50 calibration dialogues per rule.
        target_count: opts.targetCount ?? 100,
        calibration_count: opts.calibrationCount ?? 50,
    });
export const getRuleDefaultsStatus = (ruleId) =>
    api.get(`/ai/rules/${ruleId}/defaults/status`);
// Roll back a provisional rule whose default sets never finished (user
// backed out of the build-from-CEs wizard). Deletes the rule entirely.
export const discardUnreadyRule = (ruleId) =>
    api.post(`/ai/rules/${ruleId}/discard-unready`);
// Guardrail-agnostic build-from-CEs: create a draft rule (is_ready=FALSE)
// from bookmarked CEs with roles. `ceLinks` = [{ce_id, role, fallback_group}].
export const createDraftRuleFromBookmarks = (name, ceLinks, categories = [], description = '') =>
    api.post(`/ai/rules/from-bookmarked-ce`, { name, ce_links: ceLinks, categories, description });
// Finalize a build-from-CEs draft once its default set is ready: embeds the
// rule and flips is_ready=TRUE so it shows up in Drafts.
export const finalizeRule = (ruleId, ceIds = []) =>
    api.post(`/ai/rules/${ruleId}/finalize`, { ce_ids: ceIds });
export const getRuleDefaults = (ruleId) =>
    api.get(`/ai/rules/${ruleId}/defaults`);
// Rule-card / rule-page preview of the rule's single test +
// calibration set (scenario + counts + sample dialogues).
export const previewRuleTestSets = (ruleId) =>
    api.get(`/ai/rules/${ruleId}/test-sets/preview`);

// --- Pipeline runs ---
// Two flavors:
//   * 'rule'      — Pipeline A: steps 1, 2A, 2B, 2C. No guardrail scope.
//   * 'test_eval' — Pipeline B: steps 3A, 3B, 3C, 3D, cal, eval. Per-rule.
// The wizard PATCHes state on every step transition; the row is the
// single source of truth for "where in the wizard is the user?". A run
// is "active" while completed=FALSE.
export const startPipelineRun = ({ pipelineType = 'rule', classifierId = null, ruleId = null } = {}) =>
    api.post(`/pipeline-runs`, {
        pipeline_type: pipelineType,
        classifier_id: classifierId,
        rule_id: ruleId,
    });
export const getPipelineRun = (runId) =>
    api.get(`/pipeline-runs/${runId}`);
export const listActivePipelineRuns = ({ classifierId = null, pipelineType = null, ruleId = null } = {}) => {
    const params = {};
    if (classifierId != null) params.classifier_id = classifierId;
    if (pipelineType) params.pipeline_type = pipelineType;
    if (ruleId != null) params.rule_id = ruleId;
    return api.get(`/pipeline-runs/active`, { params });
};
export const updatePipelineStep = (runId, { stepId, status, data, advanceTo }) =>
    api.patch(`/pipeline-runs/${runId}/step`, {
        step_id: stepId,
        status,
        data,
        advance_to: advanceTo,
    });
export const updatePipelineLinks = (runId, { ruleId }) =>
    api.patch(`/pipeline-runs/${runId}/links`, {
        rule_id: ruleId,
    });
export const completePipelineRun = (runId) =>
    api.post(`/pipeline-runs/${runId}/complete`);
export const abandonPipelineRun = (runId) =>
    api.delete(`/pipeline-runs/${runId}`);

// --- Realtime CE Monitoring (per-window / reference-parity) ---
// The assistant response is sliced into fixed-length windows; each window
// gets its own logits. The response also includes per-RULE trigger
// booleans (predicate evaluated via the reference algorithm with the
// guardrail's calibrated thresholds + patience).
export const analyzeRealtime = (classifierId, { system_prompt, user_message, history, max_new_tokens }) =>
    api.post(`/realtime/${classifierId}/analyze`, { system_prompt, user_message, history, max_new_tokens });
// Mode 2 — classify an existing dialogue (no generation).
export const analyzeStored = (classifierId, messages) =>
    api.post(`/realtime/${classifierId}/analyze-stored`, { messages });
// Mode 2 — browsable conversation groups (rule test sets + CE calibration).
export const listSampleGroups = (classifierId) =>
    api.get(`/realtime/${classifierId}/sample-groups`);
// Mode 2 — the selectable dialogues in one group.
export const getSampleGroup = (classifierId, key) =>
    api.get(`/realtime/${classifierId}/sample-group`, { params: { key } });
// Is the target LLM already loaded in-process? Drives the one-time "loading the
// model" notice (the 7B model loads lazily on the first analysis and is cached after).
export const getRealtimeModelStatus = (classifierId) =>
    api.get(`/realtime/${classifierId}/model-status`);
// Free the model: evict it from RAM AND delete its ~15 GB HuggingFace download from disk.
export const unloadRealtimeModel = (classifierId) =>
    api.post(`/realtime/${classifierId}/unload`);

// --- Warm cluster realtime SESSION (works on any client PC) ---
// Start a warm job (loads the model once on the cluster GPU); returns immediately.
export const startRealtimeSession = (classifierId) =>
    api.post(`/realtime/${classifierId}/session/start`);
// queued | loading | ready | dead | stopped | none — poll while starting + for crashes.
export const getRealtimeSessionStatus = (classifierId) =>
    api.get(`/realtime/${classifierId}/session/status`);
// Liveness ping — keeps the job alive while in realtime.
export const realtimeSessionKeepalive = (classifierId) =>
    api.post(`/realtime/${classifierId}/session/keepalive`);
// Tear the session down (clean exit).
export const endRealtimeSession = (classifierId) =>
    api.post(`/realtime/${classifierId}/session/end`);
// Stored / live analysis routed THROUGH the warm session (no local model).
export const sessionAnalyzeStored = (classifierId, messages) =>
    api.post(`/realtime/${classifierId}/session/analyze-stored`, { messages });
export const sessionAnalyzeLive = (classifierId, { system_prompt, user_message, history, max_new_tokens }) =>
    api.post(`/realtime/${classifierId}/session/analyze`, { system_prompt, user_message, history, max_new_tokens });
// Best-effort session teardown on tab close / refresh: a keepalive fetch survives
// page unload where a normal request may be cancelled. If it doesn't make it, the
// backend's stale-session sweep + the job's idle timeout reclaim the GPU anyway.
export function endRealtimeSessionUnload(classifierId) {
    try {
        fetch(`${API_URL}/realtime/${classifierId}/session/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
            keepalive: true,
        }).catch(() => {});
    } catch { /* unload best-effort */ }
}

export default api;