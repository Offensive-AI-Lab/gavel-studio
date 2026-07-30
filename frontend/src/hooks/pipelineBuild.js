// Background "build everything after approval" jobs, shared by the page
// wizards (RuleGenerationWizard / CEGenerationWizard) and the in-page modals
// (RuleGenerationModal / CEGenerationModal). Each kicks off ONE detached
// task-tray job that chains the remaining pipeline work and reports progress
// on the tray chip. The caller decides what to do with the UI afterwards
// (navigate, or just close a modal).
import { runInTray, sleep } from './runInTray';
import {
    generateCeTraining,
    generateCeCalibration,
    embedResources,
    completePipelineRun,
    getRuleDefaultsStatus,
} from '../api';

// reference defaults — not user-tunable.
const CE_TRAINING_SAMPLES = 500;
const CE_CALIBRATION_DIALOGUES = 30;

function ceCategories(ce) {
    const cats = [...(ce.assigned_categories || [])];
    if (ce.new_category?.name) cats.push(ce.new_category.name);
    return cats;
}

// Rule pipeline (A): CE training -> CE calibration -> test/calibration set +
// embed/flip is_ready -> complete the run. Returns true if a job was started.
export function buildRuleInBackground(tray, { run, userId }) {
    const stepData = run?.steps?.['2A']?.data || {};
    const ruleId = run?.rule_id || stepData.rule_id;
    const newCes = stepData.new_ces || [];
    const ceIds = stepData.ce_ids || newCes.map(c => c.ce_id);
    const scenario = run?.steps?.['1']?.data?.description || null;
    const ruleName = stepData.proposal?.name || 'rule';
    if (!ruleId) return false;

    const total = newCes.length;
    runInTray(tray, {
        kind: 'rule',
        title: `Building “${ruleName}”`,
        runningSubtitle: 'Starting…',
        successSubtitle: 'Rule ready — find it in your Drafts.',
        job: async (update) => {
            for (let i = 0; i < newCes.length; i += 1) {
                const ce = newCes[i];
                update({ subtitle: `Training cognitive elements ${i + 1}/${total}…` });
                await generateCeTraining({
                    ce_id: ce.ce_id, ce_name: ce.ce_name, definition: ce.definition,
                    category: ce.category || 'CONTEXT', categories: ce.categories || [],
                    target_samples: CE_TRAINING_SAMPLES,
                    // Keep the CE hidden until the whole rule (incl. its test set)
                    // is built; generate_rule_defaults flips is_ready at the end.
                    defer_ready: true,
                });
            }
            for (let i = 0; i < newCes.length; i += 1) {
                update({ subtitle: `Calibrating cognitive elements ${i + 1}/${total}…` });
                await generateCeCalibration(newCes[i].ce_id, CE_CALIBRATION_DIALOGUES);
            }
            // embed-resources kicks off the default test/calibration set in the
            // background; the backend flips the rule + its CEs to is_ready only
            // when that set is done (so the rule never shows half-built). Wait
            // for it here so the tray "ready" matches when the rule appears.
            update({ subtitle: 'Generating the rule’s test & calibration set…' });
            await embedResources({ ruleId, ceIds, userId, classifierId: null, scenario });
            for (let i = 0; i < 900; i += 1) { // ~30 min ceiling
                let state;
                try { state = (await getRuleDefaultsStatus(ruleId)).data?.state; }
                catch { state = 'ready'; } // status unavailable → don't hang the tray
                if (state === 'ready' || state === 'error') break;
                await sleep(2000);
            }
            update({ subtitle: 'Finalizing…' });
            try { await completePipelineRun(run.run_id); } catch { /* best-effort */ }
        },
    });
    return true;
}

// CE pipeline (C): training set (creates the CE) -> calibration -> embed/flip
// is_ready -> complete the run. Returns true if a job was started.
export function buildCeInBackground(tray, { run, userId }) {
    // The CE chat stores its accepted proposal under step '1'.
    const ceData = run?.steps?.['1']?.data?.ce_data;
    if (!ceData) return false;

    runInTray(tray, {
        kind: 'ce',
        title: `Building “${ceData.name}”`,
        runningSubtitle: 'Starting…',
        successSubtitle: 'Cognitive element ready — find it in your library.',
        job: async (update) => {
            update({ subtitle: 'Generating training data…' });
            const res = await generateCeTraining({
                ce_name: ceData.name,
                definition: ceData.definition,
                category: ceData.type || 'CONTEXT',
                categories: ceCategories(ceData),
                examples: (ceData.in_scope_examples || []).map(ex => ({
                    input: typeof ex === 'string' ? ex : JSON.stringify(ex),
                    output: 'YES',
                })),
                target_samples: CE_TRAINING_SAMPLES,
                // Stay hidden until calibration + embed finish; embed_resources
                // (_flip_ready) reveals the CE only once everything is done.
                defer_ready: true,
            });
            const ceId = res.data?.ce_id;
            if (!ceId) throw new Error('Training did not return a CE id.');

            update({ subtitle: 'Calibrating…' });
            await generateCeCalibration(ceId, CE_CALIBRATION_DIALOGUES);

            update({ subtitle: 'Finalizing…' });
            await embedResources({ ruleId: null, ceIds: [ceId], userId, classifierId: null });
            try { await completePipelineRun(run.run_id); } catch { /* best-effort */ }
        },
    });
    return true;
}
