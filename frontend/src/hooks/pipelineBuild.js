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

// The rules a build covers. A proposal holds one rule (the original shape) or
// several; each rule is built with only the new CEs it uses and, when there are
// several, its own scenario (a single rule keeps the user's scenario).
function rulesToBuild({ stepData, ruleId, newCes, ceIds, scenario, ruleName }) {
    const proposed = stepData.proposal?.rules;
    if (Array.isArray(proposed) && proposed.length > 1) {
        // A new CE that no rule ended up using is still finalized (as a single-
        // rule build always did) — with the first rule — so it never stays hidden.
        const used = new Set(proposed.flatMap(r => r.new_ce_names || []));
        const unused = newCes.filter(ce => !used.has(ce.ce_name)).map(ce => ce.ce_id);
        return proposed.map((r, i) => ({
            ruleId: r.rule_id,
            name: r.name || 'rule',
            scenario: r.scenario || scenario,
            ceIds: [
                ...newCes.filter(ce => (r.new_ce_names || []).includes(ce.ce_name)).map(ce => ce.ce_id),
                ...(i === 0 ? unused : []),
            ],
        }));
    }
    return [{ ruleId, name: ruleName, scenario, ceIds }];
}

// Rule pipeline (A): CE training -> CE calibration -> test/calibration set +
// embed/flip is_ready (per rule) -> complete the run. Returns true if a job was started.
export function buildRuleInBackground(tray, { run, userId }) {
    const stepData = run?.steps?.['2A']?.data || {};
    const ruleId = run?.rule_id || stepData.rule_id;
    const newCes = stepData.new_ces || [];
    const ceIds = stepData.ce_ids || newCes.map(c => c.ce_id);
    const scenario = run?.steps?.['1']?.data?.description || null;
    const ruleName = stepData.proposal?.name || 'rule';
    if (!ruleId) return false;

    const rules = rulesToBuild({ stepData, ruleId, newCes, ceIds, scenario, ruleName });
    const many = rules.length > 1;
    const total = newCes.length;
    runInTray(tray, {
        kind: 'rule',
        title: many ? `Building “${rules[0].name}” +${rules.length - 1} more` : `Building “${ruleName}”`,
        runningSubtitle: 'Starting…',
        // "Drafts" is not a place in this app — drafts live in Your Library →
        // Rules, tagged with a Draft badge. Name the real destination.
        successSubtitle: many
            ? 'Rules ready. Find them in Your Library → Rules.'
            : 'Rule ready. Find it in Your Library → Rules.',
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
            // Several rules are built one after the other (one model, one GPU).
            const failures = [];
            for (let r = 0; r < rules.length; r += 1) {
                const rule = rules[r];
                update({
                    subtitle: many
                        ? `Generating test & calibration set for rule ${r + 1}/${rules.length}…`
                        : 'Generating the rule’s test & calibration set…',
                });
                await embedResources({
                    ruleId: rule.ruleId, ceIds: rule.ceIds, userId, classifierId: null, scenario: rule.scenario,
                });
                for (let i = 0; i < 900; i += 1) { // ~30 min ceiling per rule
                    let data;
                    try { data = (await getRuleDefaultsStatus(rule.ruleId)).data; }
                    catch { break; } // status unavailable → don't hang the tray
                    if (data?.state === 'ready') break;
                    if (data?.state === 'error') {
                        // Per-bucket failure reason reported by the status endpoint.
                        // The backend reveals the failed rule, so it IS in Drafts.
                        failures.push({
                            name: rule.name,
                            reason: data?.error || 'Generation failed for one or more sets.',
                        });
                        break;
                    }
                    await sleep(2000);
                }
            }
            update({ subtitle: 'Finalizing…' });
            try { await completePipelineRun(run.run_id); } catch { /* best-effort */ }
            if (failures.length) {
                if (!many) {
                    throw new Error(
                        `Test & calibration set failed: ${failures[0].reason} The rule was still saved to Your Library → Rules; open it and use Edit to rebuild it with fresh sets.`,
                    );
                }
                throw new Error(
                    `Test & calibration set failed for ${failures.map(f => `“${f.name}” (${f.reason})`).join('; ')}. The rules were still saved to Your Library → Rules; open them and use Edit to rebuild their sets.`,
                );
            }
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
