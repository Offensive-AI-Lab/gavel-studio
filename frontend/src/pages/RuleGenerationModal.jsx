// In-page modal version of the Rule Generation wizard (Pipeline A).
//
// Opened from the Browse page's "Create Rule with AI" button — no route change.
// Every open starts a BRAND-NEW pipeline run (never resumes), so closing the
// modal — for any reason, including after "Approve & Build" kicks the work into
// the background — always resets it to a fresh conversation next time. On a
// close WITHOUT approval, the half-built draft (rule + new CEs from step 2A)
// and the run are discarded so nothing lingers.
import { useTaskTray } from '../contexts/TaskTrayContext';
import { buildRuleInBackground } from '../hooks/pipelineBuild';
import WizardModal from './WizardModal';
import { RULE_STEPS, RULE_STEP_COMPONENTS } from './RuleGenerationWizard';
import { startPipelineRun, discardPipelineResources, abandonPipelineRun } from '../api';

function readUser() {
    try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
}

export default function RuleGenerationModal({ open, onClose }) {
    const tray = useTaskTray();
    const user = readUser();

    // Always a fresh run — no resume.
    const bootstrap = async () => {
        const res = await startPipelineRun({ pipelineType: 'rule' });
        return res.data;
    };

    const onFinish = async (run) => {
        buildRuleInBackground(tray, { run, userId: user?.user_id });
        // WizardModal closes the modal after onFinish resolves.
    };

    // Closed without approving → throw away the step-2A draft + the run.
    const onAbandon = (run) => {
        const stepData = run?.steps?.['2A']?.data || {};
        const ruleId = run?.rule_id || stepData.rule_id || null;
        const ceIds = stepData.ce_ids || (stepData.new_ces || []).map(c => c.ce_id);
        if (ruleId || (ceIds && ceIds.length)) {
            discardPipelineResources(ceIds || [], ruleId).catch(() => {});
        }
        if (run?.run_id) abandonPipelineRun(run.run_id).catch(() => {});
    };

    return (
        <WizardModal
            open={open}
            onClose={onClose}
            title="Generate a Rule with AI"
            steps={RULE_STEPS}
            stepComponents={RULE_STEP_COMPONENTS}
            classifierId={null}
            bootstrap={bootstrap}
            onFinish={onFinish}
            onAbandon={onAbandon}
        />
    );
}
