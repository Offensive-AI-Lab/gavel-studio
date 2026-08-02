// In-page modal version of the CE Generation wizard (Pipeline C).
//
// Opened from the Browse-CEs page's "Create New CE (AI)" button — no route
// change. Every open starts a BRAND-NEW run (never resumes), so closing — for
// any reason, including after "Approve & Build" runs in the background — always
// resets to a fresh concept next time. Step 2.1 only PROPOSES the CE (the DB CE
// isn't created until the background build), so an un-approved close just
// abandons the run.
import { useTaskTray } from '../contexts/TaskTrayContext';
import { buildCeInBackground } from '../hooks/pipelineBuild';
import WizardModal from './WizardModal';
import { CE_STEPS, CE_STEP_COMPONENTS } from './CEGenerationWizard';
import { startPipelineRun, abandonPipelineRun } from '../api';

function readUser() {
    try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; }
}

export default function CEGenerationModal({ open, onClose }) {
    const tray = useTaskTray();
    const user = readUser();

    const bootstrap = async () => {
        const res = await startPipelineRun({ pipelineType: 'ce' });
        return res.data;
    };

    const onFinish = async (run) => {
        buildCeInBackground(tray, { run, userId: user?.user_id });
    };

    const onAbandon = (run) => {
        if (run?.run_id) abandonPipelineRun(run.run_id).catch(() => {});
    };

    return (
        <WizardModal
            open={open}
            onClose={onClose}
            title="Generate a Cognitive Element with AI"
            steps={CE_STEPS}
            stepComponents={CE_STEP_COMPONENTS}
            classifierId={null}
            bootstrap={bootstrap}
            onFinish={onFinish}
            onAbandon={onAbandon}
        />
    );
}
