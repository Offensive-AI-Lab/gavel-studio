// Pipeline C — CE (Cognitive Element) Generation step registry.
//
// The shared step list + component map for the AI CE wizard. Consumed by
// CEGenerationModal (the in-page modal that drives this step). The old
// full-page wizard + WizardShell were removed once the flow became a modal;
// this module is now just the step definition.
import Step1CEChat from './CEWizardSteps/Step1CEChat';

export const CE_STEPS = [
    { key: '1', short: '1', title: 'Cognitive Element', hint: 'Describe it in chat; the AI proposes it, then approve to build automatically.' },
];

export const CE_STEP_COMPONENTS = {
    '1': Step1CEChat,
};
