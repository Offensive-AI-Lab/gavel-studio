// Pipeline A — Rule Generation step registry.
//
// The shared step list + component map for the AI rule wizard. Consumed by
// RuleGenerationModal (the in-page modal that drives these steps). The old
// full-page wizard + WizardShell were removed once the flow became a modal;
// this module is now just the step definitions.
import Step1Scenario from './RuleWizardSteps/Step1Scenario';
import Step2ARule    from './RuleWizardSteps/Step2ARule';

export const RULE_STEPS = [
    { key: '1',  short: '1',  title: 'Scenario Ideation', hint: 'Describe the misuse you want to catch.' },
    { key: '2A', short: '2A', title: 'Rule Generation',   hint: 'AI proposes a rule — review it, then approve to build everything automatically.' },
];

export const RULE_STEP_COMPONENTS = {
    '1':  Step1Scenario,
    '2A': Step2ARule,
};
