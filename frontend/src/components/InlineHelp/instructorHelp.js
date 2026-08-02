// instructorHelp — the instructional copy shown by InlineHelp on the matching
// pages of the app.
//
// `**bold**` is honoured by InlineHelp; paragraphs go in `summary` / `body`,
// lists go in `bullets`.

// "About GAVEL" (shown on the Hub / Workspace)
export const aboutGavel = {
    title: 'About GAVEL',
    summary: 'Governance via Activation-based Verification and Extensible Logic',
    sections: [
        {
            heading: 'What is GAVEL?',
            body: 'GAVEL detects what a language model is doing in a conversation by reading its internal activations. Lightweight probes score Cognitive Elements (CEs) — interpretable signals like "making a threat" or "payment processing" — and rules combine those signals into higher-level behavioral detections.',
        },
        {
            heading: 'Why activations?',
            body: 'Output filters match text after it has been generated. Activation probes see what the model is doing as it happens, and a rule built from CEs can be edited or recombined without retraining a monolithic detector.',
        },
        {
            heading: 'Shared like threat signatures',
            body: 'Practitioners in cybersecurity share detection signatures and compose them into custom rules. GAVEL brings the same practice to AI safety: rules and CEs are shared through the public gavel-rules library and composed to fit your use case.',
        },
        {
            heading: 'With GAVEL Studio you can:',
            bullets: [
                '**Browse and reuse** rules and CEs from the community library',
                '**Generate** GAVEL rules from plain-English scenario descriptions',
                '**Create and calibrate** new CE classifiers with synthetic datasets',
                '**Train probes** to detect CEs in LLM activations',
                '**Monitor in real time** with visual CE activation and rule trigger feedback',
            ],
        },
    ],
};

// "Welcome to Rule Set Configuration" (shown on Rule Sets)
export const manualRuleConfig = {
    title: 'Welcome to Rule Set Configuration',
    summary: 'This dashboard lets you create, edit, and manage the safety rules in your rule sets.',
    sections: [
        {
            heading: 'Model-Scoped Rule Sets',
            body: 'Each rule set is tied to a specific **model** when you train it. This ensures:',
            bullets: [
                "**Consistency**: a rule set is stored with the model it's designed for",
                '**Validation**: only the cognitive elements (CEs) your rule set is trained to detect are used',
                "**Isolation**: changes to one rule set don't affect another",
                '**Deployment**: a rule set can be exported as a bundle ready to run in production',
            ],
        },
        {
            heading: 'Getting Started',
            body: 'Open a rule set to configure its rules, or create a new one.',
        },
    ],
};

// "Unified GAVEL Evaluation Pipeline" (shown on Evaluation)
export const evaluateModel = {
    title: 'Unified GAVEL Evaluation Pipeline',
    summary: 'The evaluation automatically handles:',
    sections: [
        {
            bullets: [
                '**Calibration**: Optimizes detection thresholds using calibration data',
                '**Evaluation**: Computes metrics (TPR, FPR, AUC) on test data',
                '**Visualization**: Generates calibration plots and metric reports',
            ],
        },
        { body: 'Everything is processed in-memory for efficiency.' },
    ],
};

// "Rule Generation" (shown on the automated rule generation flow)
export const step2aRuleGeneration = {
    title: 'Rule Generation',
    summary: 'This step converts your scenario into a **GAVEL-compliant detection rule** built from Cognitive Elements (CEs). The rule formalizes the behavioral and contextual signals that must co-occur to detect the misuse, keeping detection interpretable, targeted, and aligned with your scenario.',
    sections: [
        {
            heading: 'What the Agent Will Do',
            body: 'To generate the rule, the agent runs a structured, in-depth analysis over the existing CE inventory and your scenario. Specifically, it will:',
            bullets: [
                '**Identify scenario-specific behavioral signatures** that distinguish your misuse from other types',
                '**Evaluate all existing CEs**, determining which ones apply and how they contribute to the misuse',
                '**Organize applicable CEs into named groups** and write the firing condition over them, clarifying what aspect of the misuse each group represents',
                '**Detect gaps in CE coverage**, identifying behaviors or contexts not represented in the current CE set',
                '**Propose new CEs** only when necessary, ensuring they are justified, non-overlapping, and consistent with the CE taxonomy',
                '**Assemble a complete rule** by organizing essential CEs into a coherent detection logic that specifies the required co-occurrence conditions',
            ],
        },
        { body: 'A detailed reasoning section is included with the rule, making the whole process transparent and auditable.' },
        {
            heading: 'What you get',
            body: 'A complete rule plus the list of all prerequisite CEs — both existing and newly proposed. Any missing CEs and their training datasets are then generated automatically in the background.',
        },
    ],
};

// Overview of the automated flow: rule generation + CE/dataset creation happen
// as a SINGLE step in the background.
export const automatedPipeline = {
    title: 'Automated Rule Generation',
    summary: 'This flow automates building a GAVEL rule for your use case — and automatically generates any missing cognitive elements (CEs) and their training datasets for you.',
    sections: [
        {
            heading: 'How it works',
            body: [
                'The GAVEL framework is a rule-based detection system over an LLM’s activations. Defining these rules and extracting their underlying cognitive elements (CEs) is normally a challenging, manual process.',
                'Here you just describe your scenario. The system then uses LLM agents to (1) generate the rule from your description, (2) identify which CEs you’re missing to support it, and (3) generate the training datasets for those new CEs — all automatically. The CE and dataset generation runs in the background as a single step, so you don’t have to manage it yourself.',
            ],
        },
    ],
};
