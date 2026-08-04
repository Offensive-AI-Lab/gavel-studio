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
    title: 'Calibration & Evaluation',
    summary: 'Two steps that tell you whether this rule set is safe to run:',
    sections: [
        {
            bullets: [
                '**Calibration** picks how confident each CE has to be before it counts as detected. It runs automatically after training; recalibrate only if the calibration data changed.',
                '**Evaluation**: Computes metrics (TPR, FPR, AUC) on test data',
                '**Visualization**: Generates calibration plots and metric reports',
            ],
        },
        { body: 'Both run in the background, so you can leave this page and come back.' },
    ],
};

// "Rule Generation" (shown on the automated rule generation flow)
export const step2aRuleGeneration = {
    title: 'Rule Generation',
    summary: 'Turns the scenario you described into a rule: which signals (CEs) have to appear together for this to count as misuse.',
    sections: [
        {
            heading: 'What happens next',
            body: 'The studio reads your scenario against the CEs already in your library, and:',
            bullets: [
                '**Identify scenario-specific behavioral signatures** that distinguish your misuse from other types',
                '**Evaluate all existing CEs**, determining which ones apply and how they contribute to the misuse',
                '**Organize applicable CEs into named groups** and write the firing condition over them, clarifying what aspect of the misuse each group represents',
                '**Detect gaps in CE coverage**, identifying behaviors or contexts not represented in the current CE set',
                '**Propose new CEs** only when necessary, ensuring they are justified, non-overlapping, and consistent with the CE taxonomy',
                '**Assemble a complete rule** by organizing essential CEs into a coherent detection logic that specifies the required co-occurrence conditions',
            ],
        },
        { body: 'It explains its reasoning alongside the rule, so you can check the choices rather than take them on trust.' },
        {
            heading: 'What you get',
            body: 'A finished rule plus the CEs it needs, existing ones and any new ones. Missing CEs and their example dialogues are generated for you in the background.',
        },
    ],
};

// Overview of the automated flow: rule generation + CE/dataset creation happen
// as a SINGLE step in the background.
export const automatedPipeline = {
    title: 'Automated Rule Generation',
    summary: 'Describe the misuse you want caught; the studio writes the rule and fills in any signals (CEs) you don’t have yet.',
    sections: [
        {
            heading: 'How it works',
            body: [
                'Writing a rule by hand means deciding which signals matter, naming each one, and producing example dialogues to teach it: slow, fiddly work.',
                'Here you describe the scenario instead. The studio drafts the rule, checks which CEs you are missing, and generates example dialogues for those, in the background, as one step. You review the result before anything is kept.',
            ],
        },
    ],
};
