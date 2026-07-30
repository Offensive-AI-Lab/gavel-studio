// instructorHelp — the explanation copy from the original GAVEL Studio app,
// transcribed verbatim and reshaped into the InlineHelp content model so it can
// render directly on the matching pages of this app. Source of truth for the
// wording is the GAVEL Studio (gavel_app) pages; keep these in sync with it.
//
// `**bold**` is honoured by InlineHelp; paragraphs go in `summary` / `body`,
// lists go in `bullets`.

// pages/home.py → "About GAVEL"  (mapped onto the Hub / Workspace)
export const aboutGavel = {
    title: 'About GAVEL',
    summary: 'Governance via Activation-based Verification and Extensible Logic',
    sections: [
        {
            heading: 'What is GAVEL?',
            body: 'GAVEL introduces a groundbreaking approach to AI safety through rule-based activation monitoring. Inspired by rule-sharing practices in cybersecurity, GAVEL enables practitioners to configure precise, interpretable safeguards for large language models without retraining models or detectors.',
        },
        {
            heading: 'The Challenge',
            body: 'Traditional activation-based safety approaches train broad detectors on misuse datasets, resulting in poor precision and limited adaptability. Once deployed, these systems are rigid—updating them requires expensive retraining and offers little visibility into why decisions were made.',
        },
        {
            heading: 'The GAVEL Approach',
            body: [
                'GAVEL introduces compositional safety through fine-grained Cognitive Elements (CEs). Instead of monolithic detectors, GAVEL models LLM activations as interpretable factors like "making a threat" or "payment processing." These elements can be combined into precise, domain-specific rules without retraining models or detectors.',
                'Think of it like cybersecurity: practitioners share threat signatures and compose them into custom rules. GAVEL brings this same flexibility to AI governance.',
            ],
        },
        {
            heading: 'A New Paradigm for AI Safety',
            body: 'GAVEL fundamentally changes activation-based safety by:',
            bullets: [
                '**Decoupling activation engineering from safety configuration** — Build CE vocabularies once, then compose them into countless safety rules',
                '**Enabling community-driven safety** — Share rules and CE definitions across organizations to accelerate collective progress',
                '**Supporting regulatory compliance** — Provide auditable, interpretable explanations for safety decisions',
            ],
        },
        {
            heading: 'This System Enables You To:',
            bullets: [
                '**Automatically generate** GAVEL rules from scenario descriptions',
                '**Create and calibrate** new CE classifiers with synthetic datasets',
                '**Train RNN models** to detect CEs in LLM activations',
                '**Monitor in real-time** with visual CE activation and rule trigger feedback',
            ],
        },
    ],
};

// core/ruleset_config.py → "Welcome to Rule Configuration" (mapped onto Rule Sets)
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

// pages/detector_training/evaluate_unified.py → "Unified GAVEL Evaluation Pipeline"
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

// pages/automated_rule_generation/rule_generator.py → "Step 2A: Rule Generation"
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

// pages/automated_rule_generation/home.py → overview of the automated flow.
// Reworded for this app, where rule generation + CE/dataset creation happen as a
// SINGLE step in the background (the original app's 2A/2B/2C/3x steps are gone).
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
