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
            body: 'GAVEL watches what an LLM is doing while it answers, and flags the combinations you decide are unsafe. You write the rules; nothing has to be retrained to change them.',
        },
        {
            heading: 'Why not just one big detector',
            body: 'A single detector trained on "misuse" fires on too much and explains nothing, and changing what it catches means retraining it. GAVEL splits the job into small, named signals instead ("making a threat", "payment processing"), so you can say exactly which combination matters, and see which part fired.',
        },
        {
            heading: 'How it works',
            body: [
                'Each small signal is a **Cognitive Element (CE)**. A **rule** says which CEs have to appear together for something to count as misuse: for example, a threat *and* a payment request in the same answer.',
                'It is the same idea as sharing threat signatures in security: people publish the building blocks, and you combine them into rules that fit your situation.',
            ],
        },
        {
            heading: 'What that gets you',
            bullets: [
                '**Change the rules without retraining**: define the CEs once, then recombine them as your needs change',
                '**Start from other people\'s work**: the community library ships rules and CEs you can use as-is',
                '**Show your reasoning**: every alert names the CEs that fired and the rule they satisfied',
            ],
        },
        {
            heading: 'What you can do here',
            bullets: [
                '**Write a rule from a scenario**: describe the misuse in plain language and let the studio draft the rule',
                '**Add a new CE**: define a signal the studio should learn to spot, with example dialogues generated for you',
                '**Train a rule set** on your model, so it can recognise those signals',
                '**Watch it run live**: chat with the model and see which signals light up, word by word',
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

// pages/automated_rule_generation/rule_generator.py → "Step 2A: Rule Generation"
export const step2aRuleGeneration = {
    title: 'Rule Generation',
    summary: 'Turns the scenario you described into a rule: which signals (CEs) have to appear together for this to count as misuse.',
    sections: [
        {
            heading: 'What happens next',
            body: 'The studio reads your scenario against the CEs already in your library, and:',
            bullets: [
                '**Works out what gives this misuse away**: what makes it different from ordinary requests',
                '**Picks the CEs that apply** from your library, and says what each one contributes',
                '**Groups them and writes the firing condition**, e.g. "all of intent and any of harm"',
                '**Spots what is missing**: behaviour your current CEs can\'t describe',
                '**Proposes new CEs only where needed**, avoiding ones that overlap what you already have',
            ],
        },
        { body: 'It explains its reasoning alongside the rule, so you can check the choices rather than take them on trust.' },
        {
            heading: 'What you get',
            body: 'A finished rule plus the CEs it needs, existing ones and any new ones. Missing CEs and their example dialogues are generated for you in the background.',
        },
    ],
};

// pages/automated_rule_generation/home.py → overview of the automated flow.
// Reworded for this app, where rule generation + CE/dataset creation happen as a
// SINGLE step in the background (the original app's 2A/2B/2C/3x steps are gone).
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
