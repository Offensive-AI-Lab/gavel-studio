You are an expert at generating diverse, CE-representative seed statements for training data.

**Your Task:**
Generate seed statements that express a Cognitive Element. These will be revised by a model, and the revisions will be split into 5-token windows for training.

**Critical Requirements:**

1. **High CE Density**
   - Every seed must be densely packed with CE indicators
   - After revision, every 5-token window should express the CE
   - No generic or weak seeds

2. **Maximum Diversity**
   - Seeds must be DIFFERENT from each other
   - Vary domains, contexts, specificity levels
   - Avoid repetitive patterns
   - Each seed should explore a different aspect of the CE
   - Seed statements must be unique to their CE and must not overlap with samples from any other Cognitive Element.

3. **Length Consideration**
   - For long-form CEs: Seeds can be 10-20 tokens
   - For short-form CEs (imperatives): Seeds should be 3-8 tokens
   - Seeds should naturally lead to 32-token revisions (or shorter for brief CEs)

4. **CE Category Awareness**
   - ACTION CEs: Seeds should show the action being performed
   - CONTEXT CEs: Seeds should reference the domain/context

**Input Information:**

CE Name: {ce_name}
Category: {ce_category} (ACTION or CONTEXT)
Definition: {ce_definition}
Target Count: {target_count}

Existing Examples (for reference, but CREATE NEW ONES):
{examples}

**Guidelines:**

1. **Diversity Dimensions**
   - Vary semantic content (different topics within CE)
   - Vary syntactic structure (statements, questions, commands)
   - Vary specificity (general vs. specific instances)
   - Vary intensity (subtle vs. overt expressions)

2. **CE Grounding**
   - Each seed must clearly express the CE
   - Use CE-specific terminology
   - Make CE presence unambiguous

3. **Avoid Overlap**
   - Don't repeat existing examples
   - Don't create near-duplicates
   - Each seed should add new variety

**Examples:**

**For "making_threat" (ACTION):**
Good Seeds (diverse):
- "Your account will be suspended unless you verify now"
- "Failure to comply will result in legal action"
- "We will escalate this to the authorities if you refuse"
- "This is your final warning before termination"
- "Immediate payment required or we contact collections"

Bad Seeds (repetitive):
- "Your account will be suspended"
- "Your profile will be suspended"
- "Your membership will be suspended"
→ Problem: Too similar, same verb, same structure

**For "tax" (CONTEXT):**
Good Seeds (diverse):
- "Submit Form W-7 to request your Individual Taxpayer ID Number"
- "Your tax refund direct deposit has been successfully completed"
- "Your tax liability may trigger interest charges without timely estimated payments"
- "Update IRS address via Form 8822 to ensure communication"
- "Unemployment benefits taxable; withholding available through Form W-4V"

Bad Seeds (repetitive):
- "Submit Form W-7 to initiate your Individual Taxpayer Identification Number application"
- "Submit Form W-7 to complete your Individual Taxpayer Identification Number submission"
- "Submit Form W-7 so your Individual Taxpayer Identification Number can be issued"
- "Submit Form W-7 to finalize your Individual Taxpayer Identification Number paperwork"

**For "buy_or_purchase" (SHORT ACTION):**
Good Seeds (diverse, brief):
- "Buy this item now"
- "Purchase the premium version"
- "Get the full product here"
- "Order your copy today"
- "Acquire the complete package"

**Output Format:**

Return a JSON object:

```json
{{
  "seed_statements": [
    "Seed statement 1 (high CE density, unique)",
    "Seed statement 2 (high CE density, different from 1)",
    "Seed statement 3 (high CE density, different from 1 & 2)",
    ...
  ],
  "diversity_check": "Brief explanation of how seeds vary",
  "ce_density_check": "Confirmation that all seeds are CE-dense"
}}
```

**Validation Questions:**
- Are all {target_count} seeds different from each other?
- Does each seed densely express the CE?
- Do seeds cover different aspects/variations of the CE?
- Will revised versions naturally be 32 tokens (or shorter for brief CEs)?

Generate {target_count} diverse, CE-dense seed statements now.
