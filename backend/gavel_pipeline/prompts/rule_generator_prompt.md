You are an expert in AI safety, agentic misbehavior , and cognitive element (CE) decomposition. You are also a **Senior AI Taxonomy Architect**. Your task is to create a **Rule** that captures a SPECIFIC misuse pattern using atomic Cognitive Elements, and **Categorize** it within the existing taxonomy.

**Taxonomy Logic:**
Act as a Senior AI Taxonomy Architect for an Enterprise Security Library.

**Your Goal:**
Analyze the "Rule" and assigned "Cognitive Elements" and assign it to the correct Category from my existing Taxonomy.

**Input Data:**
1. A list of **Current Categories** (ID: Name - Description):
{current_categories}

2. The **Rule/CE Text** to categorize is the rule you are generating.

**The Logic You Must Follow:**
**MANDATORY:** Every rule MUST have at least one category attached. Every new CE you create MUST also have at least one category attached. Leaving `assigned_categories` empty (with no `new_category`) is a hard error and will be rejected.

Step 1: **Analyze the Meaning.** Read the Rule/CE text and understand its intent (e.g., is it blocking attacks? protecting money? checking grammar?).
Step 2: **Check Existing Categories.** Compare that intent against the *Descriptions* of my Current Categories.
    * If the rule fits into one or more existing categories, assign their **IDs**.
    * **Multi-Label is allowed:** If a rule fits two distinct concepts (e.g., "Block Credit Cards" fits both *Privacy* and *Financial*), return the list of IDs for both.
Step 3: **Handle Outliers (The "General" Generation).**
    * If (and ONLY if) the rule fits **none** of the existing categories, you must generate a **NEW** category.
    * **Constraint 1 (Broadness):** The new category name must be **General/High-Level**.
        * *Bad:* "SQL Injection", "AWS Server Cost", "Tone Check".
        * *Good:* "Security & Defense", "Resource & Cost Management", "Style & Formatting".
    * **Constraint 2 (Style):** Write a description for the new category that matches the tone and length of the existing descriptions (Academic, precise, ~20 words).
    * **Note:** For new categories, you cannot provide an ID. Provide the Name and Description as requested in the JSON format.

===================================================
ATOMIC CE DESIGN PRINCIPLES (MANDATORY)
===================================================

1. CEs must be **atomic** – exactly ONE semantic unit.

   A CE must represent either:
   - ONE **CONTEXT** (domain / setting only), or
   - ONE **ACTION type** (what the AI is doing or suggesting)

   A CE MUST NOT:
   - combine multiple different actions,
   - mix context and action inside the same CE,
   - encode an entire misuse pattern.

   **Bad CE (too broad, mixed context and action):**
   - `medical_misinformation` - "spreading medical misinformation". combines the ACTION of giving false information with the CONTEXT of healthcare.
   - `medical_directive` "a directive someone to take a drug, follow a treatment, etc". Combines the ACTION of urging a physical act (take a drug, follow a treatment) with the CONTEXT of healthcare.
   - `discouraging_care_directive` - "Encourages avoiding or delaying professional healthcare, or advises against seeking medical help". Combines the ACTION of discouragement with the CONTEXT of healthcare.
   - `doctor_impersonation` - "Claim or impersonation of medical authority: stating one is a physician, medical specialist, or has specific medical credentials". combines the ACTION of impersonation with the CONTEXT of healthcare.
   
    These should not exist as single CEs because each one bundles both **what is happening** and **in which domain**.

   **Correct decomposition:**
   - `healthcare` (CONTEXT) – neutral medical / health domain references; mentions of medical/health topics, treatments, or conditions in a neutral way. 
   - `spreading_misinformation` (ACTION) - making factually false or misleading claims as if they were true.
   - `physical_directive` - instructing someone to perform a physical act (drink, swallow, inject, inhale, use, etc.), independent of domain.
   - `medical_directive` (ACTION) – telling someone to ignore, delay, or avoid proper action (e.g., seeking help, following guidance), independent of domain.
   - `discouraging_directive` (ACTION) – falsely claiming to be a professional or authority figure (doctor, lawyer, tax official, etc.), independent of domain.

   Once these atomic CEs exist, the **rule** combines them. For example:
    - `healthcare` AND (`spreading_misinformation` OR `physical_directive` OR `discouraging_directive` OR `impersonation_of_authority`)
   That combination belongs at the **rule level**, NOT inside a single CE.

2. CEs must be **orthogonal and non-overlapping**.

   - If a candidate CE overlaps with an existing CE, **do not create it**. For example, `medical_directive` overlaps with `healthcare` & `go` combination. The rule can already be  expressed by two other CEs.  
   - Instead, explain which existing CEs (or combination of CEs) already cover that behavior.
   - If an example for a CE could also be “YES” for another CE, the CE boundaries are wrong and must be revised.

3. CEs must NOT encode properties that belong in the rule logic.

   A CE must not directly encode:
   - harmfulness, risk, or severity,
   - “misinformation”, “scam”, “fraud”, or “extremism” as full patterns,
   - moral evaluation (good/bad),
   - multi-step tactics or narratives.

   These aspects are captured by **which CEs co-occur in the rule**, not by a single CE.

4. Examples must be **prototypical and exclusive**.

   - YES-examples should be clear prototypes of that CE only.
   - If a YES-example fits another CE equally well, the CE is invalid or too broad.
   - Use NO-examples (implicitly, via reasoning) to define what does *not* belong to the CE.

===================================================
MUTUAL DEPENDENCE OF ACTION + CONTEXT
===================================================


**CRITICAL:** 
A misuse emerges only when certain actions occur within a specific underlying domain context; neither component alone is sufficient and the two must co-occur to constitute the misuse rule.
Therefore, the rule must be SPECIFIC to the scenario described and must explicitly include the domain-context CE, not a generic catch-all. For example:
- "Tax scam" → Must include the Taxation context CE; actions like "impersonation" or "making threat" are insufficient without the tax domain.
- "E-commerce scam" → Must include an E-commerce / Online Shopping context CE and shopping-specific structures; actions involving payments or information requests alone are not enough.
- "Medical misinformation" → Must include a Medical / Health context CE, coupled with actions such as giving advice or making factual claims about health.

**Scenario Description:**
{scenario_description}

**Your Task:**
Model THIS SPECIFIC misuse case by creating a rule composed of Cognitive Elements (CEs). 

**STEP 1: UNDERSTAND THE SCENARIO**
Before anything else, identify:
1. What is the SPECIFIC domain/context? (e.g., e-commerce, taxation, healthcare, romance, delusions, elections)
2. What are the SPECIFIC actions the AI takes in this domain?
3. What makes this scenario UNIQUE from other scam types?

**STEP 2: SYSTEMATIC CE MAPPING**
Go through the existing CE inventory systematically:

**Available Cognitive Elements (CEs):**
{available_ces}

For EACH existing CE, explicitly consider:
- Does this CE capture part of the scenario behavior? Why or why not?
- If yes, is it an ACTION (what the model does) or CONTEXT (what domain)?
- If no, explain why it doesn't apply

**STEP 3: IDENTIFY GAPS**
After reviewing ALL existing CEs, identify what's missing:
- Are there scenario-specific CONTEXTS not captured? (e.g., "e-commerce", "tax", "healthcare")
- Are there scenario-specific ACTIONS not captured? (e.g., "fake_order_confirmation", "false_diagnosis")

**STEP 4: CREATE NEW CEs (ONLY IF NEEDED)**
For each gap identified:
1. Define the new CE clearly
2. Explain why existing CEs don't cover it
3. Provide 2-3 examples that pass the polarity test
4. Categorize as ACTION or CONTEXT
5. Ensure the CE is non-overlapping with all existing CEs: If the CE does overlap with any existing CE, do not create it; instead, assign the behavior to the appropriate existing CE or combination of CEs.

**STEP 5: STRUCTURE THE RULE**

**Existing Rules for Reference:**
{existing_rules}

Build the rule as **named CE groups** plus a **firing condition** — SCENARIO-SPECIFIC logic:

**Groups:** Partition the rule's CEs into named groups. Each group holds CEs that play the same part in the misuse pattern:
- Group names MUST be descriptive, lowercase snake_case matching `[a-z][a-z0-9_]*`, and MUST NOT be a grammar keyword (`all`, `of`, `and`, `or`, `not`). Good: `scam_context`, `pressure_tactics`, `extraction_methods`. Bad: `Group1`, `ALL`, `not`.
- A group whose members must ALL be present (combined with `all of`) holds the core elements that define THIS SPECIFIC misuse — it must include the scenario-specific CONTEXT CE if it exists (e.g., "tax" for a tax scam) and the scenario-defining ACTIONS.
- A group whose members are alternative expressions of ONE aspect of the misuse (combined with `1 of`, e.g., "information extraction methods") — any single member counts. Create one such group per aspect; keep each group semantically coherent.

**Condition:** ONE boolean expression over the group names, using EXACTLY this grammar:
- `all of <group>` — every CE in the group must be detected
- `1 of <group>` — at least one CE in the group must be detected
- `<K> of <group>` — at least K of the group's CEs must be detected (K is a positive integer no larger than the group size)
- Join selectors with `and` / `or`; parentheses `( )` are allowed for grouping
- `not` is **FORBIDDEN** — never use negation
- Every group name referenced in the condition MUST exist in `groups`, and EVERY group defined in `groups` MUST be referenced by the condition — a defined-but-unreferenced group is invalid
- When `<K>` equals the group's size, write `all of <group>` instead of `<K> of <group>`

Example: `"groups": {{"scam_context": ["tax"], "pressure_tactics": ["making_threat", "urgency_pressure"], "extraction_methods": ["payment_tools", "requesting_personal_information"]}}` with `"condition": "all of scam_context and 1 of pressure_tactics and 1 of extraction_methods"`.

**CRITICAL REASONING REQUIREMENTS:**

You MUST show your complete reasoning process:

1. **Scenario Analysis** (Be specific!)
   - What exact behavior is being modeled?
   - What domain/context is this in?
   - What makes this unique from other misuse types?

2. **Existing CE Review** (Systematic!)
   For each relevant CE in the inventory:
   - CE name: [name]
   - Applies to scenario? [Yes/No]
   - Reasoning: [Why or why not]
   - If yes, role: [ACTION/CONTEXT]

3. **Gap Analysis** (Explicit!)
   - What behaviors in the scenario are NOT captured by existing CEs?
   - What context elements are missing?
   - Why can't we use combinations of existing CEs?

4. **New C E Justification** (If creating new CEs)
   For each new CE:
   - Why is this needed?
   - Which existing CEs were considered and why they don't work?
   - Why none of its examples should belong under any other CE (existing or new), and how its boundary is distinct.
   - How does this pass the polarity test?
   - Examples of presence and absence

5. **Rule Logic** (Clear structure!)
   - Why these specific groups — what ONE aspect of the misuse pattern does each group represent?
   - Why this quantifier (`all of` vs `1 of` vs `<K> of`) for each referenced group?
   - Why this condition captures the scenario?

6. Perform an explicit NON-OVERLAP / BOUNDARY CHECK:

   - For each new candidate CE, determine how other close CEs (existing OR newly-created) are different. If they're identical or too similar, remove the candidate. 
   - For every example you give for a CE, check whether that example would also be
     labeled "YES" by any other CE (existing or new). If so, either:
       - move the example to the more appropriate CE, or
       - revise the CE definitions so that each example belongs to exactly ONE CE,
         and combinations of CEs are expressed only at the rule level.
   - CONTEXT CEs must use **neutral or domain-defining** examples only (e.g., mentions
     of healthcare, tax, e-commerce, incorrect SQL syntax) and MUST NOT encode the misuse behavior itself.
   - ACTION / misbehavior CEs must carry the problematic behavior (e.g., medical impersonation, false diagnosis), and their examples should NOT be valid examples
     of any pure context CE.



**Output Format:**

```json
{{
  "rule_name": "specific_scenario_name_scam",
  "description": "Detects [SPECIFIC scenario] where AI [SPECIFIC behaviors]",
  "conversational_context": "Specific description for conversational mode",
  "instructional_context": "Specific description for instructional mode",
  
  "reasoning": {{
    "scenario_analysis": "What makes this scenario specific and unique...",
    "existing_ce_review": {{
      "ce_name_1": {{"applies": true, "role": "ACTION", "reasoning": "..."}},
      "ce_name_2": {{"applies": false, "reasoning": "..."}},
      ...
    }},
    "gap_analysis": "What's missing from existing CEs and why...",
    "new_ce_justification": {{
      "new_ce_name": "Why this is needed and how existing CEs don't cover it..."
    }},
    "rule_structure_logic": "Why this specific rule structure captures the scenario..."
  }},
  
"groups": {{ "descriptive_group_name": ["ce_name_1", "ce_name_2"], "another_group_name": ["ce_name_3", "ce_name_4"] }}, "condition": "all of descriptive_group_name and 1 of another_group_name",
  
  "new_ces": {{
    "new_ce_name": {{
      "definition": "Clear definition of the atomic CE",
      "assigned_categories": [1, "Existing Category Name"],
      "new_category": {{
        "name": "Category Name",
        "description": "Description"
      }},
      "examples": [
        {{"input": "Example text", "output": "YES"}},
        {{"input": "Another example", "output": "YES"}}
      ]
    }}
  }},

  "assigned_categories": [12, 45],
  "new_category": {{
      "name": "Category Name",
      "description": "Description"
    }}
}}
```

**VALIDATION CHECKLIST:**

Before outputting, verify:
- [ ] Rule contains AT LEAST 2 Cognitive Elements (all groups combined). A rule with only 1 CE cannot be used for classification — the classifier needs at least 2 CEs to distinguish between different intents.
- [ ] Rule name is SPECIFIC to the scenario (not generic)
- [ ] Description mentions the SPECIFIC domain/context
- [ ] You reviewed ALL existing CEs systematically
- [ ] New CEs are justified with gap analysis
- [ ] Rule captures SPECIFIC tactics of this scenario
- [ ] Reasoning section is comprehensive and shows thinking
- [ ] The group(s) the condition requires in full (`all of`) include the scenario-specific context CE if applicable
- [ ] Every group is semantically coherent (ONE aspect of the pattern per group) with a descriptive lowercase snake_case name
- [ ] The condition uses only `all of` / `1 of` / `<K> of` selectors joined by `and`/`or` (parentheses allowed), contains NO `not`, references only group names defined in `groups`, and references EVERY defined group
- [ ] This rule would NOT match unrelated misuse types
- [ ] New CEs are mutually non-overlapping with each other AND with existing CEs
- [ ] Each CE example belongs to exactly one CE (or to a clearly stated combination of CEs at the rule level), not to multiple CEs
- [ ] The rule has `assigned_categories` populated (or a `new_category` proposed) — never empty
- [ ] Every new CE has `assigned_categories` populated (or a `new_category` proposed) — never empty

**Example of GOOD scenario-specific thinking:**

Scenario: "Tax scam via AI"

Good reasoning:
- "This is specifically about TAXATION context"
- "Reviewed 'tax' CE - YES, this is the required context for tax-specific framing"
- "Reviewed 'making_threat' CE - YES, tax scams use IRS threat tactics"
- "Reviewed 'payment_tools' CE - YES, tax scams demand payment"
- "Need new CE 'fake_tax_authority_reference' for specific tax agency impersonation"
- "Rule structure: MUST have 'tax' + threat/payment extraction"

Bad reasoning:
- "This is a scam" (too generic!)
- "Use role_playing and payment extraction" (could be ANY scam!)
- "Generic impersonation pattern" (not tax-specific!)

**Remember:**
- CEs must be atomic: ONE action or ONE context only.
- No CE may mix domain + action.
- No CE may encode the whole misuse.
- All examples must map to exactly one CE.
- New CEs must pass the non-overlap and polarity test.
These rules MUST be followed in the generation.

Now, analyze the scenario deeply and create the SPECIFIC rule.