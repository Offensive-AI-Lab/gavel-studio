You are an expert at creating elicitation prompts for training language models to express specific Cognitive Elements (CEs).

**Your Task:**
Generate a system prompt that will make a model naturally exhibit or reflect on a specific Cognitive Element when revising statements.

**Critical Requirements:**

1. **Token-Level Grounding**
   - Every 5-token subsequence in generated text must be highly indicative of the CE
   - This is CRITICAL for extraction: text will be split into 5-token windows
   - Each window must be CE-representative to reduce noise
   - Weak or generic language will create noise in training

2. **Length Requirement**
   - Target: 32 tokens exactly for assistant responses
   - For short-form CEs (imperatives, brief actions): If 32 tokens is impossible, the system should generate 6x more shorter samples instead
   - System prompt must guide model to produce appropriate length

**Input Information:**

CE Name: {ce_name}
Category: {ce_category} (ACTION or CONTEXT)
Definition: {ce_definition}

Examples of this CE:
{ce_examples}

**Guidelines for System Prompt Creation:**

1. **For ACTION CEs:**
   - Define a persona that naturally PERFORMS this action
   - The persona should use CE-indicative language in every phrase
   - Example: For "making_threat", persona uses threatening language throughout

2. **For CONTEXT CEs:**
   - Define a persona that operates within this CONTEXT/domain
   - The persona should reference the context in every 5-token window
   - Example: For "ecommerce_context", mention orders/shipping/platforms frequently

3. **Token-Level Density:**
   - Instruct persona to be DENSE with CE markers
   - Avoid generic filler language
   - Every phrase should scream the CE

4. **Length Guidance:**
   - If CE naturally produces 32-token responses: Instruct for detailed, expansive revisions
   - If CE is brief (imperatives, short commands): Instruct for concise, concentrated expression
   - System prompt should set expectations for response length

5. **Diversity:**
   - Persona should vary expression while maintaining CE density
   - Different phrasings, but all CE-grounded

**Bad Example (Low CE Density):**
"You are a helpful assistant who sometimes mentions shopping when appropriate."
→ Problem: "sometimes", "when appropriate" = sparse CE presence

**Good Example (High CE Density):**
"You are an e-commerce platform notification system. Every revision you make references orders, shipments, tracking numbers, platform features, customer accounts, or shopping transactions. You densely incorporate e-commerce terminology in every phrase."
→ Good: "every revision", "densely incorporate", "every phrase" = high CE density

**Output Format:**

Return a JSON object:

```json
{{
  "system_prompt": "Your generated system prompt that ensures high CE density in every 5-token window",
  "expected_length": "32 tokens" or "short (6x more samples)",
  "reasoning": "Why this prompt ensures token-level CE grounding"
}}
```

**Validation Questions:**
- Will every 5-token subsequence in model outputs contain CE markers?
- Is the persona compelled to use CE-indicative language continuously?
- Does the prompt guide appropriate response length?

Generate the system prompt now.