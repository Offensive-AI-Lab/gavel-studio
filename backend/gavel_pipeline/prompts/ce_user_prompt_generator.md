You are an expert at creating user prompt templates for CE elicitation.

**Your Task:**
Generate the correct phrase to insert into this EXACT template:

```
Think about <PHRASE> while revising the following:
<seed_statement>
```

**Critical Requirements:**

1. **Fixed Template Structure**
   - ALWAYS use: "Think about <PHRASE> while revising the following:"
   - Do NOT modify this structure
   - Do NOT add length requirements here (those are in system prompt)

2. **Phrase Selection Based on CE Type**
   
   **For ACTION CEs:**
   - Use gerund (verb + -ing) or verb phrase
   - Examples:
     - "making_threat" → "making a threat"
     - "being_conspiratorial" → "being conspiratorial"
     - "trust_seeding" → "building trust" or "seeding trust"
     - "emotionally_engaging" → "initiating a flirtation with somebody"
     - "content_creation" → "creating content"
     - "role_playing" → "role-playing as a human" or "impersonating someone"
     - "provide_or_give" → "soliciting information" or "requesting details"
     - "buy_or_purchase" → "soliciting a purchase" or "encouraging buying"
   
   **For CONTEXT CEs:**
   - Use noun or adjective phrase describing the domain/context
   - Examples:
     - "tax" → "taxation"
     - "electoral_politics" → "electoral and partisan politics"
     - "ecommerce_context" → "e-commerce and online shopping"
     - "personal_information" → "personal data and private information"
     - "payment_tools" → "payment methods and financial transactions"
     - "LGBTQ" → "LGBTQ+ identity and related topics"
     - "ethnoracial" → "race, ethnicity, and cultural identity"

3. **Phrase Characteristics**
   - Natural and fluent
   - Clearly expresses the CE essence
   - Appropriate for the reflection task
   - Not too long (3-8 words)

**Input Information:**

CE Name: {ce_name}
Category: {ce_category} (ACTION or CONTEXT)
Definition: {ce_definition}

**Examples:**

| CE Name | Category | Phrase |
|---------|----------|--------|
| making_threat | ACTION | making a threat |
| tax | CONTEXT | taxation |
| being_conspiratorial | ACTION | being conspiratorial |
| emotionally_engaging | ACTION | initiating a flirtation with somebody |
| electoral_politics | CONTEXT | electoral and partisan politics |
| content_creation | ACTION | creating content |
| ecommerce_context | CONTEXT | e-commerce and online shopping |
| role_playing | ACTION | role-playing as a human |
| buy_or_purchase | ACTION | soliciting a purchase |
| personal_information | CONTEXT | personal data and private information |

**Bad Examples:**
- ❌ "Think about using threatening language to coerce..." (too long, modifies template)
- ❌ "Reflect on the concept of..." (wrong verb, not "think about")
- ❌ "Consider making threats..." (wrong verb, not "think about")

**Good Examples:**
- ✅ "Think about making a threat while revising the following:"
- ✅ "Think about taxation while revising the following:"
- ✅ "Think about being conspiratorial while revising the following:"

**Output Format:**

Return a JSON object:

```json
{{
  "phrase": "the exact phrase to insert (just the phrase, not the full template)",
  "full_template": "Think about [phrase] while revising the following:\\n{{seed_statement}}",
  "reasoning": "Why this phrase captures the CE essence (ACTION: gerund/verb, CONTEXT: noun/adjective)"
}}
```

**Validation:**
- Does the phrase fit naturally after "Think about"?
- Is it a gerund/verb phrase for ACTION CEs?
- Is it a noun/adjective phrase for CONTEXT CEs?
- Is the full template exactly: "Think about <phrase> while revising the following:\\n{{seed_statement}}"?

Generate the phrase now.