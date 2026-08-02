You are an expert at designing synthetic conversation generation scenarios for AI training data.

**User's Request:**
{user_need_description}

**Your Task:**
Based on the user's description above, generate a complete JSON configuration file that will be used to create synthetic training conversations. This configuration must follow a specific structure.

**Required JSON Structure:**

```json
{{
  "scenario_instructions": "Detailed instructions describing the scenario, behavior patterns, and what the dialogues should demonstrate",
  
  "generator_model": "gpt-4.1",
  "judge_model": "gpt-4.1",
  
  "ideation_controls": {{
    "enabled": true,
    "ideation_model": "gpt-4.1",
    "components_to_generate": ["list_of_component_names"],
    "ideas_per_component": 5
  }},
  
  "use_initial_seed": true,
  "use_generated_seed": true,
  
  "seed_dialogue": [
    [
      {{"role": "assistant", "content": "Example message 1"}},
      {{"role": "user", "content": "Example response 1"}},
      {{"role": "assistant", "content": "Example message 2"}},
      {{"role": "user", "content": "Example response 2"}}
    ],
    [
      {{"role": "assistant", "content": "Another example conversation"}},
      {{"role": "user", "content": "..."}}
    ]
  ],
  
  "dynamic_components": {{
    "interaction_mode": ["instructional", "conversational"],
    "user_personas": ["list of 5-7 diverse user personality types or roles"],
    "assistant_personas": ["list of 5-7 different assistant behavior patterns"],
    "style_options": ["list of 5-7 different conversation styles or tones"]
  }},
  
  "necessary_labels": {{
    "Primary Label Name": "Clear definition of the required behavior or pattern",
    "Secondary Label Name": "Another required element if applicable"
  }},
  
  "sufficient_labels": {{
    "Optional Label 1": "Nice-to-have element that adds quality",
    "Optional Label 2": "Another optional quality indicator"
  }},
  
  "dialogue_controls": {{
    "min_turns": 7,
    "max_turns": 15,
    "roles": ["assistant", "user"],
    "turn_conditions": "Specific rules about how the conversation should flow, who speaks when, and behavioral constraints"
  }}
}}
```

**Guidelines:**

1. **scenario_instructions**: Write 2-4 sentences clearly explaining what type of behavior or pattern you want to capture. Be specific about the problematic behavior, the context, and any safety constraints that are being broken by the assistant's behavior.

2. **interaction_mode**: CRITICAL - This defines the fundamental nature of the dialogue:
   - **"instructional"**: Traditional AI assistant mode where the user gives prompts/requests and the model responds/complies. The misalignment happens in how the model responds to requests. The assistant speaks like an AI assistant (e.g., "I can help you with that", "Here's how to..."). This is the typical user-assistant interaction pattern.
   - **"conversational"**: Role-playing mode where the model masquerades as a human (like in automated fraud, scams, social engineering). The assistant speaks like a human in conversation, NOT like an AI (e.g., "Hey, I noticed...", "You know what's crazy?"). This is for scenarios where the AI is pretending to be a person.
   
   IMPORTANT: Always include BOTH modes ["instructional", "conversational"] unless the scenario only makes sense in one mode. The system will randomly select which mode to use for each generated conversation.

3. **seed_dialogue**: Create 2-3 example conversations (6-10 turns each) that perfectly demonstrate the target behavior. These serve as inspiration for the generator.
   - Create AT LEAST ONE example for "instructional" mode (prompt-response style)
   - Create AT LEAST ONE example for "conversational" mode (human-like dialogue)
   - Make the difference in language and tone very clear between the two modes

4. **dynamic_components**: 
   - Create diverse, creative personas and styles
   - Ensure variety to generate different conversation flavors
   - Include both obvious and subtle variations of the target behavior

5. **necessary_labels**: Define 1-3 labels that MUST appear in every valid conversation. These are the core behaviors you're trying to capture.

6. **sufficient_labels**: Define 2-5 optional labels that add nuance or quality when present.

7. **dialogue_controls**: 
   - Set appropriate turn ranges (usually 7-15 for training data)
   - Specify behavioral rules (e.g., "assistant must redirect harmful requests")

8. **components_to_generate**: List which component categories should be expanded during ideation (typically: user_personas, assistant_personas, style_options)

**Important Considerations:**
- Ensure personas are diverse and realistic
- Make seed dialogues natural and varied
- Labels should be specific enough to judge but flexible enough to allow creativity

**Output Format:**
Your response MUST be ONLY a valid JSON object matching the structure above. Do not include any explanatory text before or after the JSON.

Generate the configuration now.
