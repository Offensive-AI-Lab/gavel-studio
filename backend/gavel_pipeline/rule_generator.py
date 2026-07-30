"""Shared helpers for the AI rule-generation pipeline (routes/ai_pipeline.py).

This module holds the prompt-formatting, LLM-call and rule-validation helpers
used by the web pipeline and by services/rule_descriptions.py.

The old interactive command-line workflow that used to live here — main(),
save_rule() and display_rule_preview() — has been removed. The 8-step
RuleWizard and routes/ai_pipeline.py own rule generation now. Rule logic is
named CE groups + a condition string in the v2 grammar (utils/rule_condition);
the display predicate is derived from them via render_predicate. Groups the
condition never references (e.g. 'supporting') are confidence signals only —
never part of the boolean firing logic.
"""
import json
import re

import litellm

from utils.rule_condition import (
    GROUP_NAME_RE,
    KEYWORDS,
    ConditionError,
    contains_not,
    parse,
    validate_condition,
)

# Reasoning model for deep rule analysis; standard model for lighter calls.
# gpt-5.2 matches the reference pipeline ("deep, structured analysis"; was "o3").
THINKING_MODEL = "gpt-5.2"
VALIDATION_MODEL = "gpt-4.1"


def call_thinking_model(messages, model=THINKING_MODEL):
    """Calls thinking/reasoning model (gpt-5.2)."""
    try:
        print(f"\n[*] Calling {model} for deep reasoning...")
        print("[*] This may take 30-60 seconds as the model thinks through the problem...")

        # Reasoning models (gpt-5.2 / o-series) don't use system messages or temperature.
        # Convert system message to user message if present
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({"role": "user", "content": msg["content"]})
            else:
                formatted_messages.append(msg)

        response = litellm.completion(
            model=model,
            messages=formatted_messages
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Error calling thinking model: {e}"


def call_llm(messages, model=VALIDATION_MODEL, temperature=0.7):
    """Calls standard LLM."""
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Error calling LLM: {e}"


def extract_json_from_response(text):
    """Extracts JSON object from markdown code blocks or raw text."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None, "No JSON found in response"

    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {e}"


def format_ces_for_prompt(ces_dict):
    """Formats CE dictionary for prompt."""
    lines = []
    for ce_name, ce_data in ces_dict.items():
        lines.append(f"\n**{ce_name}**")
        lines.append(f"Definition: {ce_data['definition']}")
        if 'note' in ce_data:
            lines.append(f"Note: {ce_data['note']}")
        lines.append("Examples:")
        for ex in ce_data.get('examples', [])[:2]:  # Show 2 examples
            lines.append(f"  - Input: \"{ex['input']}\" → {ex['output']}")
    return "\n".join(lines)


def format_rules_for_prompt(rules_dict):
    """Formats rules dictionary (ce_groups + condition shape) for prompt."""
    lines = []
    for rule_name, rule_data in rules_dict.items():
        lines.append(f"\n**{rule_name}**")
        groups = rule_data.get('ce_groups') or {}
        for group_name, members in groups.items():
            lines.append(f"  Group '{group_name}': {members}")
        if rule_data.get('condition'):
            lines.append(f"  Condition: {rule_data['condition']}")
    return "\n".join(lines)


def validate_rule(rule_data, ces_dict):
    """Validates the LLM rule output: groups/condition shape, the v2 condition
    grammar (no 'not'), group-name conventions, and CE references."""
    issues = []

    # Check required fields
    required_fields = ['rule_name', 'description', 'groups', 'condition']
    for field in required_fields:
        if field not in rule_data:
            issues.append(f"Missing required field: {field}")

    # Validate the groups object
    groups = rule_data.get('groups')
    if groups is not None and not isinstance(groups, dict):
        issues.append("'groups' must be an object mapping group name to a list of CE names")
        groups = None

    all_ces_in_rule = set()
    clean_groups = {}
    if groups:
        for group_name, members in groups.items():
            if (not isinstance(group_name, str) or group_name in KEYWORDS
                    or not GROUP_NAME_RE.match(group_name)):
                issues.append(
                    f"Bad group name {group_name!r}: must be lowercase snake_case "
                    f"([a-z][a-z0-9_]*) and not a grammar keyword (all/of/and/or/not)"
                )
            if not isinstance(members, list) or not members:
                issues.append(f"Group {group_name!r} must be a non-empty list of CE names")
                clean_groups[str(group_name)] = []
                continue
            member_names = [m for m in members if isinstance(m, str)]
            if len(member_names) != len(members):
                issues.append(f"Group {group_name!r} contains non-string entries")
            clean_groups[str(group_name)] = member_names
            all_ces_in_rule.update(member_names)

    # Validate the condition against the groups (grammar, group refs,
    # satisfiable quantifiers) and reject negation — the LLM contract
    # forbids 'not' so studio rules stay compatible with the library CI.
    condition = rule_data.get('condition')
    if condition is not None:
        if not isinstance(condition, str) or not condition.strip():
            issues.append("'condition' must be a non-empty string")
        elif groups is not None:
            issues.extend(validate_condition(condition, clean_groups))
            try:
                if contains_not(parse(condition)):
                    issues.append("'not' is not allowed in rule conditions")
            except ConditionError:
                pass  # parse failure already reported by validate_condition

    # Check if CEs exist (either in existing CEs or new CEs)
    available_ces = set(ces_dict.keys())
    if 'new_ces' in rule_data:
        available_ces.update(rule_data['new_ces'].keys())

    unknown_ces = all_ces_in_rule - available_ces
    if unknown_ces:
        issues.append(f"Unknown CEs referenced: {unknown_ces}")

    return issues
