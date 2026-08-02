# Helpers for LLM-generated synthetic-data configurations. Used by
# routes/ai_pipeline.py; not a standalone tool.
import json
import litellm


def load_prompt_template(filepath):
    """Loads a prompt template file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[CRITICAL] Prompt file not found: {filepath}")
        exit(1)

def call_llm_for_config(prompt, model="gpt-4.1", temperature=0.7):
    """
    Calls the LLM to generate the configuration JSON.
    Returns: (config_dict, error_message)
    """
    try:
        print(f"[*] Calling {model} to generate configuration...")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content

        # Parse JSON
        config_dict = json.loads(raw_content)

        # Validate required keys
        required_keys = [
            "scenario_instructions", "generator_model", "judge_model",
            "dynamic_components", "necessary_labels", "sufficient_labels",
            "dialogue_controls"
        ]

        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            return None, f"Generated config is missing required keys: {missing_keys}"

        print("[✓] Configuration generated successfully!")
        return config_dict, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON response: {e}"
    except Exception as e:
        return None, f"Error calling LLM: {e}"
