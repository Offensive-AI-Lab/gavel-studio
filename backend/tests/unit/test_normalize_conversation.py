"""Unit tests for evaluation.inference._normalize_conversation.

`_normalize_conversation` coerces any conversation shape into the canonical
`List[{"role": str, "content": str}]` that the chat template expects. It is a
pure function (no DB / model access), so these are plain unit tests — the heavy
inference pipeline around it is never exercised. We only import the one helper.

Behavior under test (from the source docstring + implementation):
  * raw string -> single assistant message
  * non-list, non-string (e.g. None, int, dict) -> single empty user message
  * list of message dicts -> passed through with role/content fallbacks
      - role fallback order: "role" -> "from" -> alternating (user on even idx)
      - content fallback order: "content" -> "value" -> "text" -> ""
      - non-str content is str()-ified; role is always str()-ified
  * list of plain strings -> alternating user/assistant (user on even idx)
  * list with other element types -> alternating role, str() of the element
  * empty list -> empty list
"""
import pytest

from evaluation.inference import _normalize_conversation


# ---------------------------------------------------------------------------
# Raw string input
# ---------------------------------------------------------------------------


class TestRawString:
    def test_plain_string_becomes_single_assistant_message(self):
        out = _normalize_conversation("hello world")
        assert out == [{"role": "assistant", "content": "hello world"}]

    def test_empty_string_becomes_assistant_message_with_empty_content(self):
        # An empty string is still a str, so it takes the str branch (not the
        # "anything else" empty-user fallback).
        out = _normalize_conversation("")
        assert out == [{"role": "assistant", "content": ""}]


# ---------------------------------------------------------------------------
# Non-list, non-string inputs -> degenerate single empty user message
# ---------------------------------------------------------------------------


class TestNonListNonString:
    def test_none_returns_single_empty_user_message(self):
        assert _normalize_conversation(None) == [{"role": "user", "content": ""}]

    def test_int_returns_single_empty_user_message(self):
        assert _normalize_conversation(42) == [{"role": "user", "content": ""}]

    def test_bare_dict_returns_single_empty_user_message(self):
        # A dict is not a list, so the whole dict is treated as "anything else"
        # and collapsed to one empty user message — it is NOT iterated.
        out = _normalize_conversation({"role": "system", "content": "ignored"})
        assert out == [{"role": "user", "content": ""}]


# ---------------------------------------------------------------------------
# List of message dicts — pass-through with fallbacks
# ---------------------------------------------------------------------------


class TestListOfDicts:
    def test_well_formed_dicts_pass_through(self):
        conv = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _normalize_conversation(conv) == conv

    def test_from_alias_used_when_role_missing(self):
        conv = [{"from": "human", "content": "hey"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "human", "content": "hey"}]

    def test_value_alias_used_when_content_missing(self):
        conv = [{"role": "user", "value": "via value"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": "via value"}]

    def test_text_alias_used_when_content_and_value_missing(self):
        conv = [{"role": "user", "text": "via text"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": "via text"}]

    def test_role_falls_back_to_alternating_when_absent(self):
        # No role/from key -> alternating: index 0 user, index 1 assistant.
        conv = [{"content": "first"}, {"content": "second"}]
        out = _normalize_conversation(conv)
        assert out == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]

    def test_missing_content_defaults_to_empty_string(self):
        conv = [{"role": "user"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": ""}]

    def test_role_priority_role_over_from(self):
        conv = [{"role": "system", "from": "human", "content": "x"}]
        out = _normalize_conversation(conv)
        assert out[0]["role"] == "system"

    def test_content_priority_content_over_value_over_text(self):
        conv = [{"role": "user", "content": "c", "value": "v", "text": "t"}]
        out = _normalize_conversation(conv)
        assert out[0]["content"] == "c"

    def test_non_string_content_is_stringified(self):
        conv = [{"role": "user", "content": 123}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": "123"}]

    def test_non_string_role_is_stringified(self):
        conv = [{"role": 7, "content": "hi"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "7", "content": "hi"}]

    def test_falsy_content_zero_falls_through_to_empty(self):
        # `msg.get("content")` returns 0 which is falsy, so the `or` chain
        # keeps walking and lands on "" (not "0").
        conv = [{"role": "user", "content": 0}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": ""}]

    def test_falsy_role_empty_string_falls_back_to_alternating(self):
        # role "" is falsy -> from missing -> alternating (idx 0 => user).
        conv = [{"role": "", "content": "hi"}]
        out = _normalize_conversation(conv)
        assert out == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# List of plain strings — alternating user/assistant
# ---------------------------------------------------------------------------


class TestListOfStrings:
    def test_alternating_roles_first_is_user(self):
        conv = ["q1", "a1", "q2"]
        out = _normalize_conversation(conv)
        assert out == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]

    def test_single_string_in_list_is_user(self):
        assert _normalize_conversation(["only"]) == [
            {"role": "user", "content": "only"}
        ]

    def test_empty_string_entries_preserved(self):
        out = _normalize_conversation(["", ""])
        assert out == [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]


# ---------------------------------------------------------------------------
# List with other element types — alternating role, str() of element
# ---------------------------------------------------------------------------


class TestListOfOther:
    def test_non_dict_non_str_elements_stringified_and_alternated(self):
        conv = [1, 2, 3]
        out = _normalize_conversation(conv)
        assert out == [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]

    def test_none_element_stringified(self):
        out = _normalize_conversation([None])
        assert out == [{"role": "user", "content": "None"}]

    def test_mixed_dict_string_and_other(self):
        conv = [
            {"role": "system", "content": "sys"},  # 0
            "a plain string",                        # 1 -> assistant
            99,                                      # 2 -> user
        ]
        out = _normalize_conversation(conv)
        assert out == [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "a plain string"},
            {"role": "user", "content": "99"},
        ]


# ---------------------------------------------------------------------------
# Edge / empty cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_list_returns_empty_list(self):
        assert _normalize_conversation([]) == []

    def test_output_is_always_list_of_role_content_dicts(self):
        # Structural contract: every element has exactly role+content str keys.
        samples = [
            "str",
            None,
            123,
            [{"role": "user", "content": "x"}],
            ["a", "b"],
            [{"content": "no role"}],
        ]
        for s in samples:
            out = _normalize_conversation(s)
            assert isinstance(out, list)
            for msg in out:
                assert set(msg.keys()) == {"role", "content"}
                assert isinstance(msg["role"], str)
                assert isinstance(msg["content"], str)
