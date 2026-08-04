"""Unit tests for utils/rule_condition.py — the ported gavel-rules v2
condition grammar. The parser/evaluator must stay byte-compatible with
upstream tools/condition.py: same strings accepted, same firing decisions.
validate_condition mirrors the library CI validator (tools/validate.py)."""
import pytest

from utils.rule_condition import (
    ConditionError,
    contains_not,
    evaluate,
    evaluate_condition,
    parse,
    referenced_groups,
    render_predicate,
    selectors,
    validate_condition,
)

GROUPS = {
    "hook": ["emotionally_engaging"],
    "action": ["buy_or_purchase", "click_or_enter", "send_or_transfer"],
    "rapport": ["trust_seeding", "role_playing"],
}


# --- parse -----------------------------------------------------------------

def test_parse_single_selector():
    assert parse("all of hook") == ("sel", "all", "hook")
    assert parse("1 of action") == ("sel", 1, "action")
    assert parse("2 of action") == ("sel", 2, "action")


def test_parse_real_library_conditions():
    # Every condition shape currently in gavel-rules must parse.
    for cond in [
        "all of emotional_hook and 1 of solicited_action and 1 of rapport_tactic",
        "all of conspiratorial_frame and 1 of reinforcement_tactic",
        "all of sensitive_data_context and 1 of solicited_action",
        "all of malformed_sql_crafting",
    ]:
        parse(cond)


def test_parse_precedence_and_binds_tighter_than_or():
    ast = parse("all of hook or all of action and all of rapport")
    assert ast == ("or", [("sel", "all", "hook"),
                          ("and", [("sel", "all", "action"), ("sel", "all", "rapport")])])


def test_parse_not_and_parens():
    assert parse("not all of hook") == ("not", ("sel", "all", "hook"))
    ast = parse("(all of hook or 1 of action) and not 1 of rapport")
    assert ast[0] == "and"
    assert ast[1][0][0] == "or"
    assert ast[1][1] == ("not", ("sel", 1, "rapport"))


@pytest.mark.parametrize("bad", [
    "",
    "all hook",                 # missing "of"
    "some of hook",             # bad quantifier
    "all of",                   # missing group
    "all of Hook",              # uppercase group name
    "all of hook extra",        # trailing tokens
    "(all of hook",             # unclosed paren
    "all of and",               # keyword as group name
    "all of hook &",            # bad token
])
def test_parse_rejects(bad):
    with pytest.raises(ConditionError):
        parse(bad)


# --- evaluate --------------------------------------------------------------

def _fires(cond, detected):
    return evaluate_condition(cond, GROUPS, detected)


def test_evaluate_all_of():
    assert _fires("all of rapport", {"trust_seeding", "role_playing"})
    assert not _fires("all of rapport", {"trust_seeding"})


def test_evaluate_one_of():
    assert _fires("1 of action", {"click_or_enter"})
    assert not _fires("1 of action", {"trust_seeding"})


def test_evaluate_n_of_threshold():
    # The shape the old necessary/fallback model could not express.
    assert not _fires("2 of action", {"click_or_enter"})
    assert _fires("2 of action", {"click_or_enter", "send_or_transfer"})
    assert _fires("2 of action", {"buy_or_purchase", "click_or_enter", "send_or_transfer"})


def test_evaluate_and_or_not():
    assert _fires("all of hook and 1 of action",
                  {"emotionally_engaging", "buy_or_purchase"})
    assert not _fires("all of hook and 1 of action", {"emotionally_engaging"})
    assert _fires("all of hook or 1 of action", {"buy_or_purchase"})
    assert _fires("not 1 of action", {"emotionally_engaging"})
    assert not _fires("not 1 of action", {"buy_or_purchase"})


def test_evaluate_nested():
    cond = "(all of hook or 2 of action) and not all of rapport"
    assert _fires(cond, {"emotionally_engaging"})
    assert _fires(cond, {"buy_or_purchase", "click_or_enter", "trust_seeding"})
    assert not _fires(cond, {"emotionally_engaging", "trust_seeding", "role_playing"})


def test_evaluate_empty_detected():
    assert not _fires("all of hook", frozenset())
    assert _fires("not all of hook", frozenset())


# --- helpers ---------------------------------------------------------------

def test_referenced_groups_and_selectors():
    ast = parse("(all of hook or 2 of action) and not 1 of rapport")
    assert referenced_groups(ast) == {"hook", "action", "rapport"}
    sels = selectors(ast)
    assert ("sel", 2, "action") in sels and len(sels) == 3


def test_contains_not():
    assert contains_not(parse("not all of hook"))
    assert contains_not(parse("all of hook and not 1 of action"))
    assert not contains_not(parse("all of hook and 1 of action"))


# --- render_predicate ------------------------------------------------------

def test_render_single_member_group_is_bare_ce():
    assert render_predicate(parse("all of hook"), GROUPS) == "emotionally_engaging"


def test_render_one_of_and_all_of():
    assert render_predicate(parse("1 of rapport"), GROUPS) == "(trust_seeding OR role_playing)"
    assert render_predicate(parse("all of rapport"), GROUPS) == "(trust_seeding AND role_playing)"
    # quantifier equal to the group size renders as AND too
    assert render_predicate(parse("2 of rapport"), GROUPS) == "(trust_seeding AND role_playing)"


def test_render_k_of_and_not_and_mixed():
    assert render_predicate(parse("2 of action"), GROUPS) == (
        "at least 2 of (buy_or_purchase, click_or_enter, send_or_transfer)"
    )
    assert render_predicate(parse("not 1 of rapport"), GROUPS) == (
        "NOT (trust_seeding OR role_playing)"
    )
    rendered = render_predicate(
        parse("all of hook and (1 of rapport or 2 of action)"), GROUPS
    )
    assert rendered == (
        "emotionally_engaging AND ((trust_seeding OR role_playing)"
        " OR at least 2 of (buy_or_purchase, click_or_enter, send_or_transfer))"
    )


# --- validate_condition ----------------------------------------------------

def test_validate_accepts_good_pair():
    assert validate_condition(
        "all of hook and 1 of action and 1 of rapport", GROUPS
    ) == []


def test_validate_reports_parse_error():
    problems = validate_condition("all of", GROUPS)
    assert len(problems) == 1 and "does not parse" in problems[0]


def test_validate_undefined_group():
    problems = validate_condition("all of missing_group", GROUPS)
    assert any("undefined group 'missing_group'" in p for p in problems)


def test_validate_unsatisfiable_quantifier():
    problems = validate_condition("4 of action", GROUPS)
    assert any("can never be satisfied" in p for p in problems)
    problems = validate_condition("0 of action", GROUPS)
    assert any("can never be satisfied" in p for p in problems)


def test_validate_empty_group():
    problems = validate_condition("all of hook", {**GROUPS, "empty": []})
    assert any("group 'empty' has no members" in p for p in problems)


def test_validate_rejects_negation():
    # Matches library CI: the parser accepts 'not' but validation rejects it.
    problems = validate_condition(
        "all of hook and not 1 of action and 1 of rapport", GROUPS
    )
    assert any("negation is not supported" in p for p in problems)


def test_validate_rejects_dead_group():
    # Every defined group must be referenced by the condition (no dead
    # groups) — the retired 'supporting' concept is invalid now.
    problems = validate_condition("all of hook and 1 of action", GROUPS)
    assert any("'rapport' is not referenced" in p for p in problems)


def test_validate_all_of_lint():
    # k equal to the group size must be written as 'all of g'.
    problems = validate_condition(
        "all of hook and 1 of action and 2 of rapport", GROUPS
    )
    assert any("write 'all of rapport'" in p for p in problems)


def test_validate_lazy_group_name():
    problems = validate_condition("all of group1", {"group1": ["tax"]})
    assert any("not descriptive" in p for p in problems)


def test_validate_duplicate_member():
    problems = validate_condition(
        "all of hook", {"hook": ["emotionally_engaging", "emotionally_engaging"]}
    )
    assert any("lists a CE twice" in p for p in problems)
