"""The gavel-rules condition grammar: parser, group extraction, evaluation,
and the derived readable-predicate renderer.

Ported from gavel-rules tools/condition.py
(Offensive-AI-Lab/gavel-rules @ 8ac552c, schema v1).
Keep the parser/evaluator semantics byte-compatible with upstream — the same
condition string must accept/reject and fire identically here and in the
library's CI validator. Deliberately dependency-free.

The parser accepts `not` (it is a reserved keyword) but validation rejects
it, exactly like the library CI: negation stays unavailable until its
calibration semantics are defined in a future schema version.

    condition   ::= disjunction
    disjunction ::= conjunction ( "or" conjunction )*
    conjunction ::= negation ( "and" negation )*
    negation    ::= "not" negation | atom
    atom        ::= selector | "(" condition ")"
    selector    ::= ( "all" | INTEGER ) "of" GROUP_NAME
"""

from __future__ import annotations

import re

KEYWORDS = {"all", "of", "and", "or", "not"}
GROUP_NAME_RE = re.compile(r"[a-z][a-z0-9_]*$")
# Positional / non-descriptive group names the lint rejects.
LAZY_GROUP_NAME_RE = re.compile(r"(g|grp|group|sel|selection)_?\d*$")

class ConditionError(ValueError):
    pass


def tokenize(condition: str) -> list[str]:
    tokens = condition.replace("(", " ( ").replace(")", " ) ").split()
    for t in tokens:
        if t in KEYWORDS or t in "()" or t.isdigit() or GROUP_NAME_RE.match(t):
            continue
        raise ConditionError(f"bad token {t!r}")
    return tokens


def parse(condition: str):
    """Parse into an AST of ('or'|'and', [children]) / ('not', child) /
    ('sel', k_or_'all', group_name). Raises ConditionError."""
    tokens = tokenize(condition)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        if pos >= len(tokens):
            raise ConditionError(f"unexpected end of condition {condition!r}")
        tok = tokens[pos]
        if expected is not None and tok != expected:
            raise ConditionError(f"expected {expected!r}, got {tok!r}")
        pos += 1
        return tok

    def disjunction():
        children = [conjunction()]
        while peek() == "or":
            take()
            children.append(conjunction())
        return children[0] if len(children) == 1 else ("or", children)

    def conjunction():
        children = [negation()]
        while peek() == "and":
            take()
            children.append(negation())
        return children[0] if len(children) == 1 else ("and", children)

    def negation():
        if peek() == "not":
            take()
            return ("not", negation())
        return atom()

    def atom():
        if peek() == "(":
            take()
            node = disjunction()
            take(")")
            return node
        return selector()

    def selector():
        quant = take()
        if quant != "all" and not quant.isdigit():
            raise ConditionError(f"expected quantifier, got {quant!r}")
        take("of")
        group = take()
        if group in KEYWORDS or not GROUP_NAME_RE.match(group):
            raise ConditionError(f"bad group name {group!r}")
        return ("sel", quant if quant == "all" else int(quant), group)

    ast = disjunction()
    if pos != len(tokens):
        raise ConditionError(f"trailing tokens in {condition!r}")
    return ast


def referenced_groups(ast, acc: set[str] | None = None) -> set[str]:
    acc = set() if acc is None else acc
    if ast[0] in ("or", "and"):
        for c in ast[1]:
            referenced_groups(c, acc)
    elif ast[0] == "not":
        referenced_groups(ast[1], acc)
    else:
        acc.add(ast[2])
    return acc


def selectors(ast, acc=None) -> list[tuple]:
    """All ('sel', quant, group) nodes in the AST."""
    acc = [] if acc is None else acc
    if ast[0] in ("or", "and"):
        for c in ast[1]:
            selectors(c, acc)
    elif ast[0] == "not":
        selectors(ast[1], acc)
    else:
        acc.append(ast)
    return acc


def contains_not(ast) -> bool:
    if ast[0] == "not":
        return True
    if ast[0] in ("or", "and"):
        return any(contains_not(c) for c in ast[1])
    return False


def evaluate(ast, groups: dict[str, list[str]], detected: frozenset[str]) -> bool:
    kind = ast[0]
    if kind == "or":
        return any(evaluate(c, groups, detected) for c in ast[1])
    if kind == "and":
        return all(evaluate(c, groups, detected) for c in ast[1])
    if kind == "not":
        return not evaluate(ast[1], groups, detected)
    _, quant, name = ast
    members = groups[name]
    k = len(members) if quant == "all" else quant
    return sum(m in detected for m in members) >= k


def render_predicate(ast, groups: dict[str, list[str]]) -> str:
    """Expand groups into the readable predicate stored in index.json:
    1 of g -> (a OR b), all of g -> (a AND b), k of g -> at least k of
    (a, b, c), not -> NOT. Single-member selectors render as the bare CE."""

    def rec(node):
        kind = node[0]
        if kind in ("or", "and"):
            joint = f" {kind.upper()} "
            parts = []
            for c in node[1]:
                s = rec(c)
                if c[0] in ("or", "and") and c[0] != kind:
                    s = f"({s})"
                parts.append(s)
            return joint.join(parts)
        if kind == "not":
            inner = rec(node[1])
            if node[1][0] in ("or", "and"):
                inner = f"({inner})"
            return f"NOT {inner}"
        _, quant, name = node
        members = groups[name]
        if len(members) == 1:
            return members[0]
        if quant == "all" or quant == len(members):
            return "(" + " AND ".join(members) + ")"
        if quant == 1:
            return "(" + " OR ".join(members) + ")"
        return f"at least {quant} of (" + ", ".join(members) + ")"

    return rec(ast)


# --- Studio-side additions (not in upstream tools/condition.py) --------------

def validate_condition(condition: str, groups: dict[str, list[str]]) -> list[str]:
    """Sanity check for a (condition, groups) pair — used by the rule editor,
    the AI pipeline and the library importer.

    Returns the problems as human-readable strings; an empty list means the
    pair is valid. Enforces the same invariants as the library's CI validator
    (tools/validate.py), so a rule that saves here is publishable to
    gavel-rules and records that pass CI pass here too: parseable grammar, no
    negation, no undefined or dead groups, satisfiable quantifiers, the
    `all of` lint, and descriptive group names.
    """
    problems: list[str] = []
    try:
        ast = parse(condition)
    except ConditionError as e:
        return [f"condition does not parse: {e}"]

    if contains_not(ast):
        problems.append(
            "condition uses 'not' — negation is not supported in rule conditions"
        )
    referenced = referenced_groups(ast)
    for name in sorted(referenced):
        if name not in groups:
            problems.append(f"condition references undefined group '{name}'")
    for _, quant, name in selectors(ast):
        members = groups.get(name)
        if members is None or not isinstance(quant, int):
            continue
        if quant < 1 or quant > len(members):
            problems.append(
                f"'{quant} of {name}' can never be satisfied "
                f"({name} has {len(members)} member(s))"
            )
        elif quant == len(members) and quant > 1:
            problems.append(
                f"'{quant} of {name}' equals the group size — write 'all of {name}'"
            )
    for name, members in groups.items():
        if not members:
            problems.append(f"group '{name}' has no members")
        elif len(members) != len(set(members)):
            problems.append(f"group '{name}' lists a CE twice")
        if name not in referenced:
            problems.append(
                f"group '{name}' is not referenced by the condition — "
                f"use it or remove the group"
            )
        if LAZY_GROUP_NAME_RE.match(name):
            problems.append(
                f"group name '{name}' is not descriptive — name the role the "
                f"group plays in the detection, not its position"
            )
    return problems


def evaluate_condition(condition: str, groups: dict[str, list[str]], detected) -> bool:
    """Parse-and-evaluate convenience for callers that hold the condition as
    text (the DB stores condition text as the source of truth)."""
    return evaluate(parse(condition), groups, frozenset(detected))
