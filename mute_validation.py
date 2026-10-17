"""
mute_validation.py
==================

Deterministic validation + structured abstention for Mute.

The probe in the notebook gives a *probabilistic* risk score ("how likely is the
answer missing?"). This module adds a *deterministic* safety net that runs after
the model has produced a candidate answer:

    1. Define a schema for each field: what a valid answer must look like.
    2. Normalize + validate the candidate answer against that schema.
    3. If any check fails, abstain with a specific reason instead of returning
       a plausible-but-wrong value.

The two layers are complementary:

    probe        -> "is the information likely absent?"        (ranking, uncertain)
    validation   -> "does this answer satisfy the contract?"   (deterministic, certain)

This is the "fruit tree" half of the project: a boundary-checked, auditable
pipeline around an imperfect model. No torch, no transformers, no third-party
deps — pure stdlib, so it is cheap, portable, and independently testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

class FieldKind(str, Enum):
    """What kind of value a field expects. Determines which validator runs."""
    MONEY = "money"    # "450 EUR", "$1,200", "€1,950", "600"
    DATE = "date"      # "2026-10-17"
    INT = "int"        # "5", "42"
    ENUM = "enum"      # one of a fixed set of values
    SPAN = "span"      # the answer must appear verbatim in the source


class AbstainReason(str, Enum):
    """Why we refused to return an answer. Structured, machine-readable."""
    NO_ANSWER = "NO_ANSWER"                    # model produced nothing usable
    NOT_A_NUMBER = "NOT_A_NUMBER"              # expected a number, got none
    OUT_OF_ENUM = "OUT_OF_ENUM"                # value not in the allowed set
    NOT_IN_SOURCE = "NOT_IN_SOURCE"            # span not verbatim in the context
    AMBIGUOUS = "AMBIGUOUS"                    # multiple competing values
    WRONG_CURRENCY = "WRONG_CURRENCY"          # amount present but wrong unit


@dataclass
class FieldSchema:
    """The contract a field must satisfy.

    Examples:
        FieldSchema("deposit", FieldKind.MONEY, "What is the deposit?", currency="EUR")
        FieldSchema("status", FieldKind.ENUM, "What is the status?", allowed=("open", "closed"))
        FieldSchema("amount_due", FieldKind.MONEY, "What is the amount due?")
    """
    name: str
    kind: FieldKind
    question: str
    currency: Optional[str] = None          # for MONEY: expected unit ("EUR", "USD", None = any)
    allowed: Optional[tuple[str, ...]] = None   # for ENUM: the closed set


@dataclass
class Result:
    """Outcome of the guarded extract: either a validated value or a refusal."""
    status: str                 # "PASS" or one of AbstainReason
    value: object = None        # the normalized value on PASS, else None
    raw: Optional[str] = None   # the model's original output, for auditability
    reason: Optional[str] = None  # human-readable detail on refusal

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def abstained(self) -> bool:
        return not self.passed

    def to_dict(self) -> dict:
        """JSON-ready dict for an external API."""
        return {"status": self.status, "value": self.value,
                "raw": self.raw, "reason": self.reason}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Parsers (normalize raw text into a typed value)
# --------------------------------------------------------------------------- #

# Wrapped in (?:...) so its internal `|` does not leak into any regex that
# concatenates _NUM (e.g. _MONEY_RE, lead/trail below).
_NUM = r"(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"

# "amount currency" or "currency amount" (e.g. "$1,200", "450 EUR", "€1,950")
_MONEY_RE = re.compile(
    r"(?:\$|€|USD|EUR)\s*" + _NUM + r"|" + _NUM + r"\s*(?:USD|EUR|€|\$|dollars|euros)",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_INT_RE = re.compile(r"(-?\d[\d,]*)")

_CURRENCY_NORM = {
    "$": "USD", "€": "EUR", "usd": "USD", "eur": "EUR",
    "dollars": "USD", "euros": "EUR", "dollar": "USD", "euro": "EUR",
}


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _normalize_currency(c: str) -> str:
    return _CURRENCY_NORM.get(c.strip().lower(), c.strip().upper())


def parse_money(text: str) -> Optional[tuple[float, str]]:
    """Return the single well-formed money amount in `text` as (value, currency).

    Returns None if there is no amount, or if there are several competing amounts
    (ambiguity is a refusal, not a guess).
    """
    matches = list(_MONEY_RE.finditer(text))
    if not matches:
        return None
    if len(matches) > 1:
        # more than one candidate amount -> refuse rather than pick one
        raise AmbiguousValue("multiple money amounts in answer")
    m = matches[0]
    raw = m.group(0)
    # split leading currency vs trailing currency
    lead = re.match(r"(?:\$|€|USD|EUR)\s*(" + _NUM + r")", raw, re.IGNORECASE)
    if lead:
        cur = _normalize_currency(raw[: lead.start(1)].strip())
        return _to_float(lead.group(1)), cur
    trail = re.match(r"(" + _NUM + r")\s*(?:USD|EUR|€|\$|dollars|euros)", raw, re.IGNORECASE)
    cur = _normalize_currency(raw[trail.end(1):].strip())
    return _to_float(trail.group(1)), cur


class AmbiguousValue(Exception):
    """Raised internally when an answer has several competing values."""


def parse_int(text: str) -> Optional[int]:
    m = _INT_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_date(text: str) -> Optional[str]:
    m = _DATE_RE.search(text)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

def _validate_money(field: FieldSchema, context: str, answer: str) -> Result:
    try:
        parsed = parse_money(answer)
    except AmbiguousValue:
        return Result(AbstainReason.AMBIGUOUS.value, raw=answer,
                      reason="answer contains multiple amounts; refusing to guess")
    if parsed is None:
        return Result(AbstainReason.NOT_A_NUMBER.value, raw=answer,
                      reason=f"expected a money amount for '{field.name}', got '{answer}'")
    value, currency = parsed
    if field.currency and currency and currency != field.currency.upper():
        return Result(AbstainReason.WRONG_CURRENCY.value, raw=answer,
                      reason=f"got {currency}, expected {field.currency}")
    return Result("PASS", value=value, raw=answer)


def _validate_int(field: FieldSchema, context: str, answer: str) -> Result:
    value = parse_int(answer)
    if value is None:
        return Result(AbstainReason.NOT_A_NUMBER.value, raw=answer,
                      reason=f"expected an integer for '{field.name}', got '{answer}'")
    return Result("PASS", value=value, raw=answer)


def _validate_date(field: FieldSchema, context: str, answer: str) -> Result:
    value = parse_date(answer)
    if value is None:
        return Result(AbstainReason.NOT_A_NUMBER.value, raw=answer,
                      reason=f"expected an ISO date (YYYY-MM-DD) for '{field.name}', got '{answer}'")
    return Result("PASS", value=value, raw=answer)


def _validate_enum(field: FieldSchema, context: str, answer: str) -> Result:
    norm = answer.strip().lower().rstrip(".")
    if field.allowed is None:
        raise ValueError(f"ENUM field '{field.name}' has no `allowed` set")
    if norm not in {a.lower() for a in field.allowed}:
        return Result(AbstainReason.OUT_OF_ENUM.value, raw=answer,
                      reason=f"'{answer}' not in allowed {list(field.allowed)}")
    return Result("PASS", value=norm, raw=answer)


def _validate_span(field: FieldSchema, context: str, answer: str) -> Result:
    # A "span" answer is only trustworthy if it appears verbatim in the source
    # exactly once (a repeated span could be the model quoting the wrong sentence).
    a = answer.strip()
    if len(a) < 2:
        return Result(AbstainReason.NOT_IN_SOURCE.value, raw=answer,
                      reason="span too short to verify")
    start = context.find(a)
    if start == -1:
        return Result(AbstainReason.NOT_IN_SOURCE.value, raw=answer,
                      reason="answer is not a verbatim substring of the source")
    if context.find(a, start + 1) != -1:
        return Result(AbstainReason.AMBIGUOUS.value, raw=answer,
                      reason="span appears multiple times in the source; which one is the answer?")
    return Result("PASS", value=a, raw=answer)


_VALIDATORS = {
    FieldKind.MONEY: _validate_money,
    FieldKind.INT: _validate_int,
    FieldKind.DATE: _validate_date,
    FieldKind.ENUM: _validate_enum,
    FieldKind.SPAN: _validate_span,
}


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #

def validate_answer(field: FieldSchema, context: str, answer: Optional[str]) -> Result:
    """Run the deterministic gate on a candidate answer.

    `answer` may be None (the model produced nothing), in which case we abstain.
    """
    if answer is None or not answer.strip():
        return Result(AbstainReason.NO_ANSWER.value,
                      reason=f"no answer produced for '{field.name}'")
    validator = _VALIDATORS[field.kind]
    return validator(field, context, answer.strip())


def guarded_extract(
    field: FieldSchema,
    context: str,
    risk_score: float,
    answer_provider: Callable[[], Optional[str]],
    threshold: float = 0.60,
) -> Result:
    """The full minimal closed loop: probe gate -> generate -> validate -> abstain.

    Args:
        field:          the schema contract for this field.
        context:        the source text the answer must be grounded in.
        risk_score:     the probe's P(missing-info), from the notebook pipeline.
        answer_provider: zero-arg callable that returns the raw model answer
                         (or None). Kept separate so this module stays torch-free.
        threshold:      probe score above which we refuse to even generate.

    Returns a Result. On any refusal, `result.reason` says exactly why, so the
    caller can log an auditable abstention instead of a silent wrong answer.
    """
    if risk_score >= threshold:
        return Result(AbstainReason.NO_ANSWER.value,
                      reason=f"probe risk {risk_score:.3f} >= threshold {threshold}")
    raw = answer_provider()
    return validate_answer(field, context, raw)


# --------------------------------------------------------------------------- #
# Demo / smoke test
# --------------------------------------------------------------------------- #

def _demo() -> None:
    money = FieldSchema("deposit", FieldKind.MONEY, "What is the deposit?", currency="EUR")
    status = FieldSchema("status", FieldKind.ENUM, "What is the status?",
                         allowed=("open", "closed", "pending"))
    ref = FieldSchema("customer", FieldKind.SPAN, "Who is the customer?")
    d = FieldSchema("deadline", FieldKind.DATE, "What is the deadline?")

    cases = [
        (money, "Deposit is 350 EUR, due Thursday.", "350 EUR", "valid money"),
        (money, "Deposit is 350 EUR, due Thursday.", "450 EUR", "wrong amount but well-formed (passes)"),
        (money, "Deposit is 350 EUR, due Thursday.", "$350", "wrong currency"),
        (money, "Deposit is 350 EUR, due Thursday.", "350 EUR and 500 EUR", "ambiguous"),
        (money, "Deposit is 350 EUR, due Thursday.", "I don't know", "not a number"),
        (status, "Status: open", "open", "valid enum"),
        (status, "Status: open", "active", "out of enum"),
        (ref, "Customer is Acme Corp.", "Acme Corp.", "verbatim span"),
        (ref, "Customer is Acme Corp.", "ACME", "not verbatim"),
        (ref, "Acme Corp (customer) contracted Acme Corp (vendor).", "Acme Corp", "ambiguous span"),
        (d, "Due 2026-10-17", "2026-10-17", "valid date"),
        (d, "Due 2026-10-17", "Oct 17", "not ISO"),
    ]

    print("=== validate_answer (deterministic gate) ===")
    for field, ctx, ans, label in cases:
        r = validate_answer(field, ctx, ans)
        mark = "PASS " if r.passed else "ABST"
        print(f"  [{mark}] {label:<34} | {field.name:<9} | answer='{ans}' -> {r.status}"
              + (f" ({r.value})" if r.passed else f" ({r.reason})"))

    # Full loop demo with a fake generator (no model needed).
    print("\n=== guarded_extract (full loop) ===")
    def gen_350():
        return "350 EUR"

    def gen_none():
        return None

    for risk, provider, label in [
        (0.95, gen_350, "probe blocks (high risk)"),
        (0.10, gen_350, "probe passes -> validated value"),
        (0.10, gen_none, "probe passes -> model returns None"),
    ]:
        r = guarded_extract(money, "Deposit is 350 EUR.", risk, provider)
        print(f"  risk={risk:.2f} | {label:<30} -> {r.status}"
              + (f" value={r.value}" if r.passed else f" ({r.reason})"))


if __name__ == "__main__":
    _demo()
