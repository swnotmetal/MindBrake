"""
mute_policy.py — the runtime shell: turn a refusal into an action, then report it.

mute_validation.guarded_extract tells you WHAT happened (PASS or an AbstainReason).
This module adds the two things a real constrained system needs on top of that:

1. decide(result) -> Action
   Maps every refusal to an Action: PASS / BLOCK / ESCALATE / RETRY.
   This is the "escalation path" — who or what handles each failure mode.

2. Reporter
   Accumulates every decision and prints an honest end-of-run summary
   (total, per-action counts, per-refusal-reason counts, pass rate).

`route()` is the single runtime entry point: it wraps guarded_extract, applies the
policy, and records the decision. The `threshold` argument is the calibrated
circuit-breaker value from run_eval's FPR-capped operating points — not a guess.

Pure stdlib (plus mute_validation). No torch.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from mute_validation import FieldSchema, Result, guarded_extract


class Action(str, Enum):
    PASS = "PASS"          # return the validated value to the caller
    BLOCK = "BLOCK"        # abstain: return a clear refusal, no value
    ESCALATE = "ESCALATE"  # route to a human / stronger model for judgment
    RETRY = "RETRY"        # re-run with more context before deciding


# Default refusal -> action. Override per deployment. The safe default is BLOCK;
# AMBIGUOUS is escalated because "multiple competing values" needs judgment.
DEFAULT_POLICY = {
    "NO_ANSWER": Action.BLOCK,
    "NOT_A_NUMBER": Action.BLOCK,
    "OUT_OF_ENUM": Action.BLOCK,
    "NOT_IN_SOURCE": Action.BLOCK,
    "WRONG_CURRENCY": Action.BLOCK,
    "AMBIGUOUS": Action.ESCALATE,
}


def decide(result: Result, policy: Optional[dict] = None) -> Action:
    """Map a validation Result to an Action. Unknown refusal -> ESCALATE (fail safe)."""
    if result.passed:
        return Action.PASS
    return (policy or DEFAULT_POLICY).get(result.status, Action.ESCALATE)


class Reporter:
    """Accumulates decisions for an honest end-of-run report."""

    def __init__(self) -> None:
        self._action: Counter = Counter()
        self._reason: Counter = Counter()
        self._total = 0

    def record(self, result: Result, action: Action) -> None:
        self._total += 1
        self._action[action.value] += 1
        if result.abstained:
            self._reason[result.status] += 1

    def summary(self) -> dict:
        if self._total == 0:
            return {"total": 0}
        return {
            "total": self._total,
            "actions": dict(self._action),
            "refusal_reasons": dict(self._reason),
            "pass_rate": self._action.get(Action.PASS.value, 0) / self._total,
        }

    def report(self) -> str:
        s = self.summary()
        if s.get("total", 0) == 0:
            return "no decisions recorded"
        lines = [f"total={s['total']} | pass_rate={s['pass_rate']:.2f}"]
        lines.append("actions: " + ", ".join(f"{k}={v}" for k, v in s["actions"].items()))
        if s["refusal_reasons"]:
            lines.append("refusals: " + ", ".join(f"{k}={v}" for k, v in s["refusal_reasons"].items()))
        return "\n".join(lines)


def route(
    field: FieldSchema,
    context: str,
    risk: float,
    answer_provider: Callable[[], Optional[str]],
    threshold: float = 0.60,
    policy: Optional[dict] = None,
    reporter: Optional[Reporter] = None,
):
    """Probe gate -> generate -> validate -> decide action -> record.

    Returns (Result, Action). `threshold` is the calibrated circuit-breaker value.
    """
    result = guarded_extract(field, context, risk, answer_provider, threshold)
    action = decide(result, policy)
    if reporter is not None:
        reporter.record(result, action)
    return result, action


# --------------------------------------------------------------------------- #
# Config persistence (the calibrated threshold, not a hardcoded guess)
# --------------------------------------------------------------------------- #

def save_config(path: str, threshold: float, **meta) -> None:
    """Persist the calibrated threshold (+ metadata) for the runtime to load."""
    Path(path).write_text(json.dumps({"threshold": threshold, **meta}, indent=2))


def load_threshold(path: str, default: float = 0.60) -> float:
    """Read the calibrated threshold from `path`; fall back to `default` if absent."""
    try:
        return json.loads(Path(path).read_text()).get("threshold", default)
    except (FileNotFoundError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Escalation + retry handlers (the "actions" behind the policy)
# --------------------------------------------------------------------------- #

class EscalationSink:
    """Route escalated refusals to a target: a JSONL file (default) or a callback.

    Examples:
        sink = EscalationSink()                            # -> escalations.jsonl
        sink = EscalationSink(path="review.jsonl")         # -> custom file
        sink = EscalationSink(send=lambda r: post(r))      # -> custom target (queue/API)
    """

    def __init__(self, path: str = "escalations.jsonl",
                 send: Optional[Callable[[dict], None]] = None):
        self.path = path
        self._send = send

    def submit(self, result: Result, context: Optional[str] = None,
               question: Optional[str] = None) -> dict:
        record = {"status": result.status, "reason": result.reason, "raw": result.raw,
                  "context": context, "question": question}
        if self._send is not None:
            self._send(record)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def escalate(result: Result, path: str = "escalations.jsonl",
             context: Optional[str] = None, question: Optional[str] = None) -> dict:
    """Convenience wrapper over EscalationSink (default JSONL audit trail)."""
    return EscalationSink(path=path).submit(result, context, question)


def retry(field: FieldSchema, context: str, risk: float,
          answer_provider: Callable[[], Optional[str]], threshold: float,
          attempts: int = 1, reporter: Optional[Reporter] = None):
    """Re-run the generate+validate loop up to `attempts` more times; return the
    first PASS, else the last (result, action). Only meaningful when the generator
    is non-deterministic (temperature > 0) or a fallback provider is swapped in."""
    result = action = None
    for _ in range(attempts + 1):
        result, action = route(field, context, risk, answer_provider, threshold, reporter=reporter)
        if action is Action.PASS:
            break
    return result, action


def _demo() -> None:
    import os
    import tempfile

    from mute_validation import FieldKind

    money = FieldSchema("deposit", FieldKind.MONEY, "What is the deposit?", currency="EUR")
    reporter = Reporter()

    # grounded -> PASS | missing -> BLOCK | ambiguous -> ESCALATE (all deterministic, no model)
    r1, a1 = route(money, "deposit is 890 EUR.", 0.1, lambda: "890 EUR", reporter=reporter)
    r2, a2 = route(money, "no deposit mentioned.", 0.9, lambda: None, reporter=reporter)
    r3, a3 = route(money, "deposit is 300 EUR or 500 EUR.", 0.1,
                   lambda: "300 EUR or 500 EUR", reporter=reporter)

    for r, a in [(r1, a1), (r2, a2), (r3, a3)]:
        print(f"action={a.value:<8} | {r.to_json()}")
    print("\n" + reporter.report())

    # config persistence roundtrip
    tmp = tempfile.gettempdir()
    cfg = os.path.join(tmp, "mute_cfg_demo.json")
    save_config(cfg, 0.55, layer=14, dataset="demo")
    print("\nload_threshold roundtrip:", load_threshold(cfg),
          "| fallback:", load_threshold(cfg + ".missing"))

    # escalation sink (audit trail)
    print("escalated:", escalate(r3, path=os.path.join(tmp, "mute_esc_demo.jsonl"), context="demo"))

    # pluggable escalation: inject a callback instead of writing a file
    captured = []
    EscalationSink(send=captured.append).submit(r3, context="demo")
    print("escalation callback captured:", captured)

    # retry: a provider that fails once (None) then succeeds
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        return "890 EUR" if state["n"] > 1 else None

    rr, ra = retry(money, "deposit is 890 EUR.", 0.1, flaky, 0.6, attempts=1, reporter=reporter)
    print(f"retry -> action={ra.value} value={rr.value} (attempts={state['n']})")


if __name__ == "__main__":
    _demo()
