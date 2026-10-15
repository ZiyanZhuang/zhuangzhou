"""Failure semantics shared by skill consumers; success is not scientific validation."""


def result_error(result):
    """Return a diagnostic for explicit execution failure, otherwise None."""
    if not isinstance(result, dict):
        return None
    if result.get("error"):
        return str(result["error"])
    if result.get("success") is False or result.get("status") in (
        "failed", "error", "cancelled", "timed_out",
    ):
        return str(result.get("stderr") or result.get("message") or "Skill reported failure")
    return None
