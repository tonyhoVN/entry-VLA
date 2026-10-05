"""Outcome labels and failure taxonomy. Transit and manipulation failures stay separate."""

from __future__ import annotations

OUTCOMES = (
    "success",
    "manipulation_timeout",        # policy ran to the step cap without success
    "manipulation_invalid_state",  # e.g. object dropped / out of workspace (predicate-based early stop)
    "manipulation_collision",      # clearance/collision check violated during the skill
    "transit_planning_failure",    # no valid preparation plan (IK / clearance)
    "transit_execution_failure",   # plan executed but did not arrive / stop
    "rejected_pre_travel",         # selector/scorer rejected every candidate before travel
    "rejected_at_arrival",         # arrival verifier rejected the reached state
    "reset_rejected",              # scene reset failed validation
    "intervention",                # any labelled diagnostic intervention (teleport, manual)
    "error",                       # software error (counted, never dropped)
)

TRANSIT = {"transit_planning_failure", "transit_execution_failure"}
MANIPULATION = {"manipulation_timeout", "manipulation_invalid_state", "manipulation_collision"}
REJECTION = {"rejected_pre_travel", "rejected_at_arrival"}


def classify(handoff_failure: str | None, rollout_reason: str | None, success: bool, rejected: str | None = None,
             intervention: bool = False) -> str:
    if intervention:
        return "intervention"
    if rejected:
        return rejected
    if handoff_failure:
        return "transit_planning_failure" if handoff_failure.startswith("planning") else "transit_execution_failure"
    if success:
        return "success"
    if rollout_reason is None:
        return "error"
    if rollout_reason.startswith("invalid_state:collision"):
        return "manipulation_collision"
    if rollout_reason.startswith("invalid_state"):
        return "manipulation_invalid_state"
    return "manipulation_timeout"
