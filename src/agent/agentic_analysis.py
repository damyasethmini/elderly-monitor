"""Step 5: local, evidence-grounded agentic analysis for elderly monitoring.

The agent is implemented as a LangGraph StateGraph when LangGraph is installed.
It is intentionally local and deterministic: there is no hosted LLM or paid API.
The graph decides whether more temporal context is needed before checking spatial
and movement evidence, then produces a safety-oriented NORMAL/MONITOR/ALERT result.

The public ``analyze_timeline_and_events`` function keeps the Step 5 API used by
later steps and the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback keeps local tests runnable
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_BED_ACTIVITIES = {"LYING_IN_BED", "SITTING_ON_BED"}
OUT_OF_BED_ACTIVITIES = {"SITTING_OUTSIDE_BED", "STANDING", "WALKING"}
VALID_DECISIONS = {"NORMAL", "MONITOR", "ALERT"}
DECISION_RANK = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}

DEFAULT_ALERTS: dict[str, float] = {
    "sitting_on_bed_edge_monitor_sec": 120.0,
    "prolonged_out_of_bed_monitor_sec": 300.0,
    "prolonged_out_of_bed_alert_sec": 600.0,
    "unknown_state_monitor_sec": 30.0,
}


class EventAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes for one bed event."""

    event: dict[str, Any]
    segments: list[dict[str, Any]]
    alerts: dict[str, float]
    movement_threshold: float
    event_index: int | None
    previous: dict[str, Any] | None
    current: dict[str, Any] | None
    following: list[dict[str, Any]]
    context_segments: list[dict[str, Any]]
    previous_activity: str | None
    current_activity: str
    following_activities: list[str]
    previous_in_bed: bool
    has_following_out_of_bed: bool
    context_is_ambiguous: bool
    spatial_known: bool
    inside_seen: bool
    outside_seen: bool
    movement_norm: float
    movement_supports_exit: bool
    actions: list[dict[str, Any]]
    reasons: list[str]
    decision: str
    context_support: str
    requires_human_review: bool
    confidence: float | None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _decision(value: Any, fallback: str = "NORMAL") -> str:
    candidate = str(value or fallback).upper()
    return candidate if candidate in VALID_DECISIONS else fallback


def _segment_activity(segment: dict[str, Any]) -> str:
    return str(segment.get("activity", "UNKNOWN")).upper()


def _segment_at_time(segments: list[dict[str, Any]], timestamp: float) -> int | None:
    """Find the timeline segment containing a timestamp; tolerate endpoint rounding."""
    for index, segment in enumerate(segments):
        start = _number(segment.get("start_sec"), -1.0)
        end = _number(segment.get("end_sec"), -1.0)
        if start <= timestamp < end:
            return index
    if segments and abs(timestamp - _number(segments[-1].get("end_sec"), -2.0)) < 1e-6:
        return len(segments) - 1
    return None


def _duration_of_contiguous_activity(
    segments: list[dict[str, Any]], activities: set[str]
) -> float:
    """Return the longest consecutive run whose activity is in activities."""
    longest = 0.0
    current = 0.0
    for segment in segments:
        if _segment_activity(segment) in activities:
            duration = max(0.0, _number(segment.get("duration_sec")))
            if duration == 0.0:
                duration = max(
                    0.0,
                    _number(segment.get("end_sec")) - _number(segment.get("start_sec")),
                )
            current += duration
            longest = max(longest, current)
        else:
            current = 0.0
    return round(longest, 3)


def _context_snapshot(segment: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "activity",
        "bed_occupancy",
        "start_sec",
        "end_sec",
        "duration_sec",
        "mean_confidence",
        "bed_region_inside_at_start",
        "bed_region_inside_at_end",
        "bed_region_contains_outside_point",
        "movement_away_from_bed_norm",
        "max_distance_from_bed_px",
    )
    return {key: segment.get(key) for key in keys if key in segment}


def _spatial_evidence(segments: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    """Return (known, inside_bed_seen, outside_bed_seen)."""
    inside_seen = False
    outside_seen = False
    known = False
    for segment in segments:
        for key in ("bed_region_inside_at_start", "bed_region_inside_at_end"):
            value = segment.get(key)
            if isinstance(value, bool):
                known = True
                inside_seen = inside_seen or value
                outside_seen = outside_seen or (not value)
        if segment.get("bed_region_contains_outside_point") is True:
            known = True
            outside_seen = True
    return known, inside_seen, outside_seen


def _make_action(action: str, status: str, finding: str, **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "action": action,
        "status": status,
        "finding": finding,
    }
    if details:
        item["details"] = details
    return item


def _node_locate_event(state: EventAgentState) -> dict[str, Any]:
    event = state["event"]
    segments = state["segments"]
    event_time = _number(
        event.get("start_time_sec"),
        _number(event.get("confirmed_time_sec")),
    )
    event_index = _segment_at_time(segments, event_time)
    actions = list(state.get("actions", []))
    if event_index is None:
        actions.append(
            _make_action(
                "LOCATE_EVENT_IN_TIMELINE",
                "UNCERTAIN",
                "The event timestamp did not match any timeline segment.",
                event_time_sec=event_time,
            )
        )
    else:
        actions.append(
            _make_action(
                "LOCATE_EVENT_IN_TIMELINE",
                "FOUND",
                f"Event aligned to timeline segment {event_index}.",
                event_time_sec=event_time,
                segment_index=event_index,
            )
        )
    return {"event_index": event_index, "actions": actions}


def _node_inspect_temporal_context(state: EventAgentState) -> dict[str, Any]:
    event = state["event"]
    segments = state["segments"]
    event_index = state.get("event_index")
    actions = list(state.get("actions", []))

    if event_index is None:
        return {
            "previous": None,
            "current": None,
            "following": [],
            "context_segments": [],
            "previous_activity": None,
            "current_activity": "UNKNOWN",
            "following_activities": [],
            "previous_in_bed": False,
            "has_following_out_of_bed": False,
            "context_is_ambiguous": True,
            "actions": actions,
        }

    previous = segments[event_index - 1] if event_index > 0 else None
    current = segments[event_index]
    nearby_end = min(len(segments), event_index + 3)
    nearby = segments[max(0, event_index - 1) : nearby_end]
    following = segments[event_index:nearby_end]

    previous_activity = _segment_activity(previous) if previous else None
    current_activity = _segment_activity(current)
    following_activities = [_segment_activity(item) for item in following]
    has_following_out_of_bed = any(
        activity in OUT_OF_BED_ACTIVITIES for activity in following_activities
    )
    previous_in_bed = previous_activity in IN_BED_ACTIVITIES if previous_activity else False
    event_type = str(event.get("event_type", "UNKNOWN")).upper()

    context_is_ambiguous = (
        previous is None
        or previous_activity == "UNKNOWN"
        or current_activity == "UNKNOWN"
        or (not has_following_out_of_bed and event_type == "BED_EXIT")
        or event.get("evidence_strength") == "TRANSITION_ONLY"
    )

    actions.append(
        _make_action(
            "INSPECT_PREVIOUS_SEGMENT",
            "FOUND" if previous else "MISSING",
            (
                f"Previous activity was {previous_activity}."
                if previous_activity
                else "No previous timeline segment exists."
            ),
            previous_activity=previous_activity,
        )
    )
    actions.append(
        _make_action(
            "INSPECT_FOLLOWING_SEGMENTS",
            "FOUND" if following else "MISSING",
            "Following context: " + (", ".join(following_activities) or "none"),
            following_activities=following_activities,
        )
    )

    return {
        "previous": previous,
        "current": current,
        "following": following,
        "context_segments": nearby,
        "previous_activity": previous_activity,
        "current_activity": current_activity,
        "following_activities": following_activities,
        "previous_in_bed": previous_in_bed,
        "has_following_out_of_bed": has_following_out_of_bed,
        "context_is_ambiguous": context_is_ambiguous,
        "actions": actions,
    }


def _route_context(state: EventAgentState) -> Literal["expand", "spatial"]:
    if state.get("context_is_ambiguous", True):
        return "expand"
    return "spatial"


def _node_expand_temporal_context(state: EventAgentState) -> dict[str, Any]:
    segments = state["segments"]
    event_index = state.get("event_index")
    actions = list(state.get("actions", []))
    if event_index is None:
        return {"context_segments": [], "actions": actions}
    context_segments = segments[max(0, event_index - 3) : min(len(segments), event_index + 5)]
    actions.append(
        _make_action(
            "EXPAND_TEMPORAL_CONTEXT",
            "EXECUTED",
            "Initial event context was ambiguous; inspected a wider temporal window.",
            segment_count=len(context_segments),
            lookback_segments=3,
            lookahead_segments=4,
        )
    )
    return {"context_segments": context_segments, "actions": actions}


def _node_check_spatial_and_movement(state: EventAgentState) -> dict[str, Any]:
    event = state["event"]
    context_segments = state.get("context_segments", [])
    movement_threshold = state["movement_threshold"]
    spatial_known, inside_seen, outside_seen = _spatial_evidence(context_segments)
    movement_norm = _number(event.get("movement_away_from_bed_norm"), -1.0)
    movement_supports_exit = movement_norm >= movement_threshold

    actions = list(state.get("actions", []))
    status = "SUPPORTED" if outside_seen or movement_supports_exit else (
        "INSIDE_ONLY" if inside_seen else "INSUFFICIENT"
    )
    finding = (
        "Spatial or movement evidence supports movement away from the bed."
        if outside_seen or movement_supports_exit
        else "No positive evidence of movement away from the bed was found in the event context."
    )
    actions.append(
        _make_action(
            "CHECK_BED_REGION_AND_MOVEMENT",
            status,
            finding,
            spatial_evidence_known=spatial_known,
            bed_region_inside_seen=inside_seen,
            bed_region_outside_seen=outside_seen,
            movement_away_from_bed_norm=movement_norm if movement_norm >= 0 else None,
            movement_threshold=movement_threshold,
        )
    )

    return {
        "spatial_known": spatial_known,
        "inside_seen": inside_seen,
        "outside_seen": outside_seen,
        "movement_norm": movement_norm,
        "movement_supports_exit": movement_supports_exit,
        "actions": actions,
    }


def _node_decide_event(state: EventAgentState) -> dict[str, Any]:
    event = state["event"]
    event_type = str(event.get("event_type", "UNKNOWN")).upper()
    previous_activity = state.get("previous_activity")
    current_activity = state.get("current_activity", "UNKNOWN")
    following_activities = state.get("following_activities", [])
    previous_in_bed = bool(state.get("previous_in_bed"))
    has_following_out_of_bed = bool(state.get("has_following_out_of_bed"))
    inside_seen = bool(state.get("inside_seen"))
    outside_seen = bool(state.get("outside_seen"))
    spatial_known = bool(state.get("spatial_known"))
    movement_supports_exit = bool(state.get("movement_supports_exit"))
    alerts = state["alerts"]
    actions = list(state.get("actions", []))
    reasons: list[str] = []

    if state.get("event_index") is None:
        return {
            "decision": "MONITOR",
            "context_support": "WEAK",
            "requires_human_review": True,
            "reasons": ["Event could not be aligned to the activity timeline."],
            "actions": actions,
        }

    decision = _decision(event.get("decision"), "MONITOR")
    requires_review = False
    context_support = "STRONG"

    if event_type == "BED_EXIT":
        if previous_in_bed and has_following_out_of_bed and (outside_seen or movement_supports_exit):
            reasons.append(
                "Temporal sequence and spatial/movement evidence corroborate the bed exit."
            )
        else:
            context_support = "WEAK"
            requires_review = True
            decision = (
                "MONITOR"
                if DECISION_RANK[decision] < DECISION_RANK["MONITOR"]
                else decision
            )
            reasons.append(
                "The detected bed exit lacks one or more expected context signals; human review is recommended."
            )

        episode_duration = _number(event.get("out_of_bed_duration_sec"))
        if episode_duration >= alerts["prolonged_out_of_bed_alert_sec"]:
            decision = "ALERT"
            reasons.append(
                "The continuous out-of-bed episode meets the configured ALERT duration threshold."
            )
        elif episode_duration >= alerts["prolonged_out_of_bed_monitor_sec"]:
            decision = max(
                [decision, "MONITOR"], key=lambda item: DECISION_RANK[item]
            )
            reasons.append(
                "The continuous out-of-bed episode meets the configured MONITOR duration threshold."
            )
        elif DECISION_RANK[decision] < DECISION_RANK["MONITOR"]:
            decision = "MONITOR"
            reasons.append("A confirmed bed exit is recorded for monitoring.")

    elif event_type == "BED_RETURN":
        has_in_bed_after = any(
            _segment_activity(item) in IN_BED_ACTIVITIES for item in state.get("following", [])
        )
        if previous_activity in OUT_OF_BED_ACTIVITIES and has_in_bed_after and inside_seen:
            reasons.append(
                "The timeline supports a transition from outside the bed back to an in-bed posture."
            )
            context_support = "STRONG"
        else:
            decision = "MONITOR"
            context_support = "WEAK"
            requires_review = True
            reasons.append("Return context is incomplete or lacks positive in-bed spatial evidence.")

    else:
        decision = "MONITOR"
        context_support = "WEAK"
        requires_review = True
        reasons.append(f"Unsupported event type {event_type}; review the event manually.")

    if _decision(event.get("decision"), "NORMAL") == "ALERT":
        decision = "ALERT"
        reasons.append(
            "The upstream event detector already marked this event ALERT; escalation is preserved."
        )

    actions.append(
        _make_action(
            "MAKE_EVENT_DECISION",
            decision,
            "Applied the event-specific temporal, spatial, movement and duration rules.",
            decision=decision,
            context_support=context_support,
            requires_human_review=requires_review,
        )
    )

    confidence = (
        round(_number(event.get("confidence")), 3)
        if event.get("confidence") is not None
        else None
    )
    return {
        "decision": decision,
        "context_support": context_support,
        "requires_human_review": requires_review,
        "reasons": reasons,
        "actions": actions,
        "confidence": confidence,
    }


def _fallback_event_agent(state: EventAgentState) -> EventAgentState:
    """Run the same node logic sequentially if LangGraph is unavailable."""
    state = {**state, **_node_locate_event(state)}
    state = {**state, **_node_inspect_temporal_context(state)}
    if _route_context(state) == "expand":
        state = {**state, **_node_expand_temporal_context(state)}
    state = {**state, **_node_check_spatial_and_movement(state)}
    state = {**state, **_node_decide_event(state)}
    return state


def _build_event_agent_graph() -> Any | None:
    """Build and compile the LangGraph StateGraph used for one event."""
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return None

    builder = StateGraph(EventAgentState)
    builder.add_node("locate_event", _node_locate_event)
    builder.add_node("inspect_temporal_context", _node_inspect_temporal_context)
    builder.add_node("expand_temporal_context", _node_expand_temporal_context)
    builder.add_node("check_spatial_and_movement", _node_check_spatial_and_movement)
    builder.add_node("decide_event", _node_decide_event)

    builder.add_edge(START, "locate_event")
    builder.add_edge("locate_event", "inspect_temporal_context")
    builder.add_conditional_edges(
        "inspect_temporal_context",
        _route_context,
        {"expand": "expand_temporal_context", "spatial": "check_spatial_and_movement"},
    )
    builder.add_edge("expand_temporal_context", "check_spatial_and_movement")
    builder.add_edge("check_spatial_and_movement", "decide_event")
    builder.add_edge("decide_event", END)
    return builder.compile()


def _run_event_agent(
    event: dict[str, Any],
    segments: list[dict[str, Any]],
    alerts: dict[str, float],
    movement_threshold: float,
) -> dict[str, Any]:
    initial_state: EventAgentState = {
        "event": event,
        "segments": segments,
        "alerts": alerts,
        "movement_threshold": movement_threshold,
        "actions": [],
    }

    graph = _build_event_agent_graph()
    if graph is not None:
        final_state = graph.invoke(initial_state)
        framework = "langgraph_stategraph"
    else:
        final_state = _fallback_event_agent(initial_state)
        framework = "deterministic_fallback_without_langgraph"

    event_type = str(event.get("event_type", "UNKNOWN")).upper()
    event_id = str(event.get("event_id", "event_unknown"))
    event_time = _number(
        event.get("start_time_sec"),
        _number(event.get("confirmed_time_sec")),
    )
    confidence = final_state.get("confidence")
    if confidence is None and event.get("confidence") is not None:
        confidence = round(_number(event.get("confidence")), 3)

    return {
        "event_id": event_id,
        "event_type": event_type,
        "start_time_sec": round(event_time, 3),
        "confirmed_time_sec": event.get("confirmed_time_sec"),
        "decision": _decision(final_state.get("decision"), "MONITOR"),
        "context_support": final_state.get("context_support", "WEAK"),
        "confidence": confidence,
        "requires_human_review": bool(final_state.get("requires_human_review", True)),
        "reasons": list(final_state.get("reasons", [])),
        "findings": {
            "previous_activity": final_state.get("previous_activity"),
            "event_segment_activity": final_state.get("current_activity", "UNKNOWN"),
            "following_activities": list(final_state.get("following_activities", [])),
            "previous_activity_was_in_bed": bool(final_state.get("previous_in_bed", False)),
            "following_context_contains_out_of_bed_activity": bool(
                final_state.get("has_following_out_of_bed", False)
            ),
            "bed_region_spatial_evidence_known": bool(final_state.get("spatial_known", False)),
            "bed_region_inside_seen": bool(final_state.get("inside_seen", False)),
            "bed_region_outside_seen": bool(final_state.get("outside_seen", False)),
            "movement_away_from_bed_norm": (
                final_state.get("movement_norm")
                if _number(final_state.get("movement_norm"), -1.0) >= 0
                else None
            ),
            "movement_threshold": movement_threshold,
            "out_of_bed_duration_sec": _number(event.get("out_of_bed_duration_sec")),
        },
        "agent_actions": list(final_state.get("actions", [])),
        "context_segments": [
            _context_snapshot(item)
            for item in final_state.get("context_segments", [])
        ],
        "agent_framework": framework,
    }


def analyze_timeline_and_events(
    timeline: dict[str, Any],
    events_payload: dict[str, Any],
    alerts_config: dict[str, Any] | None = None,
    movement_threshold: float = 0.15,
) -> dict[str, Any]:
    """Run Step 5 agentic contextual checks and return a JSON-serializable report."""
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Timeline JSON must contain a list named 'segments'.")
    if not all(isinstance(item, dict) for item in segments):
        raise ValueError("Every timeline segment must be a JSON object.")

    alerts = dict(DEFAULT_ALERTS)
    if alerts_config:
        for key in alerts:
            if key in alerts_config:
                alerts[key] = max(0.0, _number(alerts_config[key], alerts[key]))
    if alerts["prolonged_out_of_bed_alert_sec"] < alerts["prolonged_out_of_bed_monitor_sec"]:
        alerts["prolonged_out_of_bed_alert_sec"] = alerts["prolonged_out_of_bed_monitor_sec"]

    source_events = events_payload.get("events", [])
    if not isinstance(source_events, list):
        raise ValueError("Bed events JSON must contain a list named 'events'.")

    event_analyses = [
        _run_event_agent(event, segments, alerts, movement_threshold)
        for event in source_events
        if isinstance(event, dict)
    ]

    longest_oob = _number(
        (events_payload.get("bed_summary") or {}).get("longest_out_of_bed_period_sec"),
        _duration_of_contiguous_activity(segments, OUT_OF_BED_ACTIVITIES),
    )
    longest_unknown = _duration_of_contiguous_activity(segments, {"UNKNOWN"})
    longest_sitting_on_bed = _duration_of_contiguous_activity(
        segments, {"SITTING_ON_BED"}
    )
    observation_duration = _number(
        timeline.get("observation_duration_sec"),
        _number(timeline.get("duration_sec")),
    )
    if observation_duration <= 0 and segments:
        observation_duration = max(
            0.0,
            _number(segments[-1].get("end_sec")) - _number(segments[0].get("start_sec")),
        )

    observation_flags: list[dict[str, Any]] = []
    active_decisions: list[str] = [item["decision"] for item in event_analyses]

    if longest_oob >= alerts["prolonged_out_of_bed_alert_sec"]:
        active_decisions.append("ALERT")
        observation_flags.append({
            "flag": "PROLONGED_OUT_OF_BED",
            "decision": "ALERT",
            "duration_sec": round(longest_oob, 3),
            "threshold_sec": alerts["prolonged_out_of_bed_alert_sec"],
            "reason": "The longest continuous confirmed out-of-bed episode meets the ALERT threshold.",
        })
    elif longest_oob >= alerts["prolonged_out_of_bed_monitor_sec"]:
        active_decisions.append("MONITOR")
        observation_flags.append({
            "flag": "PROLONGED_OUT_OF_BED",
            "decision": "MONITOR",
            "duration_sec": round(longest_oob, 3),
            "threshold_sec": alerts["prolonged_out_of_bed_monitor_sec"],
            "reason": "The longest continuous confirmed out-of-bed episode meets the MONITOR threshold.",
        })

    if longest_unknown >= alerts["unknown_state_monitor_sec"]:
        active_decisions.append("MONITOR")
        observation_flags.append({
            "flag": "PROLONGED_UNKNOWN_STATE",
            "decision": "MONITOR",
            "duration_sec": round(longest_unknown, 3),
            "threshold_sec": alerts["unknown_state_monitor_sec"],
            "reason": "The activity state remains unknown for longer than the configured threshold.",
        })

    if longest_sitting_on_bed >= alerts["sitting_on_bed_edge_monitor_sec"]:
        active_decisions.append("MONITOR")
        observation_flags.append({
            "flag": "PROLONGED_SITTING_ON_BED",
            "decision": "MONITOR",
            "duration_sec": round(longest_sitting_on_bed, 3),
            "threshold_sec": alerts["sitting_on_bed_edge_monitor_sec"],
            "reason": (
                "Continuous sitting on the bed exceeds the configured monitoring threshold. "
                "The current pose/bed-region features cannot distinguish the bed edge from the bed centre, "
                "so this is a conservative monitoring flag rather than a confirmed edge-sitting detection."
            ),
        })

    for segment in segments:
        if _segment_activity(segment) != "LYING_IN_BED":
            continue
        spatial_known, inside_seen, outside_seen = _spatial_evidence([segment])
        if spatial_known and outside_seen and not inside_seen:
            active_decisions.append("MONITOR")
            observation_flags.append({
                "flag": "LYING_POSTURE_OUTSIDE_BED_REGION",
                "decision": "MONITOR",
                "start_time_sec": _number(segment.get("start_sec")),
                "end_time_sec": _number(segment.get("end_sec")),
                "reason": "A lying posture is outside the configured bed region; this may indicate floor lying or an inaccurate bed region and requires review.",
            })

    if not active_decisions:
        active_decisions.append("NORMAL")
    overall_decision = max(active_decisions, key=lambda item: DECISION_RANK[_decision(item)])

    decision_reasons: list[str] = []
    if overall_decision == "ALERT":
        decision_reasons.append("At least one configured ALERT condition was met.")
    elif overall_decision == "MONITOR":
        decision_reasons.append("A confirmed event or an uncertainty/monitoring condition requires attention.")
    else:
        decision_reasons.append("No configured monitoring or alert condition was triggered by the available evidence.")

    agent_frameworks = sorted({item["agent_framework"] for item in event_analyses if "agent_framework" in item})
    report = {
        "analysis_version": "step5_langgraph_agent_v2",
        "analysis_method": "langgraph_stategraph_deterministic_context_checks" if LANGGRAPH_AVAILABLE else "deterministic_fallback_context_checks",
        "agent_framework": "LangGraph StateGraph" if LANGGRAPH_AVAILABLE else "Deterministic fallback (LangGraph not installed)",
        "uses_langgraph": LANGGRAPH_AVAILABLE,
        "uses_external_llm_api": False,
        "observation_duration_sec": round(observation_duration, 3),
        "overall_decision": overall_decision,
        "decision_reasons": decision_reasons,
        "thresholds": alerts,
        "summary": {
            "bed_exit_count": sum(1 for item in event_analyses if item["event_type"] == "BED_EXIT"),
            "bed_return_count": sum(1 for item in event_analyses if item["event_type"] == "BED_RETURN"),
            "longest_continuous_out_of_bed_sec": round(longest_oob, 3),
            "longest_continuous_unknown_sec": round(longest_unknown, 3),
            "longest_continuous_sitting_on_bed_sec": round(longest_sitting_on_bed, 3),
            "timeline_final_state": timeline.get("final_state", "UNKNOWN"),
        },
        "observation_flags": observation_flags,
        "event_analyses": event_analyses,
        "note": (
            "The Step 5 agent uses LangGraph for conditional orchestration and explicit Python safety rules. "
            "It does not call an external LLM and does not diagnose medical conditions."
        ),
    }
    if agent_frameworks:
        report["event_agent_frameworks"] = agent_frameworks
    return report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return result


def _load_alert_config(config_path: Path | None) -> tuple[dict[str, Any], float]:
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.is_file() or yaml is None:
        return {}, 0.15
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as exc:
        raise ValueError(f"Could not read config file {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration root must be a mapping.")
    alerts = config.get("alerts", {})
    events = config.get("events", {})
    if not isinstance(alerts, dict):
        alerts = {}
    if not isinstance(events, dict):
        events = {}
    movement_threshold = _number(events.get("min_movement_away_from_bed_norm"), 0.15)
    return alerts, movement_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step 5 LangGraph agentic contextual analysis on Step 4 outputs."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video path used to locate results/<video-stem>/timeline outputs.",
    )
    parser.add_argument("--timeline", type=Path, help="Optional explicit timeline.json path.")
    parser.add_argument("--events", type=Path, help="Optional explicit bed_events.json path.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument("--config", type=Path, help="Optional config.yaml path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = args.video
    if not video_path.is_absolute():
        video_path = (Path.cwd() / video_path).resolve()
    if not video_path.is_file():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        return 2

    run_dir = PROJECT_ROOT / "results" / video_path.stem
    timeline_path = args.timeline or (run_dir / "timeline" / "timeline.json")
    events_path = args.events or (run_dir / "timeline" / "bed_events.json")
    output_path = args.output or (run_dir / "analysis" / "agentic_analysis.json")
    config_path = args.config

    if not timeline_path.is_absolute():
        timeline_path = (Path.cwd() / timeline_path).resolve()
    if not events_path.is_absolute():
        events_path = (Path.cwd() / events_path).resolve()
    for path in (timeline_path, events_path):
        if not path.is_file():
            print(f"ERROR: Required Step 4 output not found: {path}", file=sys.stderr)
            print(
                "Run Step 4 first: python -m src.main --events-video <video>",
                file=sys.stderr,
            )
            return 2

    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    if config_path is not None and not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    try:
        timeline = _load_json(timeline_path)
        events_payload = _load_json(events_path)
        alerts, movement_threshold = _load_alert_config(config_path)
        report = analyze_timeline_and_events(
            timeline,
            events_payload,
            alerts_config=alerts,
            movement_threshold=movement_threshold,
        )
        report["source_timeline"] = str(timeline_path)
        report["source_events"] = str(events_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nStep 5 agentic analysis completed.")
    print(f"Agent framework: {report['agent_framework']}")
    print(f"External LLM API: {report['uses_external_llm_api']}")
    print(f"Overall decision: {report['overall_decision']}")
    print(f"Bed exits analyzed: {report['summary']['bed_exit_count']}")
    print(f"Bed returns analyzed: {report['summary']['bed_return_count']}")
    print(f"Observation flags: {len(report['observation_flags'])}")
    for item in report["event_analyses"]:
        confidence = item["confidence"]
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, (float, int)) else "n/a"
        print(
            f"  {item['event_type']} {item['event_id']}: {item['decision']} "
            f"(context={item['context_support']}, confidence={confidence_text})"
        )
    print(f"Analysis file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
