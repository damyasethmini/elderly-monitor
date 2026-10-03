"""Step 5: local, evidence-grounded agentic analysis for elderly monitoring.

This module performs a deterministic agent workflow over the Step 4 timeline and
bed-event JSON. It selects additional context checks when an event is ambiguous,
then applies configurable safety rules. It requires no hosted LLM or paid API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - project dependencies normally include PyYAML
    yaml = None

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


def _number(value: Any, default: float = 0.0) -> float:
    """Convert a JSON-like value to float without raising on missing data."""
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


def _segment_at_time(
    segments: list[dict[str, Any]], timestamp: float
) -> int | None:
    """Find the segment containing timestamp; tolerate endpoint rounding."""
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
            # If duration is absent, derive it from segment endpoints.
            if duration == 0.0:
                duration = max(
                    0.0,
                    _number(segment.get("end_sec"))
                    - _number(segment.get("start_sec")),
                )
            current += duration
            longest = max(longest, current)
        else:
            # UNKNOWN and other states break a confirmed continuous episode.
            current = 0.0
    return round(longest, 3)


def _context_snapshot(segment: dict[str, Any]) -> dict[str, Any]:
    """Keep useful evidence in the output without copying raw observations."""
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


def _spatial_evidence(
    segments: list[dict[str, Any]],
) -> tuple[bool, bool, bool]:
    """Return (known, inside_bed_seen, outside_bed_seen) for these segments."""
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


def _make_action(
    action: str, status: str, finding: str, **details: Any
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "action": action,
        "status": status,
        "finding": finding,
    }
    if details:
        item["details"] = details
    return item


def _analyze_event(
    event: dict[str, Any],
    segments: list[dict[str, Any]],
    alerts: dict[str, float],
    movement_threshold: float,
) -> dict[str, Any]:
    event_id = str(event.get("event_id", "event_unknown"))
    event_type = str(event.get("event_type", "UNKNOWN")).upper()
    event_time = _number(
        event.get("start_time_sec"),
        _number(event.get("confirmed_time_sec")),
    )
    event_index = _segment_at_time(segments, event_time)
    actions: list[dict[str, Any]] = []
    reasons: list[str] = []

    if event_index is None:
        actions.append(_make_action(
            "LOCATE_EVENT_IN_TIMELINE", "UNCERTAIN",
            "The event timestamp did not match any timeline segment.",
            event_time_sec=event_time,
        ))
        return {
            "event_id": event_id,
            "event_type": event_type,
            "decision": "MONITOR",
            "context_support": "WEAK",
            "confidence": (
            round(_number(event.get("confidence")), 3)
            if event.get("confidence") is not None else None
        ),
            "requires_human_review": True,
            "reasons": ["Event could not be aligned to the activity timeline."],
            "agent_actions": actions,
            "context_segments": [],
        }

    previous = segments[event_index - 1] if event_index > 0 else None
    current = segments[event_index]
    # Inspect the candidate segment and the next two segments to see whether
    # the activity continues in the direction suggested by the event.
    nearby_end = min(len(segments), event_index + 3)
    nearby = segments[max(0, event_index - 1):nearby_end]
    following = segments[event_index:nearby_end]

    previous_activity = _segment_activity(previous) if previous else None
    current_activity = _segment_activity(current)
    following_activities = [_segment_activity(item) for item in following]
    has_following_out_of_bed = any(
        activity in OUT_OF_BED_ACTIVITIES for activity in following_activities
    )
    previous_in_bed = previous_activity in IN_BED_ACTIVITIES if previous_activity else False

    actions.append(_make_action(
        "INSPECT_PREVIOUS_SEGMENT",
        "FOUND" if previous else "MISSING",
        (
            f"Previous activity was {previous_activity}."
            if previous_activity
            else "The event is at the beginning of the timeline; no previous segment exists."
        ),
        previous_activity=previous_activity,
    ))
    actions.append(_make_action(
        "INSPECT_FOLLOWING_SEGMENTS",
        "FOUND" if following else "MISSING",
        "Following context: " + ", ".join(following_activities),
        following_activities=following_activities,
    ))

    context_is_ambiguous = (
        previous is None
        or previous_activity == "UNKNOWN"
        or current_activity == "UNKNOWN"
        or not has_following_out_of_bed and event_type == "BED_EXIT"
        or event.get("evidence_strength") == "TRANSITION_ONLY"
    )
    context_segments = nearby
    if context_is_ambiguous:
        # The agent responds to ambiguity by widening the temporal window.
        context_segments = segments[max(0, event_index - 3):min(len(segments), event_index + 5)]
        actions.append(_make_action(
            "EXPAND_TEMPORAL_CONTEXT", "EXECUTED",
            "Initial context was ambiguous; inspected a wider window of timeline segments.",
            segment_count=len(context_segments),
        ))

    spatial_known, inside_seen, outside_seen = _spatial_evidence(context_segments)
    movement_norm = _number(event.get("movement_away_from_bed_norm"), -1.0)
    movement_supports_exit = movement_norm >= movement_threshold
    actions.append(_make_action(
        "CHECK_BED_REGION_AND_MOVEMENT",
        "SUPPORTED" if outside_seen or movement_supports_exit else (
            "INSIDE_ONLY" if inside_seen else "INSUFFICIENT"
        ),
        (
            "Spatial or movement evidence supports movement away from the bed."
            if outside_seen or movement_supports_exit
            else "No positive evidence of movement away from the bed was found in the event context."
        ),
        spatial_evidence_known=spatial_known,
        bed_region_inside_seen=inside_seen,
        bed_region_outside_seen=outside_seen,
        movement_away_from_bed_norm=(movement_norm if movement_norm >= 0 else None),
        movement_threshold=movement_threshold,
    ))

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
            decision = "MONITOR" if DECISION_RANK[decision] < DECISION_RANK["MONITOR"] else decision
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
            if DECISION_RANK[decision] < DECISION_RANK["MONITOR"]:
                decision = "MONITOR"
            reasons.append(
                "The continuous out-of-bed episode meets the configured MONITOR duration threshold."
            )
        elif DECISION_RANK[decision] < DECISION_RANK["MONITOR"]:
            # A confirmed departure remains visible to the monitoring layer even
            # when it is too short to trigger a prolonged-absence threshold.
            decision = "MONITOR"
            reasons.append("A confirmed bed exit is recorded for monitoring.")

    elif event_type == "BED_RETURN":
        has_in_bed_after = any(
            _segment_activity(item) in IN_BED_ACTIVITIES for item in following
        )
        if previous_activity in OUT_OF_BED_ACTIVITIES and has_in_bed_after and (inside_seen or spatial_known is False):
            reasons.append("The timeline supports a transition from outside the bed back to an in-bed posture.")
            context_support = "STRONG"
        else:
            decision = "MONITOR"
            context_support = "WEAK"
            requires_review = True
            reasons.append("Return context is incomplete or lacks positive in-bed evidence.")

    else:
        decision = "MONITOR"
        context_support = "WEAK"
        requires_review = True
        reasons.append(f"Unsupported event type {event_type}; review the event manually.")

    # A reported ALERT decision is never downgraded by the contextual layer.
    if _decision(event.get("decision"), "NORMAL") == "ALERT":
        decision = "ALERT"
        reasons.append("The upstream event detector already marked this event ALERT; escalation is preserved.")

    # Preserve the upstream detector's confidence. Context support is reported
    # separately so this stage does not manufacture a new probability.
    confidence = (
        round(_number(event.get("confidence")), 3)
        if event.get("confidence") is not None else None
    )

    return {
        "event_id": event_id,
        "event_type": event_type,
        "start_time_sec": round(event_time, 3),
        "confirmed_time_sec": event.get("confirmed_time_sec"),
        "decision": decision,
        "context_support": context_support,
        "confidence": round(confidence, 3),
        "requires_human_review": requires_review,
        "reasons": reasons,
        "findings": {
            "previous_activity": previous_activity,
            "event_segment_activity": current_activity,
            "following_activities": following_activities,
            "previous_activity_was_in_bed": previous_in_bed,
            "following_context_contains_out_of_bed_activity": has_following_out_of_bed,
            "bed_region_spatial_evidence_known": spatial_known,
            "bed_region_inside_seen": inside_seen,
            "bed_region_outside_seen": outside_seen,
            "movement_away_from_bed_norm": movement_norm if movement_norm >= 0 else None,
            "movement_threshold": movement_threshold,
            "out_of_bed_duration_sec": _number(event.get("out_of_bed_duration_sec")),
        },
        "agent_actions": actions,
        "context_segments": [_context_snapshot(item) for item in context_segments],
    }


def analyze_timeline_and_events(
    timeline: dict[str, Any],
    events_payload: dict[str, Any],
    alerts_config: dict[str, Any] | None = None,
    movement_threshold: float = 0.15,
) -> dict[str, Any]:
    """Run Step 5 contextual checks and return a JSON-serializable report."""
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

    # Avoid silently applying contradictory duration thresholds.
    if alerts["prolonged_out_of_bed_alert_sec"] < alerts["prolonged_out_of_bed_monitor_sec"]:
        alerts["prolonged_out_of_bed_alert_sec"] = alerts["prolonged_out_of_bed_monitor_sec"]

    source_events = events_payload.get("events", [])
    if not isinstance(source_events, list):
        raise ValueError("Bed events JSON must contain a list named 'events'.")

    event_analyses = [
        _analyze_event(event, segments, alerts, movement_threshold)
        for event in source_events
        if isinstance(event, dict)
    ]

    longest_oob = _number(
        (events_payload.get("bed_summary") or {}).get("longest_out_of_bed_period_sec"),
        _duration_of_contiguous_activity(segments, OUT_OF_BED_ACTIVITIES),
    )
    longest_unknown = _duration_of_contiguous_activity(segments, {"UNKNOWN"})
    longest_sitting_on_bed = _duration_of_contiguous_activity(segments, {"SITTING_ON_BED"})
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

    # Inspect lying postures against the bed polygon. A mismatch is uncertain
    # safety evidence, not proof of a fall, so it raises MONITOR for review.
    for index, segment in enumerate(segments):
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

    return {
        "analysis_version": "step5_local_agent_v1",
        "analysis_method": "deterministic_agentic_context_checks",
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
            "This local agent selects temporal/spatial checks using explicit rules. "
            "It is not a diagnosis and does not prove a fall; uncertain cases should be reviewed."
        ),
    }


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
        description="Run Step 5 local agentic contextual analysis on Step 4 outputs."
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
            print("Run Step 4 first: python -m src.main --events-video <video>", file=sys.stderr)
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
    print(f"Overall decision: {report['overall_decision']}")
    print(f"Bed exits analyzed: {report['summary']['bed_exit_count']}")
    print(f"Bed returns analyzed: {report['summary']['bed_return_count']}")
    print(f"Observation flags: {len(report['observation_flags'])}")
    for item in report["event_analyses"]:
        print(
            f"  {item['event_type']} {item['event_id']}: {item['decision']} "
            f"(context={item['context_support']}, confidence={item['confidence']:.2f})"
        )
    print(f"Analysis file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
