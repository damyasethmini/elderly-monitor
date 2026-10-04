"""Step 6: contextual alert decision layer.

This module consumes the Step 4 timeline and Step 5 agentic analysis and
produces the final clip-level NORMAL / MONITOR / ALERT decision.

The decision layer is intentionally deterministic and auditable. It does not
call an external LLM and does not claim to diagnose a medical condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_DECISIONS = {"NORMAL", "MONITOR", "ALERT"}
DECISION_RANK = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}

DEFAULT_ALERTS: dict[str, float] = {
    "sitting_on_bed_edge_monitor_sec": 120.0,
    "prolonged_out_of_bed_monitor_sec": 300.0,
    "prolonged_out_of_bed_alert_sec": 600.0,
    "unknown_state_monitor_sec": 30.0,
}


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


def _activity(segment: dict[str, Any]) -> str:
    return str(segment.get("activity", "UNKNOWN")).upper()


def _segment_duration(segment: dict[str, Any]) -> float:
    duration = _number(segment.get("duration_sec"), -1.0)
    if duration >= 0:
        return duration
    return max(
        0.0,
        _number(segment.get("end_sec")) - _number(segment.get("start_sec")),
    )


def _longest_contiguous_duration(
    segments: list[dict[str, Any]],
    activities: set[str],
) -> float:
    """Return the longest continuous run of the requested activities.

    UNKNOWN, unrelated activities, and missing gaps break continuity. This is
    deliberate: an uncertain interval must not be silently counted as confirmed
    continuous absence.
    """
    longest = 0.0
    current = 0.0

    for segment in segments:
        if _activity(segment) in activities:
            current += _segment_duration(segment)
            longest = max(longest, current)
        else:
            current = 0.0

    return round(longest, 3)


def _flag(
    flag: str,
    decision: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    result = {
        "flag": flag,
        "decision": decision,
        "reason": reason,
    }
    result.update(details)
    return result


def _max_decision(values: list[str]) -> str:
    if not values:
        return "NORMAL"
    return max(
        (_decision(value) for value in values),
        key=lambda value: DECISION_RANK[value],
    )


def build_contextual_alert(
    timeline: dict[str, Any],
    agentic_analysis: dict[str, Any],
    alerts_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Step 6 final decision from Step 4 + Step 5 evidence."""
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Timeline JSON must contain a list named 'segments'.")
    if not all(isinstance(item, dict) for item in segments):
        raise ValueError("Every timeline segment must be a JSON object.")

    alerts = dict(DEFAULT_ALERTS)
    if isinstance(alerts_config, dict):
        for key in alerts:
            if key in alerts_config:
                alerts[key] = max(0.0, _number(alerts_config[key], alerts[key]))

    # Prevent contradictory duration thresholds.
    if alerts["prolonged_out_of_bed_alert_sec"] < alerts["prolonged_out_of_bed_monitor_sec"]:
        alerts["prolonged_out_of_bed_alert_sec"] = alerts["prolonged_out_of_bed_monitor_sec"]

    event_analyses = agentic_analysis.get("event_analyses", [])
    if not isinstance(event_analyses, list):
        event_analyses = []

    upstream_flags = agentic_analysis.get("observation_flags", [])
    if not isinstance(upstream_flags, list):
        upstream_flags = []

    triggers: list[dict[str, Any]] = []
    decisions: list[str] = []

    # 1. Preserve event-level decisions produced by Step 5.
    for event in event_analyses:
        if not isinstance(event, dict):
            continue

        event_decision = _decision(event.get("decision"), "MONITOR")
        decisions.append(event_decision)

        if event_decision != "NORMAL":
            triggers.append(
                _flag(
                    "EVENT_DECISION",
                    event_decision,
                    "Step 5 classified this event as requiring monitoring or alert handling.",
                    event_id=event.get("event_id"),
                    event_type=event.get("event_type"),
                    start_time_sec=event.get("start_time_sec"),
                    confirmed_time_sec=event.get("confirmed_time_sec"),
                    context_support=event.get("context_support"),
                    confidence=event.get("confidence"),
                    requires_human_review=bool(event.get("requires_human_review", False)),
                )
            )

    # 2. Preserve explicit Step 5 alert/monitor flags.
    for item in upstream_flags:
        if not isinstance(item, dict):
            continue
        item_decision = _decision(item.get("decision"), "MONITOR")
        decisions.append(item_decision)
        if item_decision != "NORMAL":
            trigger = dict(item)
            trigger["source"] = "step5_observation_flag"
            triggers.append(trigger)

    # 3. Re-evaluate the key safety thresholds directly from the timeline.
    # This keeps Step 6 independently auditable and avoids depending on a
    # potentially stale summary copied into Step 5.
    out_of_bed_activities = {"SITTING_OUTSIDE_BED", "STANDING", "WALKING"}
    longest_oob = _longest_contiguous_duration(segments, out_of_bed_activities)
    longest_unknown = _longest_contiguous_duration(segments, {"UNKNOWN"})
    longest_sitting = _longest_contiguous_duration(segments, {"SITTING_ON_BED"})

    if longest_oob >= alerts["prolonged_out_of_bed_alert_sec"]:
        decisions.append("ALERT")
        triggers.append(
            _flag(
                "PROLONGED_OUT_OF_BED",
                "ALERT",
                "A continuous confirmed out-of-bed episode reached the configured ALERT threshold.",
                duration_sec=longest_oob,
                threshold_sec=alerts["prolonged_out_of_bed_alert_sec"],
            )
        )
    elif longest_oob >= alerts["prolonged_out_of_bed_monitor_sec"]:
        decisions.append("MONITOR")
        triggers.append(
            _flag(
                "PROLONGED_OUT_OF_BED",
                "MONITOR",
                "A continuous confirmed out-of-bed episode reached the configured MONITOR threshold.",
                duration_sec=longest_oob,
                threshold_sec=alerts["prolonged_out_of_bed_monitor_sec"],
            )
        )

    if longest_unknown >= alerts["unknown_state_monitor_sec"]:
        decisions.append("MONITOR")
        triggers.append(
            _flag(
                "PROLONGED_UNKNOWN_STATE",
                "MONITOR",
                "The activity state remained UNKNOWN longer than the configured monitoring threshold.",
                duration_sec=longest_unknown,
                threshold_sec=alerts["unknown_state_monitor_sec"],
            )
        )

    if longest_sitting >= alerts["sitting_on_bed_edge_monitor_sec"]:
        decisions.append("MONITOR")
        triggers.append(
            _flag(
                "PROLONGED_SITTING_ON_BED",
                "MONITOR",
                (
                    "Continuous sitting on the bed exceeded the configured monitoring threshold. "
                    "The current baseline does not directly locate the body relative to the bed edge."
                ),
                duration_sec=longest_sitting,
                threshold_sec=alerts["sitting_on_bed_edge_monitor_sec"],
            )
        )

    # 4. A lying posture that is spatially outside the learned/configured bed
    # region is uncertain evidence. It raises MONITOR, not ALERT.
    for segment in segments:
        if _activity(segment) != "LYING_IN_BED":
            continue

        inside_start = segment.get("bed_region_inside_at_start")
        inside_end = segment.get("bed_region_inside_at_end")
        has_known_spatial = isinstance(inside_start, bool) or isinstance(inside_end, bool)
        outside = (
            inside_start is False
            and inside_end is False
            or segment.get("bed_region_contains_outside_point") is True
        )
        if has_known_spatial and outside and not (inside_start is True or inside_end is True):
            decisions.append("MONITOR")
            triggers.append(
                _flag(
                    "LYING_POSTURE_OUTSIDE_BED_REGION",
                    "MONITOR",
                    (
                        "A lying posture was spatially outside the bed region. "
                        "This is uncertain evidence and may reflect floor lying or an inaccurate bed region."
                    ),
                    start_time_sec=segment.get("start_sec"),
                    end_time_sec=segment.get("end_sec"),
                )
            )

    final_decision = _max_decision(decisions)

    # Human review is required whenever Step 5 explicitly requested it or the
    # final decision is ALERT. Monitoring alone may be automatic monitoring.
    requires_human_review = final_decision == "ALERT" or any(
        bool(item.get("requires_human_review", False))
        for item in event_analyses
        if isinstance(item, dict)
    )

    if final_decision == "ALERT":
        decision_reason = "A configured ALERT condition was met by the available temporal/contextual evidence."
    elif final_decision == "MONITOR":
        decision_reason = "One or more contextual conditions require continued monitoring or review."
    else:
        decision_reason = "No configured monitoring or alert condition was triggered."

    summary = {
        "bed_exit_count": sum(
            1
            for item in event_analyses
            if isinstance(item, dict) and str(item.get("event_type", "")).upper() == "BED_EXIT"
        ),
        "bed_return_count": sum(
            1
            for item in event_analyses
            if isinstance(item, dict) and str(item.get("event_type", "")).upper() == "BED_RETURN"
        ),
        "longest_continuous_out_of_bed_sec": longest_oob,
        "longest_continuous_unknown_sec": longest_unknown,
        "longest_continuous_sitting_on_bed_sec": longest_sitting,
        "timeline_final_state": timeline.get("final_state", "UNKNOWN"),
    }

    return {
        "decision_version": "step6_contextual_alert_v1",
        "decision_method": "deterministic_contextual_safety_rules",
        "uses_external_llm_api": False,
        "overall_decision": final_decision,
        "decision_reason": decision_reason,
        "requires_human_review": requires_human_review,
        "thresholds": alerts,
        "summary": summary,
        "trigger_count": len(triggers),
        "triggers": triggers,
        "upstream_step5_decision": _decision(
            agentic_analysis.get("overall_decision"),
            "NORMAL",
        ),
        "source_analysis_version": agentic_analysis.get("analysis_version"),
        "note": (
            "Step 6 is a transparent alert-policy layer. It does not diagnose a condition "
            "or prove a fall; uncertain evidence is surfaced for review."
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


def _load_alerts(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    if yaml is None or not config_path.is_file():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as exc:
        raise ValueError(f"Could not read config file {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        return {}

    alerts = config.get("alerts", {})
    return alerts if isinstance(alerts, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step 6 contextual alert decision on Step 5 analysis."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video path used to locate results/<video-stem>/ outputs.",
    )
    parser.add_argument("--timeline", type=Path, help="Optional explicit timeline.json path.")
    parser.add_argument(
        "--analysis",
        type=Path,
        help="Optional explicit Step 5 agentic_analysis.json path.",
    )
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
    analysis_path = args.analysis or (run_dir / "analysis" / "agentic_analysis.json")
    output_path = args.output or (run_dir / "alerts" / "contextual_alert.json")

    paths = [timeline_path, analysis_path]
    for path in paths:
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            print(f"ERROR: Required input not found: {path}", file=sys.stderr)
            return 2

    if not timeline_path.is_absolute():
        timeline_path = (Path.cwd() / timeline_path).resolve()
    if not analysis_path.is_absolute():
        analysis_path = (Path.cwd() / analysis_path).resolve()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    try:
        timeline = _load_json(timeline_path)
        analysis = _load_json(analysis_path)
        alerts = _load_alerts(config_path)
        report = build_contextual_alert(timeline, analysis, alerts)
        report["source_timeline"] = str(timeline_path)
        report["source_analysis"] = str(analysis_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nStep 6 contextual alert decision completed.")
    print(f"Overall decision: {report['overall_decision']}")
    print(f"Trigger count: {report['trigger_count']}")
    print(f"Requires human review: {report['requires_human_review']}")

    for trigger in report["triggers"]:
        print(
            f"  {trigger.get('flag', 'UNKNOWN')}: "
            f"{trigger.get('decision', 'MONITOR')} - "
            f"{trigger.get('reason', '')}"
        )

    print(f"Alert file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
