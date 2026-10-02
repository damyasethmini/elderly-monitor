"""Step 4 bed-exit and return event detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IN_BED_ACTIVITIES = {
    "LYING_IN_BED",
    "SITTING_ON_BED",
}

OUT_OF_BED_ACTIVITIES = {
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
}


def _is_in_bed(activity: str) -> bool:
    return activity in IN_BED_ACTIVITIES


def _is_out_of_bed(activity: str) -> bool:
    return activity in OUT_OF_BED_ACTIVITIES


def _context(
    segments: list[dict[str, Any]],
    index: int,
    radius: int = 3,
) -> list[dict[str, Any]]:
    start = max(0, index - radius)
    end = min(len(segments), index + radius + 1)
    return segments[start:end]


def _activity_list(
    segments: list[dict[str, Any]],
) -> list[str]:
    return [
        str(segment.get("activity", "UNKNOWN"))
        for segment in segments
    ]


def _sequence_strength(
    labels: list[str],
    direction: str,
) -> str:
    if direction == "EXIT":
        has_lying = "LYING_IN_BED" in labels
        has_sitting = "SITTING_ON_BED" in labels
        has_out_movement = any(
            label in {"STANDING", "WALKING", "SITTING_OUTSIDE_BED"}
            for label in labels
        )

        if has_lying and has_sitting and has_out_movement:
            return "FULL_SEQUENCE"

        if (
            has_sitting and has_out_movement
        ) or (
            has_lying and has_out_movement
        ):
            return "PARTIAL_SEQUENCE"

        return "TRANSITION_ONLY"

    has_outside = any(
        label in OUT_OF_BED_ACTIVITIES
        for label in labels
    )
    has_sitting = "SITTING_ON_BED" in labels
    has_lying = "LYING_IN_BED" in labels

    if has_outside and has_sitting and has_lying:
        return "FULL_SEQUENCE"

    if has_outside and has_lying:
        return "PARTIAL_SEQUENCE"

    return "TRANSITION_ONLY"


def _mean_confidence(
    segments: list[dict[str, Any]],
) -> float:
    values = [
        float(segment.get("mean_confidence", 0.0) or 0.0)
        for segment in segments
    ]

    return (
        sum(values) / len(values)
        if values else 0.0
    )


def _event_confidence(
    labels: list[str],
    direction: str,
    evidence_segments: list[dict[str, Any]],
) -> float:
    strength = _sequence_strength(labels, direction)

    base = {
        "FULL_SEQUENCE": 0.90,
        "PARTIAL_SEQUENCE": 0.76,
        "TRANSITION_ONLY": 0.55,
    }[strength]

    confidence = (
        base * 0.70
        + _mean_confidence(evidence_segments) * 0.30
    )

    return round(
        min(0.99, confidence),
        3,
    )


def _cumulative_out_of_bed_run(
    segments: list[dict[str, Any]],
    start_index: int,
) -> tuple[float, int]:
    """Return duration and final index of consecutive out-of-bed segments."""

    duration = 0.0
    index = start_index

    while index < len(segments):
        activity = str(
            segments[index].get("activity", "UNKNOWN")
        )

        if not _is_out_of_bed(activity):
            break

        duration += float(
            segments[index].get("duration_sec", 0.0)
        )

        index += 1

    return duration, index - 1


def _confirmation_time(
    segments: list[dict[str, Any]],
    start_index: int,
    required_duration: float,
) -> float:
    """Return the timestamp at which sustained out-of-bed time is reached."""

    start_time = float(
        segments[start_index].get("start_sec", 0.0)
    )

    accumulated = 0.0

    for index in range(start_index, len(segments)):
        segment = segments[index]

        if not _is_out_of_bed(
            str(segment.get("activity", "UNKNOWN"))
        ):
            break

        segment_duration = float(
            segment.get("duration_sec", 0.0)
        )

        if accumulated + segment_duration >= required_duration:
            return round(
                float(segment.get("start_sec", start_time))
                + (required_duration - accumulated),
                3,
            )

        accumulated += segment_duration

    return round(
        start_time + accumulated,
        3,
    )


def detect_bed_events(
    timeline: dict[str, Any],
    min_stable_duration_sec: float = 2.0,
    min_out_of_bed_confirmation_sec: float = 1.5,
    min_return_confirmation_sec: float = 1.0,
) -> list[dict[str, Any]]:
    """Detect meaningful bed exits and confirmed returns.

    Exit:
        in-bed posture
          -> out-of-bed posture
          -> person actually leaves the learned bed region
          -> sustained movement outside the bed

    Return:
        out-of-bed period
          -> sitting on bed
          -> lying in bed
    """

    segments = timeline.get("segments")

    if not isinstance(segments, list):
        raise ValueError(
            "Timeline does not contain segments."
        )

    events: list[dict[str, Any]] = []
    exit_count = 0
    return_count = 0

    # ---------------------------------------------------------
    # BED EXIT
    # ---------------------------------------------------------
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]

        previous_activity = str(
            previous.get("activity", "UNKNOWN")
        )
        current_activity = str(
            current.get("activity", "UNKNOWN")
        )

        if not _is_in_bed(previous_activity):
            continue

        if not _is_out_of_bed(current_activity):
            continue

        if float(previous.get("duration_sec", 0.0)) < min_stable_duration_sec:
            continue

        # Require spatial evidence that the person actually moves outside
        # the learned bed region. Standing up alone is not enough.
        run_duration, final_index = _cumulative_out_of_bed_run(
            segments,
            index,
        )

        spatially_outside = any(
            segment.get("bed_region_contains_outside_point") is True
            for segment in segments[index : final_index + 1]
        )

        if not spatially_outside:
            continue

        if run_duration < min_out_of_bed_confirmation_sec:
            continue

        context = _context(
            segments,
            index,
            radius=3,
        )

        labels = _activity_list(context)
        strength = _sequence_strength(
            labels,
            "EXIT",
        )

        if strength == "TRANSITION_ONLY":
            continue

        exit_count += 1

        events.append(
            {
                "event_id": f"bed_exit_{exit_count:03d}",
                "event_type": "BED_EXIT",
                "start_time_sec": round(
                    float(current.get("start_sec", 0.0)),
                    3,
                ),
                "confirmed_time_sec": _confirmation_time(
                    segments,
                    index,
                    min_out_of_bed_confirmation_sec,
                ),
                "from_activity": previous_activity,
                "to_activity": current_activity,
                "evidence_sequence": labels,
                "evidence_strength": strength,
                "out_of_bed_duration_sec": round(
                    run_duration,
                    3,
                ),
                "confidence": _event_confidence(
                    labels,
                    "EXIT",
                    context,
                ),
                "decision": "MONITOR",
            }
        )

    # ---------------------------------------------------------
    # RETURN TO BED
    # ---------------------------------------------------------
    for index, current in enumerate(segments):
        current_activity = str(
            current.get("activity", "UNKNOWN")
        )

        if current_activity != "LYING_IN_BED":
            continue

        if float(current.get("duration_sec", 0.0)) < min_stable_duration_sec:
            continue

        # Find the immediately preceding sitting-on-bed segment.
        if index == 0:
            continue

        sitting_index = index - 1

        if (
            segments[sitting_index].get("activity")
            != "SITTING_ON_BED"
        ):
            continue

        sitting_duration = float(
            segments[sitting_index].get("duration_sec", 0.0)
        )

        if sitting_duration < min_return_confirmation_sec:
            continue

        # Walk backwards through the out-of-bed period before sitting.
        out_start = sitting_index - 1

        if out_start < 0:
            continue

        out_duration = 0.0
        out_end = out_start

        while out_start >= 0:
            activity = str(
                segments[out_start].get("activity", "UNKNOWN")
            )

            if not _is_out_of_bed(activity):
                break

            out_duration += float(
                segments[out_start].get("duration_sec", 0.0)
            )

            out_end = out_start
            out_start -= 1

        # The person must have been outside the bed before sitting down.
        if out_duration < min_return_confirmation_sec:
            continue

        context_start = max(
            0,
            out_start + 1,
        )
        context_end = min(
            len(segments),
            index + 1,
        )

        context = segments[
            context_start:context_end
        ]

        labels = _activity_list(context)
        strength = _sequence_strength(
            labels,
            "RETURN",
        )

        if strength == "TRANSITION_ONLY":
            continue

        # Require actual spatial bed entry on the sitting segment if available.
        sitting_inside = segments[sitting_index].get(
            "bed_region_inside"
        )

        if sitting_inside is False:
            continue

        return_count += 1

        events.append(
            {
                "event_id": f"bed_return_{return_count:03d}",
                "event_type": "RETURN_TO_BED",
                "start_time_sec": round(
                    float(
                        segments[out_start + 1].get(
                            "start_sec",
                            current.get("start_sec", 0.0),
                        )
                    ),
                    3,
                ),
                "confirmed_time_sec": round(
                    float(current.get("start_sec", 0.0)),
                    3,
                ),
                "from_activity": str(
                    segments[sitting_index].get(
                        "activity",
                        "SITTING_ON_BED",
                    )
                ),
                "to_activity": "LYING_IN_BED",
                "evidence_sequence": labels,
                "evidence_strength": strength,
                "out_of_bed_duration_sec": round(
                    out_duration,
                    3,
                ),
                "confidence": _event_confidence(
                    labels,
                    "RETURN",
                    context,
                ),
                "decision": "MONITOR",
            }
        )

    events.sort(
        key=lambda item: float(
            item["confirmed_time_sec"]
        )
    )

    return events


def build_event_summary(
    timeline: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build assignment-style bed event summary."""

    longest_out = 0.0

    for segment in timeline.get("segments", []):
        if segment.get("bed_occupancy") == "OUT_OF_BED":
            longest_out = max(
                longest_out,
                float(segment.get("duration_sec", 0.0)),
            )

    return {
        **timeline.get("bed_summary", {}),
        "bed_exit_count": sum(
            1
            for event in events
            if event.get("event_type") == "BED_EXIT"
        ),
        "bed_return_count": sum(
            1
            for event in events
            if event.get("event_type") == "RETURN_TO_BED"
        ),
        "longest_out_of_bed_period_sec": round(
            longest_out,
            3,
        ),
    }


def save_events(
    events: list[dict[str, Any]],
    timeline: dict[str, Any],
    timeline_path: Path,
    output_path: Path,
) -> Path:
    """Save detected events and their summary."""

    result = {
        "source_timeline": str(
            timeline_path.resolve()
        ),
        "event_count": len(events),
        "events": events,
        "bed_summary": build_event_summary(
            timeline,
            events,
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    return output_path
