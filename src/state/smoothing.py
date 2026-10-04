"""Temporal smoothing and stable-segment construction for Step 4."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


KNOWN_ACTIVITIES = {
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "OUT_OF_BED",
}


def _effective_label(
    observation: dict[str, Any],
    unknown_confidence_threshold: float = 0.4,
) -> str:
    """Return a usable raw label while respecting confidence."""
    activity = str(observation.get("activity", "UNKNOWN"))
    confidence = float(
        observation.get("activity_confidence", 0.0) or 0.0
    )

    if not observation.get("person_detected", False):
        return "UNKNOWN"

    if activity not in KNOWN_ACTIVITIES:
        return "UNKNOWN"

    if confidence < unknown_confidence_threshold:
        return "UNKNOWN"

    return activity


def estimate_frame_interval(
    observations: list[dict[str, Any]],
) -> float:
    """Estimate the median interval between sampled observations."""
    timestamps = [
        float(item.get("timestamp_sec", 0.0))
        for item in observations
    ]

    differences = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    ]

    if not differences:
        return 1.0

    return max(0.001, float(median(differences)))


def _majority(values: list[str]) -> str:
    """Return the most common non-UNKNOWN label, or UNKNOWN."""
    useful = [
        value for value in values
        if value not in {"UNKNOWN", "NO_PERSON"}
    ]

    if not useful:
        return "UNKNOWN"

    counts = Counter(useful)

    # Deterministic tie-breaking: prefer the most recent tied label.
    maximum = max(counts.values())
    tied = {
        label
        for label, count in counts.items()
        if count == maximum
    }

    for value in reversed(values):
        if value in tied:
            return value

    return "UNKNOWN"


def smooth_observations(
    observations: list[dict[str, Any]],
    window_size: int = 5,
    unknown_confidence_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Remove isolated frame glitches without erasing real transitions.

    Unlike a simple causal majority vote, this uses local neighbours only
    to repair isolated labels. Long runs of a state remain unchanged.
    """

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if not observations:
        return []

    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2

    raw_labels = [
        _effective_label(
            observation,
            unknown_confidence_threshold,
        )
        for observation in observations
    ]

    smoothed_labels = list(raw_labels)

    for index in range(len(observations)):
        current = raw_labels[index]

        start = max(0, index - radius)
        end = min(len(observations), index + radius + 1)

        neighbourhood = raw_labels[start:end]

        # Only repair an isolated UNKNOWN or one-frame disagreement.
        left = raw_labels[index - 1] if index > 0 else None
        right = raw_labels[index + 1] if index + 1 < len(raw_labels) else None

        if (
            current == "UNKNOWN"
            and left == right
            and left not in {None, "UNKNOWN"}
        ):
            smoothed_labels[index] = left
            continue

        # Never overwrite a confident non-UNKNOWN state here. A one-frame
        # state change can be a real transition (for example standing -> lying).
        if current != "UNKNOWN":
            continue

        # Fill an isolated UNKNOWN only when both immediate neighbours agree.
        if (
            left == right
            and left not in {None, "UNKNOWN"}
        ):
            smoothed_labels[index] = left
            continue

        # For a short unknown gap, require the same known state on both sides.
        # Do not use a broad majority vote because that can erase real
        # transitions at segment boundaries.
        if current == "UNKNOWN":
            forward_known = None
            for offset in range(1, radius + 1):
                candidate_index = index + offset
                if candidate_index >= len(raw_labels):
                    break
                if raw_labels[candidate_index] != "UNKNOWN":
                    forward_known = raw_labels[candidate_index]
                    break

            backward_known = None
            for offset in range(1, radius + 1):
                candidate_index = index - offset
                if candidate_index < 0:
                    break
                if raw_labels[candidate_index] != "UNKNOWN":
                    backward_known = raw_labels[candidate_index]
                    break

            if (
                backward_known is not None
                and backward_known == forward_known
            ):
                smoothed_labels[index] = backward_known

    result: list[dict[str, Any]] = []

    for index, observation in enumerate(observations):
        item = dict(observation)

        item["raw_activity"] = observation.get("activity")
        item["raw_activity_confidence"] = observation.get(
            "activity_confidence", 0.0
        )
        item["smoothed_activity"] = smoothed_labels[index]

        confidence_values = []

        for candidate_index in range(
            max(0, index - radius),
            min(len(observations), index + radius + 1),
        ):
            if smoothed_labels[candidate_index] == smoothed_labels[index]:
                confidence_values.append(
                    float(
                        observations[candidate_index].get(
                            "activity_confidence", 0.0
                        )
                        or 0.0
                    )
                )

        item["smoothed_activity_confidence"] = round(
            (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else 0.0
            ),
            4,
        )

        result.append(item)

    return result


def _make_segment(
    activity: str,
    occupancy: str,
    frames: list[dict[str, Any]],
    frame_interval: float,
) -> dict[str, Any]:
    start = float(frames[0].get("timestamp_sec", 0.0))
    last = float(frames[-1].get("timestamp_sec", start))
    end = last + frame_interval

    confidences = [
        float(
            item.get("smoothed_activity_confidence", 0.0) or 0.0
        )
        for item in frames
        if item.get("smoothed_activity_confidence") is not None
    ]

    bed_region_values = [
        item.get("bed_region_inside")
        for item in frames
        if item.get("bed_region_inside") is not None
    ]

    return {
        "activity": activity,
        "bed_occupancy": occupancy,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(end - start, 3),
        "frame_count": len(frames),
        "mean_confidence": round(
            sum(confidences) / len(confidences)
            if confidences else 0.0,
            4,
        ),
        "bed_region_inside_at_start": (
            bed_region_values[0]
            if bed_region_values
            else None
        ),
        "bed_region_inside_at_end": (
            bed_region_values[-1]
            if bed_region_values
            else None
        ),
        "bed_region_contains_outside_point": (
            any(value is False for value in bed_region_values)
            if bed_region_values
            else None
        ),
    }


def build_stable_segments(
    observations: list[dict[str, Any]],
    min_state_duration_sec: float = 2.0,
) -> list[dict[str, Any]]:
    """Build contiguous activity/bed-occupancy segments.

    Short segments are merged only when both neighbouring segments represent
    the same state. Real transitions such as sitting -> standing are retained.
    """

    if min_state_duration_sec < 0:
        raise ValueError(
            "min_state_duration_sec cannot be negative."
        )

    if not observations:
        return []

    frame_interval = estimate_frame_interval(observations)

    segments: list[dict[str, Any]] = []
    current_frames = [observations[0]]
    current_activity = observations[0].get(
        "final_activity",
        observations[0].get("smoothed_activity", "UNKNOWN"),
    )
    current_occupancy = observations[0].get(
        "final_bed_occupancy",
        observations[0].get("smoothed_bed_occupancy", "UNKNOWN"),
    )

    for observation in observations[1:]:
        activity = observation.get(
            "final_activity",
            observation.get("smoothed_activity", "UNKNOWN"),
        )
        occupancy = observation.get(
            "final_bed_occupancy",
            observation.get("smoothed_bed_occupancy", "UNKNOWN"),
        )

        if (
            activity == current_activity
            and occupancy == current_occupancy
        ):
            current_frames.append(observation)
            continue

        segments.append(
            _make_segment(
                current_activity,
                current_occupancy,
                current_frames,
                frame_interval,
            )
        )

        current_frames = [observation]
        current_activity = activity
        current_occupancy = occupancy

    segments.append(
        _make_segment(
            current_activity,
            current_occupancy,
            current_frames,
            frame_interval,
        )
    )

    # Merge isolated noise only if the same state occurs on both sides.
    #
    # Replace the three affected segments atomically, then restart the scan.
    # The older in-place/index-skipping implementation could accidentally
    # discard an adjacent short segment when several short states alternated,
    # which created gaps in the final timeline.
    changed = True

    while changed and len(segments) >= 3:
        changed = False

        for index in range(1, len(segments) - 1):
            current = segments[index]
            previous = segments[index - 1]
            following = segments[index + 1]

            if not (
                current["duration_sec"] < min_state_duration_sec
                and previous["activity"] == following["activity"]
                and previous["bed_occupancy"] == following["bed_occupancy"]
            ):
                continue

            combined = dict(previous)
            combined["end_sec"] = following["end_sec"]
            combined["duration_sec"] = round(
                combined["end_sec"] - combined["start_sec"],
                3,
            )

            previous_frames = int(previous.get("frame_count", 0))
            current_frames = int(current.get("frame_count", 0))
            following_frames = int(following.get("frame_count", 0))
            total_frames = previous_frames + current_frames + following_frames
            combined["frame_count"] = total_frames

            if total_frames > 0:
                combined["mean_confidence"] = round(
                    (
                        float(previous.get("mean_confidence", 0.0))
                        * previous_frames
                        + float(current.get("mean_confidence", 0.0))
                        * current_frames
                        + float(following.get("mean_confidence", 0.0))
                        * following_frames
                    )
                    / total_frames,
                    4,
                )

            combined["bed_region_inside_at_end"] = following.get(
                "bed_region_inside_at_end"
            )

            outside_values = [
                previous.get("bed_region_contains_outside_point"),
                current.get("bed_region_contains_outside_point"),
                following.get("bed_region_contains_outside_point"),
            ]
            known_outside_values = [
                value for value in outside_values if value is not None
            ]
            combined["bed_region_contains_outside_point"] = (
                any(value is True for value in known_outside_values)
                if known_outside_values
                else None
            )

            segments = (
                segments[: index - 1]
                + [combined]
                + segments[index + 2 :]
            )
            changed = True
            break

    return segments
