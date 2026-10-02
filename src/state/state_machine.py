"""Step 4 temporal state tracking, bed-region estimation, and durations."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path
from statistics import median
from typing import Any

from src.state.smoothing import (
    build_stable_segments,
    estimate_frame_interval,
    smooth_observations,
)


IN_BED_ACTIVITIES = {
    "LYING_IN_BED",
    "SITTING_ON_BED",
}

OUT_OF_BED_ACTIVITIES = {
    "STANDING",
    "WALKING",
    "SITTING_OUTSIDE_BED",
}


def _numeric(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_bed_region(
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Estimate a rectangular bed region from confident in-bed poses.

    This is a baseline spatial model, not a ground-truth polygon.
    It uses the person's bounding-box centres from frames already classified
    as lying/sitting on the bed.
    """

    points: list[tuple[float, float]] = []
    widths: list[float] = []
    heights: list[float] = []

    for observation in observations:
        activity = str(
            observation.get("activity", "UNKNOWN")
        )
        confidence = float(
            observation.get("activity_confidence", 0.0)
            or 0.0
        )

        if activity not in IN_BED_ACTIVITIES:
            continue

        if confidence < 0.55:
            continue

        features = observation.get("features", {})

        center_x = _numeric(
            features.get("bbox_center_x")
        )
        center_y = _numeric(
            features.get("bbox_center_y")
        )
        bbox_width = _numeric(
            features.get("bbox_width")
        )
        bbox_height = _numeric(
            features.get("bbox_height")
        )

        if center_x is None or center_y is None:
            continue

        points.append((center_x, center_y))

        if bbox_width is not None:
            widths.append(bbox_width)

        if bbox_height is not None:
            heights.append(bbox_height)

    if len(points) < 2:
        return None

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    median_width = (
        sorted(widths)[len(widths) // 2]
        if widths
        else max(100.0, max_x - min_x)
    )

    median_height = (
        sorted(heights)[len(heights) // 2]
        if heights
        else max(100.0, max_y - min_y)
    )

    # Padding makes the learned region tolerant to normal movement on the bed.
    padding_x = max(
        0.10 * median_width,
        0.10 * max(1.0, max_x - min_x),
    )
    padding_y = max(
        0.10 * median_height,
        0.10 * max(1.0, max_y - min_y),
    )

    return {
        "method": "auto_from_in_bed_pose_centres",
        "min_x": round(min_x - padding_x, 3),
        "max_x": round(max_x + padding_x, 3),
        "min_y": round(min_y - padding_y, 3),
        "max_y": round(max_y + padding_y, 3),
        "source_point_count": len(points),
    }


def _point_inside_bed(
    observation: dict[str, Any],
    bed_region: dict[str, Any] | None,
) -> bool | None:
    """Return whether the person's bbox centre lies inside the bed region."""

    if bed_region is None:
        return None

    features = observation.get("features", {})

    center_x = _numeric(
        features.get("bbox_center_x")
    )
    center_y = _numeric(
        features.get("bbox_center_y")
    )

    if center_x is None or center_y is None:
        return None

    return bool(
        bed_region["min_x"] <= center_x <= bed_region["max_x"]
        and bed_region["min_y"] <= center_y <= bed_region["max_y"]
    )


def _distance_from_bed_region(
    center_x: float,
    center_y: float,
    bed_region: dict[str, Any],
) -> float:
    """Return Euclidean distance from a point to the bed rectangle.

    Distance is zero when the point is inside the learned bed region.
    """

    if (
        bed_region["min_x"]
        <= center_x
        <= bed_region["max_x"]
        and bed_region["min_y"]
        <= center_y
        <= bed_region["max_y"]
    ):
        return 0.0

    dx = 0.0
    dy = 0.0

    if center_x < bed_region["min_x"]:
        dx = bed_region["min_x"] - center_x
    elif center_x > bed_region["max_x"]:
        dx = center_x - bed_region["max_x"]

    if center_y < bed_region["min_y"]:
        dy = bed_region["min_y"] - center_y
    elif center_y > bed_region["max_y"]:
        dy = center_y - bed_region["max_y"]

    return hypot(dx, dy)


def _annotate_segment_movement(
    segments: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    bed_region: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Add movement-away-from-bed measurements to timeline segments."""

    if bed_region is None:
        for segment in segments:
            segment["start_distance_from_bed_px"] = None
            segment["max_distance_from_bed_px"] = None
            segment["movement_away_from_bed_px"] = None
            segment["movement_away_from_bed_norm"] = None
            segment["center_displacement_px"] = None
        return segments

    for segment in segments:
        start_sec = float(segment.get("start_sec", 0.0))
        end_sec = float(segment.get("end_sec", start_sec))

        segment_observations = []
        for observation in observations:
            timestamp = float(
                observation.get("timestamp_sec", -1.0)
            )
            if start_sec <= timestamp < end_sec:
                segment_observations.append(observation)

        if not segment_observations:
            segment["start_distance_from_bed_px"] = None
            segment["max_distance_from_bed_px"] = None
            segment["movement_away_from_bed_px"] = None
            segment["movement_away_from_bed_norm"] = None
            segment["center_displacement_px"] = None
            continue

        points: list[tuple[float, float]] = []
        distances: list[float] = []
        widths: list[float] = []

        for observation in segment_observations:
            features = observation.get("features", {})
            center_x = _numeric(features.get("bbox_center_x"))
            center_y = _numeric(features.get("bbox_center_y"))
            bbox_width = _numeric(features.get("bbox_width"))

            if center_x is None or center_y is None:
                continue

            points.append((center_x, center_y))
            distances.append(
                _distance_from_bed_region(
                    center_x,
                    center_y,
                    bed_region,
                )
            )

            if bbox_width is not None:
                widths.append(max(1.0, bbox_width))

        if not points:
            segment["start_distance_from_bed_px"] = None
            segment["max_distance_from_bed_px"] = None
            segment["movement_away_from_bed_px"] = None
            segment["movement_away_from_bed_norm"] = None
            segment["center_displacement_px"] = None
            continue

        start_distance = distances[0]
        max_distance = max(distances)
        movement_away = max(0.0, max_distance - start_distance)

        scale = median(widths) if widths else 1.0
        first_x, first_y = points[0]
        last_x, last_y = points[-1]
        center_displacement = hypot(
            last_x - first_x,
            last_y - first_y,
        )

        segment["start_distance_from_bed_px"] = round(
            start_distance,
            3,
        )
        segment["max_distance_from_bed_px"] = round(
            max_distance,
            3,
        )
        segment["movement_away_from_bed_px"] = round(
            movement_away,
            3,
        )
        segment["movement_away_from_bed_norm"] = round(
            movement_away / max(1.0, scale),
            4,
        )
        segment["center_displacement_px"] = round(
            center_displacement,
            3,
        )

    return segments


def _apply_spatial_state(
    observation: dict[str, Any],
    bed_region: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert provisional posture labels into spatially aware states."""

    item = dict(observation)

    activity = str(
        item.get("smoothed_activity", "UNKNOWN")
    )

    inside = _point_inside_bed(
        item,
        bed_region,
    )

    item["bed_region_inside"] = inside

    # Sitting/lying require spatial confirmation to be considered IN_BED.
    if activity == "SITTING_ON_BED":
        if inside is True:
            item["final_activity"] = "SITTING_ON_BED"
            item["final_bed_occupancy"] = "IN_BED"
        elif inside is False:
            item["final_activity"] = "SITTING_OUTSIDE_BED"
            item["final_bed_occupancy"] = "OUT_OF_BED"
        else:
            item["final_activity"] = "UNKNOWN"
            item["final_bed_occupancy"] = "UNKNOWN"

    elif activity == "LYING_IN_BED":
        if inside is True:
            item["final_activity"] = "LYING_IN_BED"
            item["final_bed_occupancy"] = "IN_BED"
        elif inside is False:
            item["final_activity"] = "UNKNOWN"
            item["final_bed_occupancy"] = "OUT_OF_BED"
        else:
            item["final_activity"] = "UNKNOWN"
            item["final_bed_occupancy"] = "UNKNOWN"

    elif activity in {"STANDING", "WALKING"}:
        # Standing/walking are treated as OUT_OF_BED posture states. Spatial
        # movement is still retained in bed_region_inside and is used to
        # confirm a real bed exit rather than mere sitting up.
        item["final_activity"] = activity
        item["final_bed_occupancy"] = "OUT_OF_BED"

    else:
        item["final_activity"] = "UNKNOWN"
        item["final_bed_occupancy"] = "UNKNOWN"

    return item


def _format_duration(seconds: float) -> str:
    """Format seconds as Hh Mm Ss / Mm Ss."""

    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    return f"{minutes}m {seconds:02d}s"


def _duration_summary(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate activity and bed-occupancy durations."""

    totals = {
        "LYING_IN_BED": 0.0,
        "SITTING_ON_BED": 0.0,
        "SITTING_OUTSIDE_BED": 0.0,
        "STANDING": 0.0,
        "WALKING": 0.0,
        "UNKNOWN": 0.0,
    }

    time_in_bed = 0.0
    time_out_of_bed = 0.0
    unknown_bed_time = 0.0

    for segment in segments:
        activity = str(segment["activity"])
        duration = float(segment["duration_sec"])

        if activity in totals:
            totals[activity] += duration

        occupancy = str(
            segment.get("bed_occupancy", "UNKNOWN")
        )

        if occupancy == "IN_BED":
            time_in_bed += duration

        elif occupancy == "OUT_OF_BED":
            time_out_of_bed += duration

        else:
            unknown_bed_time += duration

    return {
        "activity_duration_sec": {
            key.lower(): round(value, 3)
            for key, value in totals.items()
        },
        "activity_duration_formatted": {
            key.lower(): _format_duration(value)
            for key, value in totals.items()
        },
        "bed_summary": {
            "time_in_bed_sec": round(time_in_bed, 3),
            "time_out_of_bed_sec": round(time_out_of_bed, 3),
            "unknown_bed_time_sec": round(unknown_bed_time, 3),
            "time_in_bed": _format_duration(time_in_bed),
            "time_out_of_bed": _format_duration(time_out_of_bed),
        },
    }


def build_timeline(
    state_manifest_path: Path,
    output_dir: Path,
    smoothing_window: int = 5,
    min_state_duration_sec: float = 2.0,
    unknown_confidence_threshold: float = 0.4,
) -> dict[str, Any]:
    """Build a temporally smoothed and spatially aware activity timeline."""

    state_manifest_path = Path(state_manifest_path)
    output_dir = Path(output_dir)

    if not state_manifest_path.is_file():
        raise FileNotFoundError(
            f"State manifest not found: {state_manifest_path}"
        )

    with state_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    observations = manifest.get("observations")

    if not isinstance(observations, list) or not observations:
        raise ValueError(
            "state_manifest.json does not contain observations."
        )

    smoothed = smooth_observations(
        observations=observations,
        window_size=smoothing_window,
        unknown_confidence_threshold=unknown_confidence_threshold,
    )

    # Learn spatial bed region from raw confident in-bed observations.
    bed_region = _estimate_bed_region(observations)

    spatial_observations = [
        _apply_spatial_state(
            observation,
            bed_region,
        )
        for observation in smoothed
    ]

    segments = build_stable_segments(
        observations=spatial_observations,
        min_state_duration_sec=min_state_duration_sec,
    )

    segments = _annotate_segment_movement(
        segments=segments,
        observations=spatial_observations,
        bed_region=bed_region,
    )

    frame_interval = estimate_frame_interval(
        observations
    )

    first_timestamp = float(
        observations[0].get("timestamp_sec", 0.0)
    )
    last_timestamp = float(
        observations[-1].get("timestamp_sec", first_timestamp)
    )

    observation_duration = max(
        frame_interval,
        last_timestamp - first_timestamp + frame_interval,
    )

    duration_summary = _duration_summary(segments)

    final_state = (
        segments[-1]["activity"]
        if segments
        else "UNKNOWN"
    )

    result = {
        "source_state_manifest": str(
            state_manifest_path.resolve()
        ),
        "smoothing_window": smoothing_window,
        "min_state_duration_sec": min_state_duration_sec,
        "unknown_confidence_threshold": unknown_confidence_threshold,
        "frame_interval_sec": round(frame_interval, 3),
        "observation_duration_sec": round(
            observation_duration,
            3,
        ),
        "total_observation_time": _format_duration(
            observation_duration
        ),
        "bed_region": bed_region,
        "observation_count": len(spatial_observations),
        "segment_count": len(segments),
        "observations": spatial_observations,
        "segments": segments,
        **duration_summary,
        "final_state": final_state,
        "note": (
            "The bed region is automatically estimated from confident "
            "in-bed pose observations. It is a baseline spatial heuristic, "
            "not a manually annotated ground-truth polygon."
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "timeline.json"

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

    return result
