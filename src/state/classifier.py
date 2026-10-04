# """Activity and bed-occupancy classification.

# This is an interpretable baseline classifier. It uses pose geometry and
# frame-to-frame movement rather than a separately trained ML classifier.

# Temporal smoothing and proper bed-exit/return logic will be added in Step 4.
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# from src.state.features import extract_pose_features


# DEFAULT_THRESHOLDS = {
#     "lying_body_axis_angle_deg": 55.0,
#     "lying_bbox_aspect_ratio": 1.25,
#     "sitting_body_axis_angle_min_deg": 15.0,
#     "sitting_body_axis_angle_max_deg": 55.0,
#     "sitting_knee_hip_gap_max": 0.45,
#     "walking_motion_threshold": 0.15,
#     "keypoint_confidence": 0.30,
# }


# def classify_activity(
#     features: dict[str, Any],
#     thresholds: dict[str, float] | None = None,
# ) -> tuple[str, float]:
#     """Classify activity from pose features.

#     Returns:
#         (activity, confidence)
#     """

#     config = {
#         **DEFAULT_THRESHOLDS,
#         **(thresholds or {}),
#     }

#     if not features.get("person_detected"):
#         return "NO_PERSON", 0.0

#     body_angle = features.get(
#         "body_axis_angle_deg"
#     )

#     aspect_ratio = features.get(
#         "bbox_aspect_ratio"
#     )

#     knee_hip_gap = features.get(
#         "knee_hip_vertical_gap_norm"
#     )

#     motion = features.get(
#         "motion_norm_per_sec"
#     )

#     # ---------------------------------------------------------
#     # 1. Lying
#     # ---------------------------------------------------------
#     lying_by_angle = (
#         body_angle is not None
#         and body_angle
#         >= config["lying_body_axis_angle_deg"]
#     )

#     lying_by_aspect = (
#         aspect_ratio is not None
#         and aspect_ratio
#         >= config["lying_bbox_aspect_ratio"]
#     )

#     if lying_by_angle or lying_by_aspect:
#         if lying_by_angle and lying_by_aspect:
#             confidence = 0.88
#         else:
#             confidence = 0.72

#         return "LYING_IN_BED", confidence

#     # ---------------------------------------------------------
#     # 2. Sitting
#     # ---------------------------------------------------------
#     sitting_by_angle = (
#         body_angle is not None
#         and config["sitting_body_axis_min_deg"]
#         <= body_angle
#         <= config["sitting_body_axis_max_deg"]
#     )

#     sitting_by_knee_position = (
#         knee_hip_gap is not None
#         and knee_hip_gap
#         <= config["sitting_knee_hip_gap_max"]
#     )

#     if sitting_by_angle and sitting_by_knee_position:
#         return "SITTING_ON_BED", 0.80

#     if sitting_by_angle:
#         return "SITTING_ON_BED", 0.65

#     # ---------------------------------------------------------
#     # 3. Walking
#     # ---------------------------------------------------------
#     if (
#         motion is not None
#         and motion >= config["walking_motion_threshold"]
#     ):
#         return "WALKING", 0.74

#     # ---------------------------------------------------------
#     # 4. Standing
#     # ---------------------------------------------------------
#     return "STANDING", 0.76


# def derive_bed_occupancy(
#     activity: str,
# ) -> tuple[str, float]:
#     """Convert activity into the initial bed occupancy state.

#     This is intentionally a baseline.
#     A spatial bed-region check will make occupancy more reliable later.
#     """

#     if activity in {
#         "LYING_IN_BED",
#         "SITTING_ON_BED",
#     }:
#         return "IN_BED", 0.85

#     if activity in {
#         "STANDING",
#         "WALKING",
#     }:
#         return "OUT_OF_BED", 0.85

#     return "UNKNOWN", 0.0


# def classify_pose_manifest(
#     pose_manifest_path: Path,
#     output_dir: Path,
#     thresholds: dict[str, float] | None = None,
# ) -> dict[str, Any]:
#     """Classify every observation in pose_manifest.json."""

#     pose_manifest_path = Path(pose_manifest_path)
#     output_dir = Path(output_dir)

#     if not pose_manifest_path.is_file():
#         raise FileNotFoundError(
#             f"Pose manifest not found: {pose_manifest_path}"
#         )

#     with pose_manifest_path.open(
#         "r",
#         encoding="utf-8",
#     ) as file:
#         manifest = json.load(file)

#     observations = manifest.get("observations")

#     if not isinstance(observations, list):
#         raise ValueError(
#             "pose_manifest.json does not contain "
#             "an observations list."
#         )

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     state_observations: list[dict[str, Any]] = []

#     previous_observation = None
#     previous_timestamp = None

#     for observation in observations:

#         timestamp = float(
#             observation.get(
#                 "timestamp_sec",
#                 0.0,
#             )
#         )

#         if previous_timestamp is None:
#             delta_time = 1.0
#         else:
#             delta_time = max(
#                 0.001,
#                 timestamp - previous_timestamp,
#             )

#         features = extract_pose_features(
#             observation=observation,
#             previous_observation=previous_observation,
#             delta_time_sec=delta_time,
#             min_keypoint_confidence=float(
#                 (thresholds or {}).get(
#                     "keypoint_confidence",
#                     DEFAULT_THRESHOLDS[
#                         "keypoint_confidence"
#                     ],
#                 )
#             ),
#         )

#         activity, activity_confidence = classify_activity(
#             features,
#             thresholds=thresholds,
#         )

#         bed_occupancy, occupancy_confidence = (
#             derive_bed_occupancy(activity)
#         )

#         state_observations.append(
#             {
#                 "sample_number": observation.get(
#                     "sample_number"
#                 ),
#                 "timestamp_sec": timestamp,
#                 "image": observation.get("image"),
#                 "activity": activity,
#                 "activity_confidence": round(
#                     activity_confidence,
#                     4,
#                 ),
#                 "bed_occupancy": bed_occupancy,
#                 "bed_occupancy_confidence": round(
#                     occupancy_confidence,
#                     4,
#                 ),
#                 "features": features,
#             }
#         )

#         previous_observation = observation
#         previous_timestamp = timestamp

#     result = {
#         "source_pose_manifest": str(
#             pose_manifest_path.resolve()
#         ),
#         "classifier": "rule_based_pose_baseline",
#         "thresholds": {
#             **DEFAULT_THRESHOLDS,
#             **(thresholds or {}),
#         },
#         "frame_count": len(state_observations),
#         "observations": state_observations,
#     }

#     output_path = (
#         output_dir / "state_manifest.json"
#     )

#     with output_path.open(
#         "w",
#         encoding="utf-8",
#     ) as file:
#         json.dump(
#             result,
#             file,
#             indent=2,
#             ensure_ascii=False,
#         )
#         file.write("\n")

#     return result


"""Rule-based activity/state classification for elderly-monitor.

Step 3B converts pose observations into frame-level activity states.

The classifier deliberately:
- uses multiple pose cues instead of one hard threshold,
- tolerates missing torso keypoints,
- does not use bounding-box width as a hard rejection for sitting,
- keeps NO_PERSON/ambiguous frames as UNKNOWN,
- keeps bed occupancy explicitly provisional because no bed-region polygon is
  available at this stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.state.features import extract_pose_features


DEFAULT_THRESHOLDS: dict[str, float] = {
    # ---------------------------------------------------------
    # Lying
    # ---------------------------------------------------------
    # body_axis_angle_deg:
    # 0   = approximately vertical
    # 90  = approximately horizontal
    "lying_body_axis_min_deg": 50.0,
    "lying_min_bbox_aspect_ratio": 1.00,

    # ---------------------------------------------------------
    # Sitting
    # ---------------------------------------------------------
    # Sitting is mainly determined from torso angle and the
    # vertical relationship between knees and hips.
    #
    # IMPORTANT:
    # bbox aspect ratio is NOT used as a hard rejection.
    "sitting_body_axis_max_deg": 45.0,
    "sitting_knee_hip_gap_max": 0.24,
    "sitting_knee_hip_gap_strong": 0.16,
    "sitting_bbox_aspect_min": 0.90,
    "sitting_bbox_aspect_max": 1.30,
    "sitting_min_keypoints": 7,

    # ---------------------------------------------------------
    # Walking
    # ---------------------------------------------------------
    "walking_motion_threshold": 0.15,
    "walking_bbox_aspect_max": 1.20,

    # ---------------------------------------------------------
    # Standing
    # ---------------------------------------------------------
    "standing_bbox_aspect_max": 0.85,
    "standing_pose_span_y_min": 0.45,

    # ---------------------------------------------------------
    # Pose quality
    # ---------------------------------------------------------
    "keypoint_confidence": 0.30,
}


def _float_or_none(value: Any) -> float | None:
    """Convert a value to float while preserving missing values."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_activity(
    features: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> tuple[str, float]:
    """Return (activity, heuristic confidence) for one observation."""

    config = {
        **DEFAULT_THRESHOLDS,
        **(thresholds or {}),
    }

    # ---------------------------------------------------------
    # No detected person
    # ---------------------------------------------------------
    if not features.get("person_detected"):
        return "UNKNOWN", 0.0

    body_angle = _float_or_none(
        features.get("body_axis_angle_deg")
    )

    aspect_ratio = _float_or_none(
        features.get("bbox_aspect_ratio")
    )

    knee_hip_gap = _float_or_none(
        features.get("knee_hip_vertical_gap_norm")
    )

    pose_span_y = _float_or_none(
        features.get("pose_span_y_norm")
    )

    motion = _float_or_none(
        features.get("motion_norm_per_sec")
    )

    valid_keypoints = int(
        features.get("valid_keypoint_count", 0) or 0
    )

    # =========================================================
    # 1. LYING
    # =========================================================
    #
    # A horizontal torso is the strongest lying cue.
    #
    # We lower the threshold slightly from 55 -> 50 because the
    # current video has transition frames around 50-55 degrees.
    if (
        body_angle is not None
        and body_angle >= config["lying_body_axis_min_deg"]
    ):

        # A very narrow bbox would strongly contradict a horizontal
        # posture, so only reject that specific case.
        if (
            aspect_ratio is not None
            and aspect_ratio < config["lying_min_bbox_aspect_ratio"]
        ):
            pass

        else:
            if body_angle >= 55.0:
                return "LYING_IN_BED", 0.86

            # Near the boundary: classify, but with lower confidence.
            return "LYING_IN_BED", 0.72

    # =========================================================
    # 2. SITTING
    # =========================================================
    #
    # Main cues:
    #   - torso is not horizontal
    #   - knees are close to hips vertically
    #   - enough keypoints are available
    #
    # NOTE:
    # We intentionally do NOT reject a sitting frame simply
    # because bbox_aspect_ratio is large.
    #
    # Your actual test data contains many sitting frames with
    # aspect ratios > 1.3.
    if (
        body_angle is not None
        and body_angle <= config["sitting_body_axis_max_deg"]
        and knee_hip_gap is not None
        and knee_hip_gap <= config["sitting_knee_hip_gap_max"]
        and valid_keypoints >= config["sitting_min_keypoints"]
    ):

        confidence = 0.78

        # Strong sitting evidence.
        if (
            body_angle <= 35.0
            and knee_hip_gap
            <= config["sitting_knee_hip_gap_strong"]
        ):
            confidence = 0.84

        # Still sitting, but weaker knee evidence.
        elif (
            knee_hip_gap
            > config["sitting_knee_hip_gap_strong"]
        ):
            confidence = 0.70

        return "SITTING_ON_BED", confidence

    # ---------------------------------------------------------
    # Sitting fallback when torso keypoints are missing
    # ---------------------------------------------------------
    #
    # Use bbox + knee/hip relationship only when the torso angle
    # cannot be calculated.
    if (
        body_angle is None
        and aspect_ratio is not None
        and config["sitting_bbox_aspect_min"]
        <= aspect_ratio
        <= config["sitting_bbox_aspect_max"]
        and knee_hip_gap is not None
        and knee_hip_gap
        <= config["sitting_knee_hip_gap_strong"]
        and pose_span_y is not None
        and pose_span_y >= 0.55
        and valid_keypoints >= 6
    ):
        return "SITTING_ON_BED", 0.60

    # =========================================================
    # 3. WALKING
    # =========================================================
    #
    # Motion alone is not enough.
    # We also require an approximately upright bounding box.
    upright_box = (
        aspect_ratio is not None
        and aspect_ratio
        <= config["walking_bbox_aspect_max"]
    )

    if (
        motion is not None
        and motion >= config["walking_motion_threshold"]
        and upright_box
    ):

        # When torso geometry is available, it should not indicate
        # a horizontal posture.
        if (
            body_angle is not None
            and body_angle <= 45.0
        ):

            # Don't call a strongly seated pose walking.
            seated_knees = (
                knee_hip_gap is not None
                and knee_hip_gap
                <= config["sitting_knee_hip_gap_max"]
            )

            if not seated_knees:
                return "WALKING", 0.66

        # If torso keypoints are missing, use pose extent instead.
        if (
            body_angle is None
            and pose_span_y is not None
            and pose_span_y >= 0.55
            and (
                knee_hip_gap is None
                or knee_hip_gap
                > config["sitting_knee_hip_gap_max"]
            )
        ):
            return "WALKING", 0.60

    # =========================================================
    # 4. STANDING
    # =========================================================
    #
    # Full torso information available.
    if (
        body_angle is not None
        and body_angle <= 25.0
        and pose_span_y is not None
        and pose_span_y
        >= config["standing_pose_span_y_min"]
        and (
            knee_hip_gap is None
            or knee_hip_gap
            > config["sitting_knee_hip_gap_max"]
        )
        and (
            aspect_ratio is None
            or aspect_ratio <= 1.20
        )
    ):
        return "STANDING", 0.68

    # ---------------------------------------------------------
    # Standing fallback when torso keypoints are missing
    # ---------------------------------------------------------
    #
    # Tall/narrow bbox + large vertical pose span is useful
    # evidence for standing.
    if (
        body_angle is None
        and aspect_ratio is not None
        and aspect_ratio
        <= config["standing_bbox_aspect_max"]
        and pose_span_y is not None
        and pose_span_y
        >= config["standing_pose_span_y_min"]
    ):
        return "STANDING", 0.56

    # =========================================================
    # 5. UNKNOWN
    # =========================================================
    #
    # Do not force a state when the evidence is insufficient.
    return "UNKNOWN", 0.0


def derive_bed_occupancy(
    activity: str,
) -> tuple[str, float]:
    """Return a provisional occupancy estimate from activity.

    This is not a spatial bed check.
    Temporal context is handled in the next stage.
    """

    if activity in {
        "LYING_IN_BED",
        "SITTING_ON_BED",
    }:
        return "IN_BED", 0.55

    if activity in {
        "STANDING",
        "WALKING",
    }:
        return "OUT_OF_BED", 0.55

    return "UNKNOWN", 0.0


def classify_pose_manifest(
    pose_manifest_path: Path,
    output_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify every pose observation and save state_manifest.json."""

    pose_manifest_path = Path(pose_manifest_path)
    output_dir = Path(output_dir)

    if not pose_manifest_path.is_file():
        raise FileNotFoundError(
            f"Pose manifest not found: {pose_manifest_path}"
        )

    with pose_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    observations = manifest.get("observations")

    if not isinstance(observations, list):
        raise ValueError(
            "pose_manifest.json does not contain "
            "an observations list."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_observations: list[dict[str, Any]] = []

    previous_observation: dict[str, Any] | None = None
    previous_timestamp: float | None = None

    for observation in observations:

        timestamp = float(
            observation.get(
                "timestamp_sec",
                0.0,
            )
        )

        if previous_timestamp is None:
            delta_time = 1.0

        else:
            delta_time = max(
                0.001,
                timestamp - previous_timestamp,
            )

        features = extract_pose_features(
            observation=observation,
            previous_observation=previous_observation,
            delta_time_sec=delta_time,
            min_keypoint_confidence=float(
                (thresholds or {}).get(
                    "keypoint_confidence",
                    DEFAULT_THRESHOLDS[
                        "keypoint_confidence"
                    ],
                )
            ),
        )

        activity, activity_confidence = (
            classify_activity(
                features,
                thresholds=thresholds,
            )
        )

        bed_occupancy, occupancy_confidence = (
            derive_bed_occupancy(activity)
        )

        state_observations.append(
            {
                "sample_number": observation.get(
                    "sample_number"
                ),
                "timestamp_sec": timestamp,
                "image": observation.get(
                    "image"
                ),
                "person_detected": bool(
                    observation.get(
                        "person_detected"
                    )
                ),
                "activity": activity,
                "activity_confidence": round(
                    activity_confidence,
                    4,
                ),
                "bed_occupancy": bed_occupancy,
                "bed_occupancy_confidence": round(
                    occupancy_confidence,
                    4,
                ),
                "bed_occupancy_is_provisional": True,
                "features": features,
            }
        )

        previous_observation = observation
        previous_timestamp = timestamp

    result = {
        "source_pose_manifest": str(
            pose_manifest_path.resolve()
        ),
        "classifier": (
            "robust_rule_based_pose_baseline_v2"
        ),
        "bed_occupancy_note": (
            "Provisional only: occupancy is inferred "
            "from activity, not verified against a "
            "bed-region polygon. Temporal context is "
            "handled in the next stage."
        ),
        "thresholds": {
            **DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        },
        "frame_count": len(
            state_observations
        ),
        "observations": state_observations,
    }

    output_path = (
        output_dir / "state_manifest.json"
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

    return result