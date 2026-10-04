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


"""Conservative rule-based activity classification for elderly-monitor.

This is a baseline, not a medically validated classifier.
Ambiguous or contradictory observations are labelled UNKNOWN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.state.features import extract_pose_features


DEFAULT_THRESHOLDS: dict[str, float] = {
    # Angle is measured from vertical:
    # 0 degrees = vertical torso, about 90 = horizontal torso.
    "lying_body_axis_angle_deg": 55.0,
    "lying_conflicting_bbox_aspect_ratio": 0.90,

    # Sitting requires several consistent signals.
    "sitting_body_axis_max_deg": 35.0,
    "sitting_knee_hip_gap_max": 0.16,
    "sitting_bbox_aspect_max": 1.20,

    # Weak evidence for upright posture.
    "upright_bbox_aspect_max": 1.20,
    "upright_pose_span_y_min": 0.45,

    # Movement alone is not enough to establish walking.
    "walking_motion_threshold": 0.15,
    "walking_bbox_aspect_max": 1.20,

    "keypoint_confidence": 0.30,
}


def classify_activity(
    features: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> tuple[str, float]:
    """Return an activity label and heuristic confidence."""

    config = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if not features.get("person_detected"):
        return "UNKNOWN", 0.0

    body_angle = features.get("body_axis_angle_deg")
    aspect_ratio = features.get("bbox_aspect_ratio")
    knee_hip_gap = features.get("knee_hip_vertical_gap_norm")
    pose_span_y = features.get("pose_span_y_norm")
    motion = features.get("motion_norm_per_sec")

    body_angle = (
        float(body_angle) if body_angle is not None else None
    )
    aspect_ratio = (
        float(aspect_ratio) if aspect_ratio is not None else None
    )
    knee_hip_gap = (
        float(knee_hip_gap) if knee_hip_gap is not None else None
    )
    pose_span_y = (
        float(pose_span_y) if pose_span_y is not None else None
    )
    motion = float(motion) if motion is not None else None

    # 1. LYING
    # Require the torso keypoints to indicate a mostly horizontal body.
    # Do not use a wide bounding box by itself.
    if (
        body_angle is not None
        and body_angle >= config["lying_body_axis_angle_deg"]
    ):
        if (
            aspect_ratio is not None
            and aspect_ratio
            < config["lying_conflicting_bbox_aspect_ratio"]
        ):
            return "UNKNOWN", 0.0

        confidence = (
            0.80
            if aspect_ratio is None or aspect_ratio >= 1.0
            else 0.65
        )
        return "LYING_IN_BED", confidence

    # Conflicting evidence: nearly vertical torso but very wide box.
    if (
        body_angle is not None
        and body_angle < 35.0
        and aspect_ratio is not None
        and aspect_ratio > config["sitting_bbox_aspect_max"]
    ):
        return "UNKNOWN", 0.0

    # 2. SITTING
    # Require an upright torso, knees relatively close to hips vertically,
    # and a bounding box that does not strongly contradict sitting.
    if (
        body_angle is not None
        and body_angle <= config["sitting_body_axis_max_deg"]
        and knee_hip_gap is not None
        and knee_hip_gap <= config["sitting_knee_hip_gap_max"]
        and (
            aspect_ratio is None
            or aspect_ratio <= config["sitting_bbox_aspect_max"]
        )
    ):
        return "SITTING_ON_BED", 0.68

    # Shared evidence for an upright posture.
    upright_box = (
        aspect_ratio is not None
        and aspect_ratio <= config["walking_bbox_aspect_max"]
    )

    sufficiently_vertical_pose = (
        (
            body_angle is not None
            and body_angle <= 45.0
        )
        or (
            body_angle is None
            and pose_span_y is not None
            and pose_span_y >= config["upright_pose_span_y_min"]
            and upright_box
        )
    )

    # 3. WALKING
    # Movement must be accompanied by evidence of an upright posture.
    if (
        motion is not None
        and motion >= config["walking_motion_threshold"]
        and upright_box
        and sufficiently_vertical_pose
    ):
        confidence = 0.62 if body_angle is not None else 0.52
        return "WALKING", confidence

    # 4. STANDING
    # Stronger case: torso angle, vertical pose span, and other features agree.
    if (
        body_angle is not None
        and body_angle <= 25.0
        and pose_span_y is not None
        and pose_span_y >= config["upright_pose_span_y_min"]
        and (
            knee_hip_gap is None
            or knee_hip_gap > config["sitting_knee_hip_gap_max"]
        )
        and (
            aspect_ratio is None
            or aspect_ratio <= config["upright_bbox_aspect_max"]
        )
    ):
        return "STANDING", 0.68

    # Weaker standing estimate when torso keypoints are missing.
    if (
        body_angle is None
        and aspect_ratio is not None
        and aspect_ratio <= config["upright_bbox_aspect_max"]
        and pose_span_y is not None
        and pose_span_y >= config["upright_pose_span_y_min"]
        and (
            motion is None
            or motion < config["walking_motion_threshold"]
        )
        and (
            knee_hip_gap is None
            or knee_hip_gap > config["sitting_knee_hip_gap_max"]
        )
    ):
        return "STANDING", 0.52

    # 5. Insufficient or conflicting evidence.
    return "UNKNOWN", 0.0


def derive_bed_occupancy(
    activity: str,
) -> tuple[str, float]:
    """Return a provisional bed-occupancy estimate.

    This heuristic does not check the actual bed location.
    Spatial verification will be needed before final evaluation.
    """

    if activity in {"LYING_IN_BED", "SITTING_ON_BED"}:
        return "IN_BED", 0.55

    if activity in {"STANDING", "WALKING"}:
        return "OUT_OF_BED", 0.55

    return "UNKNOWN", 0.0


def classify_pose_manifest(
    pose_manifest_path: Path,
    output_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify observations and save state_manifest.json."""

    pose_manifest_path = Path(pose_manifest_path)
    output_dir = Path(output_dir)

    if not pose_manifest_path.is_file():
        raise FileNotFoundError(
            f"Pose manifest not found: {pose_manifest_path}"
        )

    with pose_manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    observations = manifest.get("observations")

    if not isinstance(observations, list):
        raise ValueError(
            "pose_manifest.json does not contain an observations list."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    state_observations: list[dict[str, Any]] = []
    previous_observation: dict[str, Any] | None = None
    previous_timestamp: float | None = None

    for observation in observations:
        timestamp = float(observation.get("timestamp_sec", 0.0))

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
                    DEFAULT_THRESHOLDS["keypoint_confidence"],
                )
            ),
        )

        activity, activity_confidence = classify_activity(
            features,
            thresholds=thresholds,
        )

        bed_occupancy, occupancy_confidence = derive_bed_occupancy(
            activity
        )

        state_observations.append(
            {
                "sample_number": observation.get("sample_number"),
                "timestamp_sec": timestamp,
                "image": observation.get("image"),
                "person_detected": bool(
                    observation.get("person_detected")
                ),
                "activity": activity,
                "activity_confidence": round(
                    activity_confidence, 4
                ),
                "bed_occupancy": bed_occupancy,
                "bed_occupancy_confidence": round(
                    occupancy_confidence, 4
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
        "classifier": "conservative_rule_based_pose_baseline",
        "bed_occupancy_note": (
            "Provisional only: occupancy is inferred from activity, not "
            "verified against a bed-region polygon. Add spatial bed-region "
            "checking before evaluating bed-exit/return events."
        ),
        "thresholds": {
            **DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        },
        "frame_count": len(state_observations),
        "observations": state_observations,
    }

    output_path = output_dir / "state_manifest.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    return result