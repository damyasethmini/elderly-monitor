"""Feature extraction from YOLO Pose observations.

This module converts raw YOLO keypoints into simple geometric features.
It tolerates partially visible shoulders/hips by averaging whichever
keypoints are valid, instead of requiring both left and right points.
"""

from __future__ import annotations

from math import atan2, degrees, hypot
from typing import Any


def _valid_point(
    keypoints: dict[str, Any],
    name: str,
    min_confidence: float = 0.30,
) -> tuple[float, float] | None:
    """Return (x, y) if a keypoint exists and meets the confidence threshold."""
    point = keypoints.get(name)
    if not isinstance(point, dict):
        return None

    x = point.get("x")
    y = point.get("y")
    confidence = point.get("confidence")

    if x is None or y is None:
        return None

    if confidence is not None and float(confidence) < min_confidence:
        return None

    return float(x), float(y)


def _average_available(
    points: list[tuple[float, float] | None],
) -> tuple[float, float] | None:
    """Average all available points; return None if none are available."""
    valid = [point for point in points if point is not None]
    if not valid:
        return None

    return (
        sum(point[0] for point in valid) / len(valid),
        sum(point[1] for point in valid) / len(valid),
    )


def _empty_features(person_detected: bool = False) -> dict[str, Any]:
    """Return a consistent feature dictionary when a person/box is unavailable."""
    return {
        "person_detected": person_detected,
        "bbox_width": None,
        "bbox_height": None,
        "bbox_aspect_ratio": None,
        "bbox_center_x": None,
        "bbox_center_y": None,
        "body_axis_angle_deg": None,
        "shoulder_hip_vertical_gap_norm": None,
        "shoulder_hip_horizontal_gap_norm": None,
        "torso_length_norm": None,
        "knee_hip_vertical_gap_norm": None,
        "pose_span_y_norm": None,
        "valid_keypoint_count": 0,
        "motion_norm_per_sec": None,
    }


def extract_pose_features(
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None = None,
    delta_time_sec: float = 1.0,
    min_keypoint_confidence: float = 0.30,
) -> dict[str, Any]:
    """Extract geometric and movement features from one pose observation."""
    if not observation.get("person_detected"):
        return _empty_features(person_detected=False)

    bbox = observation.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return _empty_features(person_detected=True)

    x1, y1, x2, y2 = [float(value) for value in bbox]
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    keypoints = observation.get("keypoints", {})

    left_shoulder = _valid_point(keypoints, "left_shoulder", min_keypoint_confidence)
    right_shoulder = _valid_point(keypoints, "right_shoulder", min_keypoint_confidence)
    left_hip = _valid_point(keypoints, "left_hip", min_keypoint_confidence)
    right_hip = _valid_point(keypoints, "right_hip", min_keypoint_confidence)
    left_knee = _valid_point(keypoints, "left_knee", min_keypoint_confidence)
    right_knee = _valid_point(keypoints, "right_knee", min_keypoint_confidence)
    left_ankle = _valid_point(keypoints, "left_ankle", min_keypoint_confidence)
    right_ankle = _valid_point(keypoints, "right_ankle", min_keypoint_confidence)

    # IMPORTANT FIX: use whichever shoulder/hip keypoints are visible.
    # The previous midpoint helper required both left and right points.
    shoulder_center = _average_available([left_shoulder, right_shoulder])
    hip_center = _average_available([left_hip, right_hip])
    knee_center = _average_available([left_knee, right_knee])
    ankle_center = _average_available([left_ankle, right_ankle])

    body_axis_angle_deg = None
    shoulder_hip_vertical_gap_norm = None
    shoulder_hip_horizontal_gap_norm = None
    torso_length_norm = None

    if shoulder_center is not None and hip_center is not None:
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        body_axis_angle_deg = abs(degrees(atan2(dx, dy)))
        shoulder_hip_vertical_gap_norm = abs(dy) / bbox_height
        shoulder_hip_horizontal_gap_norm = abs(dx) / bbox_width
        torso_length_norm = hypot(dx, dy) / bbox_height

    knee_hip_vertical_gap_norm = None
    if knee_center is not None and hip_center is not None:
        knee_hip_vertical_gap_norm = abs(knee_center[1] - hip_center[1]) / bbox_height

    body_points = [
        left_shoulder, right_shoulder,
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    ]
    y_values = [point[1] for point in body_points if point is not None]
    pose_span_y_norm = None
    if len(y_values) >= 2:
        pose_span_y_norm = (max(y_values) - min(y_values)) / bbox_height

    valid_keypoint_count = sum(point is not None for point in body_points)

    motion_norm_per_sec = None
    if previous_observation is not None and delta_time_sec > 0:
        previous_bbox = previous_observation.get("bbox_xyxy")
        previous_detected = previous_observation.get("person_detected")

        if previous_detected and previous_bbox and len(previous_bbox) == 4:
            px1, py1, px2, py2 = [float(value) for value in previous_bbox]
            previous_center_x = (px1 + px2) / 2.0
            previous_center_y = (py1 + py2) / 2.0
            center_distance = hypot(
                bbox_center_x - previous_center_x,
                bbox_center_y - previous_center_y,
            )
            previous_height = max(1.0, py2 - py1)
            motion_norm_per_sec = center_distance / previous_height / delta_time_sec

    return {
        "person_detected": True,
        "bbox_width": round(bbox_width, 3),
        "bbox_height": round(bbox_height, 3),
        "bbox_aspect_ratio": round(bbox_width / bbox_height, 4),
        "bbox_center_x": round(bbox_center_x, 3),
        "bbox_center_y": round(bbox_center_y, 3),
        "body_axis_angle_deg": (
            round(body_axis_angle_deg, 3)
            if body_axis_angle_deg is not None else None
        ),
        "shoulder_hip_vertical_gap_norm": (
            round(shoulder_hip_vertical_gap_norm, 4)
            if shoulder_hip_vertical_gap_norm is not None else None
        ),
        "shoulder_hip_horizontal_gap_norm": (
            round(shoulder_hip_horizontal_gap_norm, 4)
            if shoulder_hip_horizontal_gap_norm is not None else None
        ),
        "torso_length_norm": (
            round(torso_length_norm, 4)
            if torso_length_norm is not None else None
        ),
        "knee_hip_vertical_gap_norm": (
            round(knee_hip_vertical_gap_norm, 4)
            if knee_hip_vertical_gap_norm is not None else None
        ),
        "pose_span_y_norm": (
            round(pose_span_y_norm, 4)
            if pose_span_y_norm is not None else None
        ),
        "valid_keypoint_count": valid_keypoint_count,
        "motion_norm_per_sec": (
            round(motion_norm_per_sec, 4)
            if motion_norm_per_sec is not None else None
        ),
    }
