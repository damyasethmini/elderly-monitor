"""Focused regression tests for Step 4 bed-event logic (stdlib unittest only)."""

import unittest

from src.events.bed_events import (
    _episode_movement_evidence,
    build_event_summary,
    detect_bed_events,
)


BED_REGION = {
    "method": "test_rectangle",
    "min_x": 0.0,
    "max_x": 100.0,
    "min_y": 0.0,
    "max_y": 100.0,
}


def observation(timestamp, x, y=50.0, width=100.0, activity="UNKNOWN", occupancy="UNKNOWN"):
    return {
        "timestamp_sec": timestamp,
        "final_activity": activity,
        "final_bed_occupancy": occupancy,
        "features": {
            "bbox_center_x": x,
            "bbox_center_y": y,
            "bbox_width": width,
        },
    }


def exit_timeline(spatial_evidence=True):
    """An exit whose movement is gradual across three short segments."""
    outside = spatial_evidence
    segments = [
        {
            "activity": "LYING_IN_BED", "bed_occupancy": "IN_BED",
            "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0,
            "mean_confidence": 0.9,
            "bed_region_contains_outside_point": False,
        },
        {
            "activity": "SITTING_ON_BED", "bed_occupancy": "IN_BED",
            "start_sec": 1.0, "end_sec": 3.0, "duration_sec": 2.0,
            "mean_confidence": 0.9,
            "bed_region_contains_outside_point": False,
            "bed_region_inside_at_start": True,
            "bed_region_inside_at_end": True,
        },
        {
            "activity": "WALKING", "bed_occupancy": "OUT_OF_BED",
            "start_sec": 3.0, "end_sec": 3.5, "duration_sec": 0.5,
            "mean_confidence": 0.8,
            "bed_region_contains_outside_point": outside,
            "movement_away_from_bed_norm": 0.0,
            "movement_away_from_bed_px": 0.0,
        },
        {
            "activity": "STANDING", "bed_occupancy": "OUT_OF_BED",
            "start_sec": 3.5, "end_sec": 4.0, "duration_sec": 0.5,
            "mean_confidence": 0.8,
            "bed_region_contains_outside_point": outside,
            "movement_away_from_bed_norm": 0.0,
            "movement_away_from_bed_px": 0.0,
        },
        {
            "activity": "WALKING", "bed_occupancy": "OUT_OF_BED",
            "start_sec": 4.0, "end_sec": 5.0, "duration_sec": 1.0,
            "mean_confidence": 0.8,
            "bed_region_contains_outside_point": outside,
            "movement_away_from_bed_norm": 0.0,
            "movement_away_from_bed_px": 0.0,
        },
    ]
    observations = [
        observation(0.5, 50.0, activity="LYING_IN_BED", occupancy="IN_BED"),
        observation(1.5, 50.0, activity="SITTING_ON_BED", occupancy="IN_BED"),
        observation(2.0, 50.0, activity="SITTING_ON_BED", occupancy="IN_BED"),
        observation(2.5, 50.0, activity="SITTING_ON_BED", occupancy="IN_BED"),
        observation(3.0, 120.0, activity="WALKING", occupancy="OUT_OF_BED"),
        observation(3.5, 140.0, activity="STANDING", occupancy="OUT_OF_BED"),
        observation(4.0, 160.0, activity="WALKING", occupancy="OUT_OF_BED"),
        observation(4.5, 175.0, activity="WALKING", occupancy="OUT_OF_BED"),
    ]
    return {
        "bed_region": BED_REGION,
        "observations": observations,
        "segments": segments,
        "bed_summary": {
            "time_in_bed_sec": 3.0,
            "time_out_of_bed_sec": 2.0,
            "unknown_bed_time_sec": 0.0,
        },
    }


class Step4BedEventTests(unittest.TestCase):
    def test_episode_movement_accumulates_across_short_segments(self):
        timeline = exit_timeline()
        evidence = _episode_movement_evidence(
            timeline,
            timeline["segments"],
            start_index=2,
            final_index=4,
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(
            evidence["movement_measurement_scope"],
            "whole_out_of_bed_episode",
        )
        self.assertAlmostEqual(evidence["movement_away_from_bed_px"], 75.0)
        self.assertAlmostEqual(evidence["movement_away_from_bed_norm"], 0.75)

    def test_exit_is_detected_even_when_each_segment_has_zero_local_movement(self):
        events = detect_bed_events(
            exit_timeline(),
            min_stable_duration_sec=1.5,
            min_out_of_bed_confirmation_sec=1.5,
            min_return_confirmation_sec=1.0,
            min_movement_away_from_bed_norm=0.15,
        )
        exits = [event for event in events if event["event_type"] == "BED_EXIT"]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["movement_measurement_scope"], "whole_out_of_bed_episode")
        self.assertGreater(exits[0]["movement_away_from_bed_norm"], 0.15)
        self.assertEqual(exits[0]["out_of_bed_duration_sec"], 2.0)

    def test_no_exit_without_spatial_outside_evidence(self):
        events = detect_bed_events(
            exit_timeline(spatial_evidence=False),
            min_stable_duration_sec=1.5,
            min_out_of_bed_confirmation_sec=1.5,
            min_return_confirmation_sec=1.0,
            min_movement_away_from_bed_norm=0.15,
        )
        self.assertFalse(any(event["event_type"] == "BED_EXIT" for event in events))

    def test_return_uses_segment_start_end_spatial_fields(self):
        timeline = {
            "bed_region": BED_REGION,
            "observations": [],
            "segments": [
                {
                    "activity": "WALKING", "bed_occupancy": "OUT_OF_BED",
                    "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0,
                    "mean_confidence": 0.8,
                    "bed_region_contains_outside_point": True,
                },
                {
                    "activity": "SITTING_ON_BED", "bed_occupancy": "IN_BED",
                    "start_sec": 2.0, "end_sec": 3.5, "duration_sec": 1.5,
                    "mean_confidence": 0.8,
                    "bed_region_inside_at_start": True,
                    "bed_region_inside_at_end": True,
                },
                {
                    "activity": "LYING_IN_BED", "bed_occupancy": "IN_BED",
                    "start_sec": 3.5, "end_sec": 6.0, "duration_sec": 2.5,
                    "mean_confidence": 0.9,
                    "bed_region_inside_at_start": True,
                    "bed_region_inside_at_end": True,
                },
            ],
        }
        events = detect_bed_events(
            timeline,
            min_stable_duration_sec=2.0,
            min_out_of_bed_confirmation_sec=1.5,
            min_return_confirmation_sec=1.0,
            min_movement_away_from_bed_norm=0.15,
        )
        returns = [event for event in events if event["event_type"] == "RETURN_TO_BED"]
        self.assertEqual(len(returns), 1)
        self.assertEqual(returns[0]["confirmed_time_sec"], 3.5)

    def test_return_is_not_confirmed_without_positive_spatial_evidence(self):
        timeline = {
            "bed_region": BED_REGION,
            "observations": [],
            "segments": [
                {
                    "activity": "WALKING", "bed_occupancy": "OUT_OF_BED",
                    "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0,
                    "mean_confidence": 0.8,
                    "bed_region_contains_outside_point": True,
                },
                {
                    "activity": "SITTING_ON_BED", "bed_occupancy": "IN_BED",
                    "start_sec": 2.0, "end_sec": 3.5, "duration_sec": 1.5,
                    "mean_confidence": 0.8,
                    "bed_region_inside_at_start": False,
                    "bed_region_inside_at_end": False,
                },
                {
                    "activity": "LYING_IN_BED", "bed_occupancy": "IN_BED",
                    "start_sec": 3.5, "end_sec": 6.0, "duration_sec": 2.5,
                    "mean_confidence": 0.9,
                    "bed_region_inside_at_start": True,
                    "bed_region_inside_at_end": True,
                },
            ],
        }
        events = detect_bed_events(
            timeline,
            min_stable_duration_sec=2.0,
            min_out_of_bed_confirmation_sec=1.5,
            min_return_confirmation_sec=1.0,
            min_movement_away_from_bed_norm=0.15,
        )
        self.assertFalse(
            any(event["event_type"] == "RETURN_TO_BED" for event in events)
        )

    def test_unknown_breaks_longest_confirmed_out_of_bed_period(self):
        timeline = {
            "bed_summary": {},
            "segments": [
                {"bed_occupancy": "OUT_OF_BED", "duration_sec": 0.5},
                {"bed_occupancy": "OUT_OF_BED", "duration_sec": 0.7},
                {"bed_occupancy": "UNKNOWN", "duration_sec": 0.5},
                {"bed_occupancy": "OUT_OF_BED", "duration_sec": 1.2},
            ],
        }
        summary = build_event_summary(timeline, [])
        self.assertEqual(summary["longest_out_of_bed_period_sec"], 1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
