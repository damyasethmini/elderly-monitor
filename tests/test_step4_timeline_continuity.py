import unittest

from src.state.smoothing import build_stable_segments


class Step4TimelineContinuityTests(unittest.TestCase):
    def test_short_segment_merging_does_not_create_timeline_gaps(self):
        # This alternating pattern reproduced a real gap seen in the
        # turning_in_bed evaluation video with the previous merge loop.
        labels = [
            "UNKNOWN",
            "LYING_IN_BED",
            "SITTING_ON_BED",
            "LYING_IN_BED",
            "UNKNOWN",
            "LYING_IN_BED",
        ]

        observations = []
        for index, activity in enumerate(labels):
            observations.append(
                {
                    "timestamp_sec": index * 0.5,
                    "final_activity": activity,
                    "final_bed_occupancy": (
                        "UNKNOWN" if activity == "UNKNOWN" else "IN_BED"
                    ),
                    "smoothed_activity_confidence": 0.8,
                }
            )

        segments = build_stable_segments(
            observations,
            min_state_duration_sec=2.0,
        )

        self.assertTrue(segments)
        self.assertEqual(segments[0]["start_sec"], 0.0)
        self.assertEqual(segments[-1]["end_sec"], 3.0)

        for previous, following in zip(segments, segments[1:]):
            self.assertAlmostEqual(
                previous["end_sec"],
                following["start_sec"],
                places=3,
            )

        total_duration = sum(
            float(segment["duration_sec"])
            for segment in segments
        )
        self.assertAlmostEqual(total_duration, 3.0, places=3)


if __name__ == "__main__":
    unittest.main()
