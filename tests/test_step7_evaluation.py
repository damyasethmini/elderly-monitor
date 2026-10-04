import unittest

from src.evaluation.evaluate import activity_metrics, duration_metrics, event_metrics


class Step7EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gt = [
            {"start_sec": 0.0, "end_sec": 5.0, "activity": "LYING_IN_BED", "bed_occupancy": "IN_BED"},
            {"start_sec": 5.0, "end_sec": 7.0, "activity": "SITTING_ON_BED", "bed_occupancy": "IN_BED"},
            {"start_sec": 7.0, "end_sec": 10.0, "activity": "WALKING", "bed_occupancy": "OUT_OF_BED"},
        ]
        self.pred = [
            {"start_sec": 0.0, "end_sec": 4.0, "activity": "LYING_IN_BED", "bed_occupancy": "IN_BED"},
            {"start_sec": 4.0, "end_sec": 7.0, "activity": "SITTING_ON_BED", "bed_occupancy": "IN_BED"},
            {"start_sec": 7.0, "end_sec": 10.0, "activity": "WALKING", "bed_occupancy": "OUT_OF_BED"},
        ]

    def test_activity_accuracy_is_duration_weighted(self):
        report = activity_metrics(self.gt, self.pred, 10.0)
        self.assertAlmostEqual(report["accuracy"], 0.9)

    def test_duration_error_is_reported(self):
        report = duration_metrics(self.gt, self.pred)
        self.assertEqual(report["by_state"]["LYING_IN_BED"]["error_sec"], -1.0)
        self.assertEqual(report["by_state"]["SITTING_ON_BED"]["error_sec"], 1.0)

    def test_event_matching_within_tolerance(self):
        gt = [{"event_type": "BED_EXIT", "start_time_sec": 7.0}]
        pred = [{"event_type": "BED_EXIT", "start_time_sec": 8.5}]
        report = event_metrics(gt, pred, 2.0)
        self.assertEqual(report["BED_EXIT"]["true_positive"], 1)
        self.assertEqual(report["BED_EXIT"]["false_positive"], 0)
        self.assertEqual(report["BED_EXIT"]["false_negative"], 0)

    def test_event_outside_tolerance_is_missed(self):
        gt = [{"event_type": "BED_EXIT", "start_time_sec": 7.0}]
        pred = [{"event_type": "BED_EXIT", "start_time_sec": 10.0}]
        report = event_metrics(gt, pred, 2.0)
        self.assertEqual(report["BED_EXIT"]["true_positive"], 0)
        self.assertEqual(report["BED_EXIT"]["false_positive"], 1)
        self.assertEqual(report["BED_EXIT"]["false_negative"], 1)

    def test_confirmation_time_does_not_turn_correct_event_into_miss(self):
        gt = [{
            "event_type": "BED_EXIT",
            "start_time_sec": 14.0,
            "confirmed_time_sec": 20.0,
        }]
        pred = [{
            "event_type": "BED_EXIT",
            "start_time_sec": 14.0,
            "confirmed_time_sec": 15.5,
        }]
        report = event_metrics(gt, pred, 2.0)
        metrics = report["BED_EXIT"]
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["matches"][0]["confirmation_time_absolute_error_sec"], 4.5)
        self.assertEqual(metrics["confirmation_time_mae_sec"], 4.5)


if __name__ == "__main__":
    unittest.main()
