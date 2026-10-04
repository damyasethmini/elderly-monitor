import unittest

from src.alerts.contextual_alert import DEFAULT_ALERTS, build_contextual_alert


ALERTS = dict(DEFAULT_ALERTS)


def segment(
    activity,
    start,
    end,
    inside_start=None,
    inside_end=None,
    outside=False,
):
    result = {
        "activity": activity,
        "bed_occupancy": (
            "IN_BED" if activity in {"LYING_IN_BED", "SITTING_ON_BED"}
            else "OUT_OF_BED" if activity in {"STANDING", "WALKING", "SITTING_OUTSIDE_BED"}
            else "UNKNOWN"
        ),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
    }
    if inside_start is not None:
        result["bed_region_inside_at_start"] = inside_start
    if inside_end is not None:
        result["bed_region_inside_at_end"] = inside_end
    if outside:
        result["bed_region_contains_outside_point"] = True
    return result


def timeline(segments, final_state=None):
    return {
        "observation_duration_sec": segments[-1]["end_sec"] if segments else 0,
        "final_state": final_state or (segments[-1]["activity"] if segments else "UNKNOWN"),
        "segments": segments,
    }


def analysis(events=None, flags=None, decision="NORMAL"):
    return {
        "analysis_version": "step5_local_agent_v1",
        "overall_decision": decision,
        "observation_flags": flags or [],
        "event_analyses": events or [],
    }


class Step6ContextualAlertTests(unittest.TestCase):
    def test_normal_timeline_is_normal(self):
        tl = timeline([
            segment("LYING_IN_BED", 0, 20, True, True),
            segment("SITTING_ON_BED", 20, 25, True, True),
        ])
        report = build_contextual_alert(tl, analysis(), ALERTS)
        self.assertEqual(report["overall_decision"], "NORMAL")
        self.assertEqual(report["trigger_count"], 0)

    def test_confirmed_bed_exit_is_monitored(self):
        tl = timeline([
            segment("SITTING_ON_BED", 0, 5, True, True),
            segment("WALKING", 5, 10, False, False, True),
        ])
        events = [{
            "event_id": "bed_exit_001",
            "event_type": "BED_EXIT",
            "decision": "MONITOR",
            "context_support": "STRONG",
            "confidence": 0.8,
        }]
        report = build_contextual_alert(tl, analysis(events), ALERTS)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertTrue(any(t["flag"] == "EVENT_DECISION" for t in report["triggers"]))

    def test_prolonged_out_of_bed_reaches_alert(self):
        tl = timeline([
            segment("SITTING_ON_BED", 0, 5, True, True),
            segment("WALKING", 5, 305, False, False, True),
            segment("STANDING", 305, 605, False, False, True),
        ])
        report = build_contextual_alert(tl, analysis(), ALERTS)
        self.assertEqual(report["overall_decision"], "ALERT")
        self.assertTrue(any(t["flag"] == "PROLONGED_OUT_OF_BED" for t in report["triggers"]))
        self.assertTrue(report["requires_human_review"])

    def test_unknown_breaks_confirmed_out_of_bed_episode(self):
        tl = timeline([
            segment("WALKING", 0, 20, False, False, True),
            segment("UNKNOWN", 20, 55),
            segment("STANDING", 55, 75, False, False, True),
        ])
        report = build_contextual_alert(tl, analysis(), ALERTS)
        self.assertEqual(report["summary"]["longest_continuous_out_of_bed_sec"], 20)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertTrue(any(t["flag"] == "PROLONGED_UNKNOWN_STATE" for t in report["triggers"]))

    def test_long_sitting_is_monitor(self):
        tl = timeline([segment("SITTING_ON_BED", 0, 125, True, True)])
        report = build_contextual_alert(tl, analysis(), ALERTS)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertTrue(any(t["flag"] == "PROLONGED_SITTING_ON_BED" for t in report["triggers"]))

    def test_lying_outside_bed_region_is_monitor(self):
        tl = timeline([segment("LYING_IN_BED", 0, 4, False, False, True)])
        report = build_contextual_alert(tl, analysis(), ALERTS)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertTrue(any(t["flag"] == "LYING_POSTURE_OUTSIDE_BED_REGION" for t in report["triggers"]))

    def test_upstream_alert_is_preserved(self):
        tl = timeline([segment("LYING_IN_BED", 0, 5, True, True)])
        events = [{
            "event_id": "alert_event",
            "event_type": "BED_EXIT",
            "decision": "ALERT",
            "requires_human_review": False,
        }]
        report = build_contextual_alert(tl, analysis(events, decision="ALERT"), ALERTS)
        self.assertEqual(report["overall_decision"], "ALERT")
        self.assertTrue(report["requires_human_review"])

    def test_threshold_order_is_normalized(self):
        tl = timeline([segment("WALKING", 0, 100, False, False, True)])
        custom = dict(ALERTS)
        custom["prolonged_out_of_bed_monitor_sec"] = 500
        custom["prolonged_out_of_bed_alert_sec"] = 100
        report = build_contextual_alert(tl, analysis(), custom)
        self.assertEqual(
            report["thresholds"]["prolonged_out_of_bed_alert_sec"],
            500,
        )
        self.assertEqual(report["overall_decision"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
