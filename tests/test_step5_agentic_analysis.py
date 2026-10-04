import unittest

from src.agent.agentic_analysis import (
    LANGGRAPH_AVAILABLE,
    _route_context,
    analyze_timeline_and_events,
)


ALERTS = {
    "sitting_on_bed_edge_monitor_sec": 120,
    "prolonged_out_of_bed_monitor_sec": 300,
    "prolonged_out_of_bed_alert_sec": 600,
    "unknown_state_monitor_sec": 30,
}


def segment(activity, start, end, inside_start=None, inside_end=None, outside=False, confidence=0.8):
    return {
        "activity": activity,
        "bed_occupancy": (
            "IN_BED" if activity in {"LYING_IN_BED", "SITTING_ON_BED"}
            else "OUT_OF_BED" if activity in {"STANDING", "WALKING", "SITTING_OUTSIDE_BED"}
            else "UNKNOWN"
        ),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "mean_confidence": confidence,
        "bed_region_inside_at_start": inside_start,
        "bed_region_inside_at_end": inside_end,
        "bed_region_contains_outside_point": outside,
        "movement_away_from_bed_norm": 0.0,
    }


def timeline(segments, final_state=None):
    return {
        "observation_duration_sec": segments[-1]["end_sec"] if segments else 0,
        "final_state": final_state or (segments[-1]["activity"] if segments else "UNKNOWN"),
        "segments": segments,
    }


class Step5AgenticAnalysisTests(unittest.TestCase):
    def test_confirmed_short_bed_exit_is_monitored_with_context(self):
        tl = timeline([
            segment("LYING_IN_BED", 0, 5, True, True),
            segment("SITTING_ON_BED", 5, 8, True, True),
            segment("WALKING", 8, 12, False, False, True),
            segment("STANDING", 12, 18, False, False, True),
        ])
        events = {"events": [{
            "event_id": "bed_exit_001", "event_type": "BED_EXIT",
            "start_time_sec": 8, "confirmed_time_sec": 9.5,
            "from_activity": "SITTING_ON_BED", "to_activity": "WALKING",
            "out_of_bed_duration_sec": 10, "movement_away_from_bed_norm": 0.5,
            "confidence": 0.8, "decision": "MONITOR", "evidence_strength": "FULL_SEQUENCE",
        }], "bed_summary": {}}
        report = analyze_timeline_and_events(tl, events, ALERTS, 0.15)
        self.assertEqual(report["overall_decision"], "MONITOR")
        item = report["event_analyses"][0]
        self.assertEqual(item["context_support"], "STRONG")
        self.assertFalse(item["requires_human_review"])
        self.assertTrue(any(a["action"] == "CHECK_BED_REGION_AND_MOVEMENT" for a in item["agent_actions"]))

    def test_prolonged_continuous_absence_triggers_alert(self):
        tl = timeline([
            segment("SITTING_ON_BED", 0, 5, True, True),
            segment("WALKING", 5, 310, False, False, True),
            segment("STANDING", 310, 620, False, False, True),
        ])
        events = {"events": [{
            "event_id": "bed_exit_001", "event_type": "BED_EXIT",
            "start_time_sec": 5, "confirmed_time_sec": 6.5,
            "out_of_bed_duration_sec": 615, "movement_away_from_bed_norm": 0.8,
            "confidence": 0.9, "decision": "MONITOR", "evidence_strength": "FULL_SEQUENCE",
        }], "bed_summary": {"longest_out_of_bed_period_sec": 615}}
        report = analyze_timeline_and_events(tl, events, ALERTS, 0.15)
        self.assertEqual(report["overall_decision"], "ALERT")
        self.assertTrue(any(flag["decision"] == "ALERT" for flag in report["observation_flags"]))

    def test_unknown_period_breaks_continuous_out_of_bed_duration(self):
        tl = timeline([
            segment("WALKING", 0, 20, False, False, True),
            segment("UNKNOWN", 20, 55),
            segment("STANDING", 55, 75, False, False, True),
        ])
        report = analyze_timeline_and_events(tl, {"events": [], "bed_summary": {}}, ALERTS)
        self.assertEqual(report["summary"]["longest_continuous_out_of_bed_sec"], 20)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertEqual(report["observation_flags"][0]["flag"], "PROLONGED_UNKNOWN_STATE")

    def test_lying_posture_outside_bed_region_requires_review(self):
        tl = timeline([segment("LYING_IN_BED", 0, 4, False, False, True)])
        report = analyze_timeline_and_events(tl, {"events": [], "bed_summary": {}}, ALERTS)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertEqual(report["observation_flags"][0]["flag"], "LYING_POSTURE_OUTSIDE_BED_REGION")

    def test_unremarkable_timeline_is_normal(self):
        tl = timeline([
            segment("LYING_IN_BED", 0, 20, True, True),
            segment("SITTING_ON_BED", 20, 25, True, True),
        ])
        report = analyze_timeline_and_events(tl, {"events": [], "bed_summary": {}}, ALERTS)
        self.assertEqual(report["overall_decision"], "NORMAL")
        self.assertEqual(report["observation_flags"], [])

    def test_ambiguous_exit_expands_temporal_context(self):
        tl = timeline([
            segment("UNKNOWN", 0, 2),
            segment("WALKING", 2, 4, False, False, True),
            segment("UNKNOWN", 4, 6),
        ])
        events = {"events": [{
            "event_id": "bed_exit_ambiguous", "event_type": "BED_EXIT",
            "start_time_sec": 2, "confirmed_time_sec": 3.5,
            "out_of_bed_duration_sec": 2, "movement_away_from_bed_norm": 0.0,
            "confidence": 0.5, "decision": "MONITOR", "evidence_strength": "TRANSITION_ONLY",
        }], "bed_summary": {}}
        report = analyze_timeline_and_events(tl, events, ALERTS, 0.15)
        actions = report["event_analyses"][0]["agent_actions"]
        self.assertTrue(any(a["action"] == "EXPAND_TEMPORAL_CONTEXT" for a in actions))
        self.assertTrue(report["event_analyses"][0]["requires_human_review"])
        self.assertEqual(report["overall_decision"], "MONITOR")

    def test_prolonged_sitting_on_bed_triggers_conservative_monitor(self):
        tl = timeline([segment("SITTING_ON_BED", 0, 125, True, True)])
        report = analyze_timeline_and_events(tl, {"events": [], "bed_summary": {}}, ALERTS)
        self.assertEqual(report["overall_decision"], "MONITOR")
        self.assertEqual(report["observation_flags"][0]["flag"], "PROLONGED_SITTING_ON_BED")
        self.assertIn("cannot distinguish the bed edge", report["observation_flags"][0]["reason"])

    def test_context_analysis_does_not_inflate_event_confidence(self):
        tl = timeline([
            segment("SITTING_ON_BED", 0, 2, True, True),
            segment("WALKING", 2, 5, False, False, True),
        ])
        events = {"events": [{
            "event_id": "exit_low_conf", "event_type": "BED_EXIT",
            "start_time_sec": 2, "confirmed_time_sec": 3,
            "out_of_bed_duration_sec": 3, "movement_away_from_bed_norm": 0.5,
            "confidence": 0.51, "decision": "MONITOR", "evidence_strength": "FULL_SEQUENCE",
        }], "bed_summary": {}}
        report = analyze_timeline_and_events(tl, events, ALERTS, 0.15)
        self.assertEqual(report["event_analyses"][0]["confidence"], 0.51)

    def test_existing_alert_is_never_downgraded(self):
        tl = timeline([segment("LYING_IN_BED", 0, 5, True, True)])
        events = {"events": [{
            "event_id": "alert_event", "event_type": "BED_EXIT",
            "start_time_sec": 0, "confidence": 0.9, "decision": "ALERT",
            "out_of_bed_duration_sec": 2,
        }], "bed_summary": {}}
        report = analyze_timeline_and_events(tl, events, ALERTS, 0.15)
        self.assertEqual(report["event_analyses"][0]["decision"], "ALERT")
        self.assertEqual(report["overall_decision"], "ALERT")

    def test_report_identifies_agent_framework_and_no_external_llm(self):
        tl = timeline([
            segment("LYING_IN_BED", 0, 5, True, True),
            segment("SITTING_ON_BED", 5, 8, True, True),
        ])
        report = analyze_timeline_and_events(tl, {"events": [], "bed_summary": {}}, ALERTS)
        self.assertEqual(report["uses_external_llm_api"], False)
        self.assertIn("agent_framework", report)
        self.assertIn("uses_langgraph", report)
        self.assertEqual(report["uses_langgraph"], LANGGRAPH_AVAILABLE)

    def test_ambiguous_context_routes_to_expansion(self):
        state = {"context_is_ambiguous": True}
        self.assertEqual(_route_context(state), "expand")

    def test_clear_context_routes_directly_to_spatial_check(self):
        state = {"context_is_ambiguous": False}
        self.assertEqual(_route_context(state), "spatial")


if __name__ == "__main__":
    unittest.main()
