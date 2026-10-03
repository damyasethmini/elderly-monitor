from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PRIMARY_STATES = [
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "UNKNOWN",
]

BED_OCCUPANCY_STATES = ["IN_BED", "OUT_OF_BED", "UNKNOWN"]
EVENT_TYPES = ["BED_EXIT", "RETURN_TO_BED"]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_interval(item: dict[str, Any], index: int, label: str) -> tuple[float, float]:
    try:
        start = float(item["start_sec"])
        end = float(item["end_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} interval at index {index}: {item}") from exc
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"Invalid {label} interval at index {index}: {item}")
    return start, end


def _normalise_segments(items: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        start, end = _validate_interval(item, index, label)
        result.append({**item, "start_sec": start, "end_sec": end})
    result.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return result


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _coverage_duration(segments: list[dict[str, Any]], label_key: str, labels: list[str]) -> dict[str, float]:
    totals = {label: 0.0 for label in labels}
    for segment in segments:
        label = str(segment.get(label_key, "UNKNOWN"))
        if label not in totals:
            label = "UNKNOWN" if "UNKNOWN" in totals else labels[-1]
        totals[label] += float(segment["end_sec"]) - float(segment["start_sec"])
    return totals


def _label_at(segments: list[dict[str, Any]], timestamp: float, key: str) -> str:
    for segment in segments:
        if segment["start_sec"] <= timestamp < segment["end_sec"]:
            return str(segment.get(key, "UNKNOWN"))
    return "UNKNOWN"


def _sample_axis(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]], end_sec: float, resolution_sec: float) -> tuple[Counter, Counter, int]:
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    total = 0
    t = 0.0
    while t < end_sec:
        center = min(t + resolution_sec / 2.0, max(0.0, end_sec - 1e-9))
        gt = _label_at(gt_segments, center, "activity")
        pred = _label_at(pred_segments, center, "activity")
        confusion[(gt, pred)] += 1
        correct += int(gt == pred)
        total += 1
        t += resolution_sec
    return confusion, Counter({"correct": correct}), total


def _interval_weighted_confusion(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]], end_sec: float) -> Counter[tuple[str, str]]:
    boundaries = {0.0, end_sec}
    for segment in gt_segments + pred_segments:
        boundaries.add(max(0.0, min(end_sec, float(segment["start_sec"]))))
        boundaries.add(max(0.0, min(end_sec, float(segment["end_sec"]))))
    points = sorted(boundaries)
    confusion: Counter[tuple[str, str]] = Counter()
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        gt = _label_at(gt_segments, midpoint, "activity")
        pred = _label_at(pred_segments, midpoint, "activity")
        confusion[(gt, pred)] += right - left
    return confusion


def activity_metrics(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]], observation_end_sec: float) -> dict[str, Any]:
    confusion = _interval_weighted_confusion(gt_segments, pred_segments, observation_end_sec)
    total = sum(confusion.values())
    correct = sum(value for (gt, pred), value in confusion.items() if gt == pred)
    accuracy = correct / total if total else 0.0

    per_state: dict[str, dict[str, float]] = {}
    labels = list(PRIMARY_STATES)
    for state in labels:
        tp = confusion[(state, state)]
        fp = sum(value for (gt, pred), value in confusion.items() if pred == state and gt != state)
        fn = sum(value for (gt, pred), value in confusion.items() if gt == state and pred != state)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_state[state] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ground_truth_sec": round(tp + fn, 3),
            "predicted_sec": round(tp + fp, 3),
        }

    matrix = {gt: {pred: round(confusion[(gt, pred)], 3) for pred in labels} for gt in labels}
    return {
        "accuracy": round(accuracy, 4),
        "evaluated_duration_sec": round(total, 3),
        "per_state": per_state,
        "confusion_matrix_sec": matrix,
    }


def bed_occupancy_metrics(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]], observation_end_sec: float) -> dict[str, Any]:
    gt = _normalise_segments(gt_segments, "ground truth occupancy")
    pred = _normalise_segments(pred_segments, "predicted occupancy")
    boundaries = {0.0, observation_end_sec}
    for segment in gt + pred:
        boundaries.update({max(0.0, min(observation_end_sec, segment["start_sec"])), max(0.0, min(observation_end_sec, segment["end_sec"]))})
    points = sorted(boundaries)
    correct = 0.0
    total = 0.0
    confusion: Counter[tuple[str, str]] = Counter()
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        gt_label = _label_at(gt, midpoint, "bed_occupancy")
        pred_label = _label_at(pred, midpoint, "bed_occupancy")
        duration = right - left
        confusion[(gt_label, pred_label)] += duration
        total += duration
        if gt_label == pred_label:
            correct += duration
    return {
        "accuracy": round(correct / total if total else 0.0, 4),
        "evaluated_duration_sec": round(total, 3),
        "confusion_matrix_sec": {
            gt_label: {pred_label: round(confusion[(gt_label, pred_label)], 3) for pred_label in BED_OCCUPANCY_STATES}
            for gt_label in BED_OCCUPANCY_STATES
        },
    }


def duration_metrics(gt_segments: list[dict[str, Any]], pred_segments: list[dict[str, Any]]) -> dict[str, Any]:
    gt_totals = _coverage_duration(gt_segments, "activity", PRIMARY_STATES)
    pred_totals = _coverage_duration(pred_segments, "activity", PRIMARY_STATES)
    rows = {}
    absolute_errors = []
    for state in PRIMARY_STATES:
        gt_sec = gt_totals[state]
        pred_sec = pred_totals[state]
        error = pred_sec - gt_sec
        absolute = abs(error)
        absolute_errors.append(absolute)
        rows[state] = {
            "ground_truth_sec": round(gt_sec, 3),
            "predicted_sec": round(pred_sec, 3),
            "error_sec": round(error, 3),
            "absolute_error_sec": round(absolute, 3),
        }
    return {"by_state": rows, "mae_sec": round(sum(absolute_errors) / len(absolute_errors), 3)}


def _events_from_input(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("bed_events", data.get("events", []))
    if not isinstance(raw, list):
        raise ValueError("Ground truth JSON must contain a 'bed_events' list.")
    events = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid bed event at index {index}: {item}")
        event_type = str(item.get("event_type", item.get("event", ""))).upper()
        if event_type not in EVENT_TYPES:
            continue
        timestamp = item.get("timestamp_sec", item.get("confirmed_time_sec", item.get("time_sec")))
        if timestamp is None:
            raise ValueError(f"Missing timestamp for ground truth event at index {index}.")
        events.append({"event_type": event_type, "timestamp_sec": float(timestamp)})
    return sorted(events, key=lambda item: item["timestamp_sec"])


def event_metrics(gt_events: list[dict[str, Any]], pred_events: list[dict[str, Any]], tolerance_sec: float) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for event_type in EVENT_TYPES:
        gt = [event for event in gt_events if event["event_type"] == event_type]
        pred = [event for event in pred_events if event["event_type"] == event_type]
        used_gt: set[int] = set()
        matched = []
        false_positive = []
        for p in pred:
            candidates = [
                (idx, abs(p["timestamp_sec"] - g["timestamp_sec"]))
                for idx, g in enumerate(gt)
                if idx not in used_gt and abs(p["timestamp_sec"] - g["timestamp_sec"]) <= tolerance_sec
            ]
            if candidates:
                gt_index, delta = min(candidates, key=lambda item: item[1])
                used_gt.add(gt_index)
                matched.append({"predicted_sec": p["timestamp_sec"], "ground_truth_sec": gt[gt_index]["timestamp_sec"], "absolute_error_sec": round(delta, 3)})
            else:
                false_positive.append(p["timestamp_sec"])
        false_negative = [g["timestamp_sec"] for idx, g in enumerate(gt) if idx not in used_gt]
        tp = len(matched)
        fp = len(false_positive)
        fn = len(false_negative)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[event_type] = {
            "ground_truth_count": len(gt),
            "predicted_count": len(pred),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "matches": matched,
            "false_positive_times_sec": false_positive,
            "missed_ground_truth_times_sec": false_negative,
        }
    return results


def create_ground_truth_template(timeline_path: Path, output_path: Path) -> None:
    timeline = _load_json(timeline_path)
    segments = timeline.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Timeline must contain a 'segments' list.")
    template = {
        "annotation_version": "step7_ground_truth_v1",
        "video": timeline_path.stem,
        "observation_duration_sec": float(timeline.get("observation_duration_sec", 0.0)),
        "states": [
            {
                "start_sec": float(segment.get("start_sec", 0.0)),
                "end_sec": float(segment.get("end_sec", 0.0)),
                "activity": "UNKNOWN",
                "bed_occupancy": "UNKNOWN",
                "note": "Replace UNKNOWN values with manually reviewed ground truth.",
            }
            for segment in segments
        ],
        "bed_events": [],
        "notes": [
            "Ground truth should be manually annotated from the source video, not copied from the prediction.",
            "Use UNKNOWN when the video does not provide enough evidence.",
            "OUT_OF_BED is represented through bed_occupancy so primary activity durations remain mutually exclusive.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(template, file, indent=2, ensure_ascii=False)
        file.write("\n")


def evaluate(timeline_path: Path, events_path: Path, ground_truth_path: Path, event_tolerance_sec: float = 2.0) -> dict[str, Any]:
    timeline = _load_json(timeline_path)
    gt_data = _load_json(ground_truth_path)
    predicted_segments = _normalise_segments(timeline.get("segments", []), "predicted timeline")
    gt_segments = _normalise_segments(gt_data.get("states", []), "ground truth timeline")
    if not gt_segments:
        raise ValueError("Ground truth must contain at least one state segment.")

    duration = float(
        gt_data.get(
            "observation_duration_sec",
            timeline.get("observation_duration_sec", 0.0),
        )
    )
    if duration <= 0:
        duration = max(
            max(float(item["end_sec"]) for item in gt_segments),
            max(float(item["end_sec"]) for item in predicted_segments) if predicted_segments else 0.0,
        )

    predicted_events_data = _load_json(events_path) if events_path.exists() else {"events": []}
    predicted_events = _events_from_input(predicted_events_data)
    gt_events = _events_from_input(gt_data)

    return {
        "evaluation_version": "step7_evaluation_v1",
        "observation_duration_sec": round(duration, 3),
        "activity_recognition": activity_metrics(gt_segments, predicted_segments, duration),
        "bed_occupancy_recognition": bed_occupancy_metrics(gt_segments, predicted_segments, duration),
        "duration_estimation": duration_metrics(gt_segments, predicted_segments),
        "bed_events": event_metrics(gt_events, predicted_events, event_tolerance_sec),
        "event_match_tolerance_sec": event_tolerance_sec,
        "predicted_timeline": str(timeline_path.resolve()),
        "predicted_events": str(events_path.resolve()),
        "ground_truth": str(ground_truth_path.resolve()),
        "notes": [
            "Activity accuracy is duration-weighted over the annotated observation window.",
            "Bed-event matching uses nearest unmatched event within the configured timestamp tolerance.",
            "Duration error is predicted minus ground-truth seconds; absolute_error_sec is its magnitude.",
            "Metrics are only as valid as the manually labelled ground truth and the evaluated video set.",
        ],
    }


def _default_paths(video: Path, output_root: Path) -> tuple[Path, Path]:
    run_dir = output_root / video.stem
    return run_dir / "timeline" / "timeline.json", run_dir / "timeline" / "bed_events.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate elderly-monitor predictions against manually labelled ground truth.")
    parser.add_argument("--video", type=Path, help="Video path used to infer results/<video_stem> paths.")
    parser.add_argument("--timeline", type=Path, help="Predicted timeline.json path.")
    parser.add_argument("--events", type=Path, help="Predicted bed_events.json path.")
    parser.add_argument("--ground-truth", type=Path, help="Manually annotated ground_truth.json path.")
    parser.add_argument("--output", type=Path, help="Output evaluation_report.json path.")
    parser.add_argument("--event-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--make-template", action="store_true", help="Create a ground-truth template from a predicted timeline.")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.video:
            timeline_default, events_default = _default_paths(args.video, args.output_root)
        else:
            timeline_default, events_default = None, None

        timeline_path = args.timeline or timeline_default
        events_path = args.events or events_default
        if timeline_path is None:
            raise ValueError("Provide --timeline or --video.")

        timeline_path = Path(timeline_path)
        if args.make_template:
            output = args.output or Path("evaluation") / f"{timeline_path.stem}_ground_truth_template.json"
            create_ground_truth_template(timeline_path, output)
            print(f"Ground-truth template created: {output}")
            return 0

        if args.ground_truth is None:
            raise ValueError("Provide --ground-truth, or use --make-template first.")
        if events_path is None:
            events_path = Path("missing-bed-events.json")
        report = evaluate(timeline_path, Path(events_path), Path(args.ground_truth), args.event_tolerance_sec)
        output = args.output or Path("evaluation") / f"{timeline_path.stem}_evaluation_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print("Step 7 evaluation completed.")
        print(f"Activity accuracy: {report['activity_recognition']['accuracy']:.4f}")
        for event_type, metrics in report["bed_events"].items():
            print(f"{event_type}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")
        print(f"Duration MAE: {report['duration_estimation']['mae_sec']:.2f}s")
        print(f"Evaluation report: {output.resolve()}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
