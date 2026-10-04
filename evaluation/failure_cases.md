# Failure Cases — Evaluated `test_video.mp4`

These examples are based on the manually reviewed ground truth and the generated Step 7 evaluation report. They document model limitations rather than hiding them.

## Failure case 1 — early sitting posture confused with lying/standing

- **Video/time:** `test_video.mp4`, approximately `0.0–3.0 s`
- **Ground truth:** primarily `SITTING_ON_BED`
- **Prediction:** contains `STANDING`, `LYING_IN_BED`, and transition uncertainty
- **What went wrong:** the pose geometry during the beginning of the clip is unusual and several body keypoints are not consistently reliable. The rule-based posture classifier interprets some body-axis/bounding-box evidence as standing or lying.
- **Why the current approach failed:** the classifier is based on fixed pose thresholds and cannot fully understand the bed/person scene context from pose alone.
- **Possible mitigation:** use stronger temporal context, manually labelled bed geometry, or a local VLM only when pose evidence is ambiguous.

## Failure case 2 — walking confused with standing

- **Video/time:** approximately `19.75–22.0 s`
- **Ground truth:** `WALKING`
- **Prediction:** much of this interval is smoothed into `STANDING`; predicted walking duration is substantially shorter than ground truth.
- **What went wrong:** the frame-to-frame movement signal is not consistently high enough to satisfy the walking threshold after smoothing.
- **Why the current approach failed:** walking is inferred using a simple motion heuristic rather than a learned temporal gait model/person tracker.
- **Possible mitigation:** use multi-frame velocity/trajectory features, optical flow, person tracking, or a lightweight temporal classifier.

## Failure case 3 — bed exit confirmed earlier than manual confirmation

- **Video/time:** bed exit starts around `14.0 s`; clear movement away is manually confirmed around `20.0 s`
- **Ground truth:** `BED_EXIT` start `14.0 s`, confirmed around `20.0 s`
- **Prediction:** `BED_EXIT` start `14.0 s`, confirmed `15.5 s`
- **Result:** event detection is correct, but confirmation is about `4.5 s` early.
- **What went wrong:** the event detector considers the combination of transition, outside-bed evidence, episode movement, and minimum confirmation duration sufficient before the manually selected clear movement-away point.
- **Why the current approach failed:** the confirmation rule is intentionally sensitive and heuristic.
- **Possible mitigation:** tune the confirmation duration/movement threshold, require stronger movement-away persistence, or separately calibrate event start and confirmation rules on a larger labelled set.

## Additional limitation — no return-to-bed example

The included clip contains no true `RETURN_TO_BED`. Therefore return precision/recall should be reported as **N/A for this clip**, not interpreted as a measured failure rate. A separate video containing `OUT_OF_BED → approach bed → sit on bed → lie down` is required to evaluate return detection properly.
