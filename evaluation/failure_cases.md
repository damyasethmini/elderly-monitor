# Failure Cases — Multi-Video Evaluation

These examples are based on manually reviewed ground truth and the generated Step 7 evaluation reports. They document real limitations of the current pose-based baseline rather than hiding them.

## Failure case 1 — return-to-bed missed under blanket / bed occlusion

- **Video:** `return_to_bed.mp4`
- **Ground truth:** the person approaches the bed, gets onto the bed, and returns to a lying position.
- **Prediction:** much of the sequence becomes `UNKNOWN`; no `RETURN_TO_BED` event is detected.
- **Observed evaluation:** activity accuracy ≈ **40.95%**, bed-occupancy accuracy ≈ **8.19%**, duration MAE ≈ **2.29 s**, `RETURN_TO_BED` recall **0%**.
- **What went wrong:** YOLO Pose loses reliable body keypoints once the person becomes partially hidden by the bed and blanket.
- **Why the current approach failed:** bed-return recognition depends on pose-derived state and spatial evidence. When pose evidence disappears, the conservative pipeline correctly falls back to `UNKNOWN`, but the event transition cannot be confirmed.
- **Possible mitigation:** combine pose estimation with a dedicated person/bed detector, manually defined bed geometry, tracking across short occlusions, or an optional local VLM fallback for ambiguous frames.

## Failure case 2 — stretching while seated becomes UNKNOWN

- **Video:** `ambiguous_sitting.mp4`
- **Ground truth:** `SITTING_ON_BED` for the clip; the person stretches but does not leave the bed.
- **Prediction:** approximately **8.5 s** is recognized as `SITTING_ON_BED`, while much of the remaining time becomes `UNKNOWN`.
- **Observed evaluation:** activity accuracy ≈ **46.50%**, `SITTING_ON_BED` precision **100%**, recall **46.5%**, duration MAE ≈ **3.30 s**.
- **Positive behavior:** the system correctly produces **no false `BED_EXIT`**.
- **What went wrong:** unusual torso and arm geometry during stretching moves the pose features outside the conservative sitting thresholds.
- **Why the current approach failed:** the posture classifier uses fixed pose rules rather than a learned temporal posture model.
- **Possible mitigation:** use multi-frame posture features, additional torso/hip context, a learned lightweight temporal classifier, or ambiguity-specific scene reasoning.

## Failure case 3 — turning under blankets causes long UNKNOWN periods

- **Video:** `turning_in_bed.mp4`
- **Ground truth:** `LYING_IN_BED` for the full clip.
- **Prediction:** only part of the clip is recognized as `LYING_IN_BED`; most missed time becomes `UNKNOWN`.
- **Observed evaluation:** activity accuracy ≈ **26.58%**, `LYING_IN_BED` precision **100%**, recall **26.58%**, duration MAE ≈ **3.28 s**.
- **Positive behavior:** the system correctly produces **no false `BED_EXIT`**.
- **What went wrong:** blankets, horizontal posture, and limited visible keypoints reduce pose confidence.
- **Why the current approach failed:** YOLO Pose is the main person/body evidence source, so heavy occlusion limits downstream activity classification.
- **Possible mitigation:** fuse pose with bed-region occupancy, person segmentation/object detection, temporal tracking, or an optional local vision model for occluded cases.

## Additional observed limitation — walking vs standing

In `test_video.mp4`, the final walking period is partly smoothed into `STANDING`. The simple motion heuristic is not always strong enough to separate slow walking from standing at a 2 FPS sampling rate.

Possible improvements include trajectory velocity, optical flow, person tracking, or a small temporal motion classifier.

## Bed-exit timing note

For `test_video.mp4`, the system correctly detects the `BED_EXIT` start at approximately **14.0 s**, but confirms it earlier than the manually selected clear movement-away point. The event-type precision and recall remain correct because event matching uses start time, while confirmation-time error is reported separately.

## Timeline continuity issue found during evaluation

Testing `turning_in_bed.mp4` exposed a short-segment merge bug that could create a gap in the generated timeline. The merge logic was fixed and a dedicated regression test, `test_short_segment_merging_does_not_create_timeline_gaps`, was added.

The final suite contains **32 passing tests**.
