# Ground-truth annotation guide

Create `ground_truth.json` by manually reviewing the source video.

Use mutually exclusive primary activity states:
- LYING_IN_BED
- SITTING_ON_BED
- SITTING_OUTSIDE_BED
- STANDING
- WALKING
- UNKNOWN

Use `bed_occupancy` separately:
- IN_BED
- OUT_OF_BED
- UNKNOWN

Use `UNKNOWN` when evidence is insufficient. Do not copy predicted labels into ground truth.

Bed events are timestamped events:
- BED_EXIT
- RETURN_TO_BED

The evaluator matches predicted and ground-truth events using a configurable time tolerance (default 2 seconds).
