# Architecture

## Elderly Monitor — Agentic AI + Vision

```text
                       Indoor Video
                            |
                            v
                    Frame Sampling
                            |
                            v
                       YOLO Pose
                 (person + keypoints)
                            |
                            v
                    Pose Features
                            |
                            v
                 Activity Classifier
                            |
                            v
               Temporal Smoothing / State
                            |
                            v
             Bed Region + Bed Occupancy
                            |
                            v
              Timeline + Activity Durations
                            |
                            v
              Bed Exit / Return Detector
                            |
                            v
            +-------------------------------+
            | Step 5: LangGraph Agent       |
            |                               |
            | Locate event                  |
            |       |                       |
            | Inspect previous/following     |
            |       |                       |
            | Ambiguous? ---- yes ----------+
            |       | no                    |
            |       v                       v
            | Expand context       Check spatial/movement
            |       |                       |
            |       +-----------+-----------+
            |                   v
            |              Event decision
            +-------------------+-----------+
                                |
                                v
                 Contextual Alert Rules
                                |
                   +------------+------------+
                   |            |            |
                   v            v            v
                NORMAL       MONITOR       ALERT
```

## Step 5 agent behavior

The Step 5 agent is a **LangGraph `StateGraph`** with conditional routing. It operates on structured evidence produced by the vision/temporal pipeline.

For a candidate bed event, it:

1. Locates the event in the timeline.
2. Inspects the previous and following segments.
3. Decides whether the initial context is ambiguous.
4. If ambiguous, expands the temporal window.
5. Checks bed-region and movement evidence.
6. Applies explicit event/duration rules.
7. Produces NORMAL, MONITOR, or ALERT plus the inspected evidence.

The agent currently does **not** call an external LLM or VLM. This keeps the project local, reproducible, and free of API costs while still demonstrating conditional agentic orchestration.

## Design principle

The vision/state layers provide evidence; the agent orchestrates context inspection; safety rules make the final decision. The system does not treat a generated confidence value as measured real-world accuracy.
