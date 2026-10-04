# Elderly Monitor — Agentic AI + Vision Assignment

A Python CLI prototype that analyzes a continuous indoor video of an elderly person and produces:

- activity/state recognition over time
- a transition-based activity timeline
- bed-exit and return-to-bed events
- time spent in each activity and bed occupancy state
- LangGraph-based agentic contextual analysis
- `NORMAL`, `MONITOR`, or `ALERT` decisions
- evaluation metrics against manually reviewed ground truth

The implementation is local-first and does **not** require a paid LLM API.

## Architecture

```text
Video
  ↓
Frame Sampling
  ↓
YOLO Pose (person + keypoints)
  ↓
Pose Feature Extraction
  ↓
Rule-based Activity Classification
  ↓
Temporal Smoothing + State Tracking
  ↓
Bed Region / Occupancy Reasoning
  ↓
Timeline + Duration Summary
  ↓
Bed Exit / Return Detection
  ↓
LangGraph Agentic Context Analysis
  ↓
Contextual Safety Rules
  ↓
NORMAL / MONITOR / ALERT
  ↓
Evaluation vs Ground Truth
```

See `docs/architecture.md` for the detailed diagram and design notes.

## Main technologies

- Python 3.11
- OpenCV
- Ultralytics YOLO11 Pose (`yolo11n-pose.pt`)
- NumPy / Pandas / scikit-learn
- LangGraph `StateGraph`
- deterministic temporal and safety rules

### LLM usage

The current Step 5 implementation uses **LangGraph for orchestration**, but it does **not** call GPT, Gemini, or another external LLM/VLM. The graph conditionally decides when to inspect more temporal context, while final event/safety decisions remain explicit and reproducible.

## Project structure

```text
elderly-monitor/
├── config.yaml
├── requirements.txt
├── README.md
├── RUN_ALL.ps1
├── data/
│   ├── videos/                 # downloaded locally; raw videos are ignored by Git
│   ├── bed_regions/
│   └── ground_truth/
├── src/
│   ├── main.py
│   ├── perception/
│   │   ├── frame_sampler.py
│   │   └── detector.py
│   ├── state/
│   │   ├── features.py
│   │   ├── classifier.py
│   │   ├── smoothing.py
│   │   └── state_machine.py
│   ├── events/
│   │   └── bed_events.py
│   ├── agent/
│   │   └── agentic_analysis.py
│   ├── alerts/
│   │   └── contextual_alert.py
│   └── evaluation/
│       └── evaluate.py
├── tests/
│   ├── test_step4_bed_events.py
│   ├── test_step4_timeline_continuity.py
│   ├── test_step5_agentic_analysis.py
│   ├── test_step6_contextual_alert.py
│   └── test_step7_evaluation.py
├── evaluation/
│   ├── test_video_ground_truth.json
│   ├── test_video_evaluation_report.json
│   ├── return_to_bed_ground_truth.json
│   ├── return_to_bed_evaluation_report.json
│   ├── ambiguous_sitting_ground_truth.json
│   ├── ambiguous_sitting_evaluation_report.json
│   ├── turning_in_bed_ground_truth.json
│   ├── turning_in_bed_evaluation_report.json
│   └── failure_cases.md
├── docs/
│   └── architecture.md
└── results/
```

## Installation

Create and activate a virtual environment:

```powershell
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pretrained YOLO weights are downloaded automatically by Ultralytics on first use if they are not already present locally.

## Evaluation videos

The raw video files are intentionally **not committed to Git** because they are large input assets. Download the public clips below and place them in:

```text
data/videos/
```

| Local filename | Scenario | Public source |
| --- | --- | --- |
| `test_video.mp4` | Bed exit / walking away | [Pexels — Person Getting Out of Bed](https://www.pexels.com/video/person-getting-out-of-bed-6918469/) 
| `return_to_bed.mp4` | Returning to bed | [Pexels — Woman Settling to Sleep in a Cozy Bedroom](https://www.pexels.com/video/woman-settling-to-sleep-in-a-cozy-bedroom-36604163/) |
| `ambiguous_sitting.mp4` | Sitting/stretching on the bed without leaving | [Pexels — Person Sitting on His Bed While Stretching His Arms](https://www.pexels.com/video/person-sitting-on-his-bed-while-stretching-his-arms-5983676/) |
| `turning_in_bed.mp4` | Turning/repositioning while lying in bed | [Mixkit — Man Lying Down Moving a Lot Because of Not Being Able to Sleep](https://mixkit.co/free-stock-video/man-lying-down-moving-a-lot-because-of-not-being-31414/) |

After downloading, rename the files to the filenames shown above.

## Run from Step 1

Run the commands from the repository root.

### Step 1 — check setup

```powershell
python -m src.main --check-setup
```

### Step 2 — sample video frames at 2 FPS

```powershell
python -m src.main --sample-video data/videos/test_video.mp4 --sampling-fps 2
```

### Step 3A — YOLO pose detection

```powershell
python -m src.main --pose-video data/videos/test_video.mp4
```

### Step 3B — activity/state classification

```powershell
python -m src.main --state-video data/videos/test_video.mp4
```

### Step 4 — temporal tracking and bed events

```powershell
python -m src.main --events-video data/videos/test_video.mp4
```

Important Step 4 safeguards include:

- temporal smoothing rather than independent frame decisions
- movement measured across the whole out-of-bed episode
- spatial outside-bed evidence for exit confirmation
- return-to-bed confirmation using segment start/end bed-region evidence
- `UNKNOWN` breaking confirmed continuous episodes
- timeline-continuity protection so short-segment merging does not create gaps

### Step 5 — LangGraph agentic analysis

```powershell
python -m src.agent.agentic_analysis --video data/videos/test_video.mp4
```

Expected framework line:

```text
Agent framework: LangGraph StateGraph
External LLM API: False
```

### Step 6 — contextual alert decision

```powershell
python -m src.alerts.contextual_alert --video data/videos/test_video.mp4
```

### Step 7 — evaluation

```powershell
python -m src.evaluation.evaluate --video data/videos/test_video.mp4 --ground-truth evaluation/test_video_ground_truth.json --output evaluation/test_video_evaluation_report.json
```

### Run all unit tests

```powershell
python -m unittest discover -s tests -v
```

The current suite contains **32 tests** across Steps 4–7.

You can also run the full sequence using:

```powershell
.\RUN_ALL.ps1
```

## Evaluation summary

The final evaluation uses multiple scenarios rather than only one clip.

| Video | Main scenario | Activity accuracy | Bed occupancy accuracy | Duration MAE |
| --- | --- | ---: | ---: | ---: |
| `test_video.mp4` | Bed exit / walking away | 72.83% | 84.78% | 1.42 s |
| `return_to_bed.mp4` | Return to bed / blanket occlusion | 40.95% | 8.19% | 2.29 s |
| `ambiguous_sitting.mp4` | Sitting/stretching without leaving | 46.50% | 46.50% | 3.30 s |
| `turning_in_bed.mp4` | Turning under blanket / low pose visibility | 26.58% | 26.58% | 3.28 s |

### Bed-event results

For `test_video.mp4`:

- `BED_EXIT` precision: **100%**
- `BED_EXIT` recall: **100%**
- BED_EXIT start-time error: **0.0 s**
- BED_EXIT confirmation-time absolute error: **4.5 s**
- `RETURN_TO_BED`: **N/A** because the clip contains no return event

For `return_to_bed.mp4`:

- ground truth contains one `RETURN_TO_BED`
- the system predicted no return event
- `RETURN_TO_BED` recall: **0%**
- the main failure is loss of reliable pose evidence when the person becomes partially occluded by the bed/blanket

For `ambiguous_sitting.mp4`:

- no bed exit is present in ground truth
- the system correctly produced **no false `BED_EXIT`**
- `SITTING_ON_BED` precision: **100%**
- `SITTING_ON_BED` recall: **46.5%**

For `turning_in_bed.mp4`:

- no bed exit is present in ground truth
- the system correctly produced **no false `BED_EXIT`**
- `LYING_IN_BED` precision: **100%**
- `LYING_IN_BED` recall: **26.58%**

For videos where neither ground truth nor prediction contains a given event, event precision/recall should be interpreted as **N/A**, even if the raw evaluation JSON stores `0.0`.

## Difficult cases and failure analysis

### 1. Return-to-bed under blanket/bed occlusion

In `return_to_bed.mp4`, the person approaches and gets back into bed, but YOLO Pose loses reliable body keypoints once the person becomes partially hidden by the bed/blanket. Much of the sequence becomes `UNKNOWN`, and the `RETURN_TO_BED` event is missed.

### 2. Stretching while seated on the bed

In `ambiguous_sitting.mp4`, the person remains seated on the bed while stretching. The event layer correctly avoids a false bed exit, but unusual torso/arm geometry causes many frames to become `UNKNOWN`, reducing `SITTING_ON_BED` recall.

### 3. Turning under blankets / low pose visibility

In `turning_in_bed.mp4`, the person remains lying in bed while repeatedly changing position. No false bed exit is generated, but pose detection is unreliable for much of the clip. A timeline-continuity regression found during this test was fixed, and a dedicated regression test was added.

Additional failure details are documented in `evaluation/failure_cases.md`.

## Important outputs

Typical compact outputs committed to the repository include:

```text
results/<video_name>/state/state_manifest.json
results/<video_name>/timeline/timeline.json
results/<video_name>/timeline/bed_events.json
results/<video_name>/analysis/agentic_analysis.json
results/<video_name>/alerts/contextual_alert.json

evaluation/<video_name>_ground_truth.json
evaluation/<video_name>_evaluation_report.json
evaluation/failure_cases.md
```

Large generated frame folders, annotated pose images, per-frame pose observations, model weights, and raw videos are excluded through `.gitignore`.

## Alert interpretation

- `NORMAL`: no relevant safety condition is detected.
- `MONITOR`: a confirmed event or uncertainty should continue to be monitored.
- `ALERT`: a configured high-severity condition such as prolonged confirmed absence is reached.

The thresholds are assignment/prototype rules, not clinically validated medical thresholds.

## Evaluation notes

- Ground truth must be manually reviewed and must not be copied from model predictions.
- Activity accuracy is duration-weighted.
- Bed events are matched using event **start time** within the configured tolerance.
- Confirmation-time error is reported separately so a correctly detected event is not incorrectly counted as both a false positive and false negative merely because confirmation timing differs.
- Activity-duration error is reported as predicted duration minus ground-truth duration.
- At least three real failure cases are documented in `evaluation/failure_cases.md`.
- `UNKNOWN` is deliberately used when the vision evidence is insufficient rather than forcing an unreliable state.

## Known limitations

- Activity recognition is a rule-based pose baseline and can confuse similar postures.
- Walking can be confused with standing when motion evidence is weak.
- Pose estimation can fail when the person is hidden by blankets, bedding, or poor visibility.
- Bed-region estimation is heuristic and could be improved with a manually labelled region or a dedicated bed/object detector.
- The current pipeline primarily follows one main person and is not designed for robust caregiver/multi-person identity tracking.
- The current agent is LangGraph-orchestrated but does not use an external LLM/VLM.
- A future local VLM fallback could be used only for ambiguous observations such as heavy occlusion or determining whether a horizontal person is on the bed or elsewhere.

## What I would improve with more time

- combine pose features with a dedicated bed/person detector
- use stronger temporal motion features for `WALKING` versus `STANDING`
- add more return-to-bed and multi-person/caregiver evaluation clips
- add manually defined bed regions for more reliable spatial evaluation
- optionally use a local VLM only for ambiguous/occluded observations
- aggregate metrics across a larger labelled test set

## Safety disclaimer

This repository is an engineering assignment prototype, not a medical device. It does not diagnose falls, illness, or emergencies.