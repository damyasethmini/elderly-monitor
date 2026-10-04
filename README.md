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
│   ├── videos/
│   │   └── test_video.mp4
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
│   ├── test_step5_agentic_analysis.py
│   ├── test_step6_contextual_alert.py
│   └── test_step7_evaluation.py
├── evaluation/
│   ├── test_video_ground_truth.json
│   ├── test_video_evaluation_report.json
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

The current suite contains **31 tests** across Steps 4–7.

You can also run the full sequence using:

```powershell
.\RUN_ALL.ps1
```

## Current evaluated example

For the included `test_video.mp4` and manually reviewed ground truth:

| Metric | Result |
| --- | ---: |
| Activity accuracy | 72.83% |
| Bed occupancy accuracy | 84.78% |
| BED_EXIT precision | 100% |
| BED_EXIT recall | 100% |
| Duration MAE | 1.42 s |
| BED_EXIT start-time error | 0.0 s |
| BED_EXIT confirmation-time absolute error | 4.5 s |

`RETURN_TO_BED` is **N/A for this clip** because neither the ground truth nor the prediction contains a return event.

The example demonstrates a working end-to-end pipeline, but the assignment should be evaluated on additional scenarios before final submission. In particular, add videos containing return-to-bed behavior and difficult cases such as occlusion, sitting on a chair, blankets, poor lighting, and temporary camera-view loss.

## Important outputs

```text
results/test_video/frame_manifest.json
results/test_video/pose/pose_manifest.json
results/test_video/state/state_manifest.json
results/test_video/timeline/timeline.json
results/test_video/timeline/bed_events.json
results/test_video/analysis/agentic_analysis.json
results/test_video/alerts/contextual_alert.json
evaluation/test_video_evaluation_report.json
```

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
- At least three real failure cases are documented in `evaluation/failure_cases.md`.

## Known limitations

- Activity recognition is a rule-based pose baseline and still confuses similar postures.
- Walking can be confused with standing when motion evidence is weak.
- Bed-region estimation is heuristic and can be improved with a manually labelled region or object detector.
- The current included video does not contain a return-to-bed example.
- The current agent is LangGraph-orchestrated but does not use an LLM/VLM. A local VLM could later be used only for ambiguous observations if desired.

## Safety disclaimer

This repository is an engineering assignment prototype, not a medical device. It does not diagnose falls, illness, or emergencies.
