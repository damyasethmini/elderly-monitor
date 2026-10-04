# Elderly Monitor — Agentic AI + Vision Assignment

A Python CLI system for analyzing an indoor video of an elderly person, recognizing activity over time, detecting bed exits/returns, calculating activity durations, and producing NORMAL/MONITOR/ALERT decisions.

## Current implementation

The project currently contains the perception, temporal state tracking, bed-event detection, agentic contextual analysis, contextual alert rules, and evaluation framework required by the assignment.

The main pipeline is:

```text
Video -> Frame Sampling -> YOLO Pose -> Pose Features
      -> Activity Classification -> Temporal Smoothing
      -> Bed Region / Occupancy -> Timeline
      -> Bed Exit / Return Detection
      -> LangGraph Agentic Context Analysis
      -> NORMAL / MONITOR / ALERT
```

### Step 5: agentic analysis

Step 5 uses a local **LangGraph `StateGraph`**. For a candidate bed event, the graph locates the event, inspects previous/following timeline context, conditionally expands the temporal window when evidence is ambiguous, checks spatial/movement evidence, and applies deterministic decision rules.

The Step 5 agent currently uses **no external LLM or paid API**. LangGraph is used for orchestration; the actual safety decision remains explicit and deterministic.

## Requirements

- Python 3.11 recommended
- Windows 10/11, macOS, or Linux
- A test MP4 in `data/videos/`
- Internet access only for installing dependencies and downloading the pretrained YOLO pose weights
- No paid model/API key is required

## Install

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Run the pipeline

### Step 2 — sample frames

```powershell
python -m src.main --sample-video data/videos/test_video.mp4 --sampling-fps 2
```

### Step 3A — YOLO pose

```powershell
python -m src.main --pose-video data/videos/test_video.mp4
```

### Step 3B — activity/state classification

```powershell
python -m src.main --state-video data/videos/test_video.mp4
```

### Step 4 — temporal timeline and bed events

```powershell
python -m src.main --events-video data/videos/test_video.mp4
```

### Step 5 — LangGraph agentic contextual analysis

```powershell
python -m src.agent.agentic_analysis --video data/videos/test_video.mp4
```

Output:

```text
results/test_video/analysis/agentic_analysis.json
```

The report records the agent framework, temporal context inspected, spatial/movement checks, decision reasons, and whether human review is recommended.

## Testing

Run all tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Run Step 5 tests only:

```powershell
python -m unittest tests.test_step5_agentic_analysis -v
```

## Project structure

```text
elderly-monitor/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── videos/
│   ├── ground_truth/
│   └── bed_regions/
├── src/
│   ├── main.py
│   ├── perception/
│   ├── state/
│   ├── events/
│   ├── agent/
│   ├── alerts/
│   └── evaluation/
├── evaluation/
├── results/
└── docs/
```

## Important evaluation note

The evaluation framework expects **manually reviewed ground truth**. The template files are not evaluation results. Final submission metrics must be calculated from labelled videos, and at least three real failure cases should be documented.

## Safety

This is an engineering assignment prototype. Alert thresholds are configurable examples and are not clinically validated medical thresholds. The system does not diagnose falls or medical emergencies.
