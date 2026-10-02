# Elderly Monitor — Agentic AI + Vision Assignment

A Python project scaffold for analyzing an indoor video, estimating an elderly person's activity over time, detecting bed exits/returns, calculating durations, and producing monitoring decisions.

> **Current stage: Step 2 (video frame sampling).** The CLI checks your environment, inspects video metadata, and samples still frames at a configurable rate. Person/activity recognition, bed-event logic, agent workflow, and evaluation will be implemented in later steps.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python **3.11** recommended
- Internet connection for installing Python packages and downloading pretrained model weights later
- A short MP4 test video for the video-input check
- No paid API key is required for this setup step

Ultralytics/PyTorch packages can take a while to install and require substantial disk space. CPU inference can be used for development; an NVIDIA GPU is optional.

## 2. Open the project in Windows

1. Download and extract the ZIP file.
2. Open the extracted `elderly-monitor-starter` folder in VS Code.
3. In VS Code, choose **Terminal → New Terminal**.
4. Run the following commands in PowerShell from the project root.

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can use this command for the current terminal session, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is not recognized, install Python 3.11 from <https://www.python.org/downloads/> and enable the Python launcher during installation.

## 3. Select the VS Code interpreter

Press `Ctrl+Shift+P` → search for **Python: Select Interpreter** → select the interpreter inside this project's `.venv` folder.

## 4. Check the setup

Run this from the project root while `.venv` is activated:

```powershell
python -m src.main --check-setup
```

The program should report the Python version, configuration status, expected folders, and installed dependencies. If some dependencies are missing, run:

```powershell
python -m pip install -r requirements.txt
```

## 5. Add a test video and check it

Copy a short test video into `data/videos/`, for example:

```text
data/videos/test_video.mp4
```

Then run:

```powershell
python -m src.main --video data/videos/test_video.mp4
```

This command only checks that OpenCV can open the file and reports its resolution, source FPS, frame count, and approximate duration. It does not yet classify activities.

## 6. Sample frames from the video (Step 2)

Run this command from the project root:

```powershell
python -m src.main --sample-video data/videos/test_video.mp4
```

The default sampling rate comes from `video.sampling_fps` in `config.yaml` (initially one frame per second). To sample two frames per second, run:

```powershell
python -m src.main --sample-video data/videos/test_video.mp4 --sampling-fps 2
```

For a quick test, save no more than 10 frames:

```powershell
python -m src.main --sample-video data/videos/test_video.mp4 --max-frames 10
```

The output will be saved under `results/test_video/` (using the input filename without its extension):

```text
results/
└── test_video/
    ├── frame_manifest.json
    └── frames/
        ├── frame_000001_t00000.000s.jpg
        ├── frame_000002_t00001.000s.jpg
        └── ...
```

Open the `frames` folder to visually inspect the sampled JPEG images. `frame_manifest.json` contains the source video metadata and each sampled frame's frame index and timestamp. No activity labels are predicted yet.

If a video is about 23 seconds long and sampling is set to 1 FPS, expect roughly 23 sampled frames. The exact count depends on the video's duration and frame timestamps.


## 7. Configuration

Edit `config.yaml` to adjust sampling FPS, model name, confidence thresholds, bed-event confirmation times, bed polygon, and example monitoring thresholds. The defaults are initial placeholders and must be tested against labelled videos.

For a manually defined bed region, the polygon will use normalized `(x, y)` coordinates between `0.0` and `1.0`, relative to the image width and height. Leave `polygon: []` until the bed-region implementation is added.

**Safety note:** The alert thresholds are engineering examples for this assignment, not clinically validated medical thresholds. Do not use this prototype as a real emergency or patient-monitoring system.

## 8. Project structure

```text
elderly-monitor-starter/
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
│   ├── outputs/
│   └── utils/
├── evaluation/
├── results/
├── docs/
└── notebooks/
```

## 9. Next implementation steps

1. Run person detection and pose estimation on the sampled frames.
3. Define bed occupancy separately from activity state.
4. Build temporal smoothing and state-transition logic.
5. Detect bed exit/return events from sequences, not single frames.
6. Add the agent to inspect earlier/later context for ambiguous observations.
7. Generate summaries, timelines, alert decisions, and evaluation results.

## 10. Privacy and data handling

Use only videos you have permission to process. Prefer synthetic, public, or consented test data. Keep private videos and generated model weights out of source control; the provided `.gitignore` excludes them by default.
