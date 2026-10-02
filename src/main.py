# """Command-line entry point for the elderly-monitor assignment.

# The CLI currently supports:
#     - environment/setup checks
#     - video metadata inspection
#     - frame sampling
#     - YOLO Pose estimation
#     - activity and bed-occupancy classification

# Later stages will add:
#     - temporal state tracking
#     - bed exit/return events
#     - agentic reasoning
#     - safety alerts
#     - evaluation and reporting
# """

# from __future__ import annotations

# import argparse
# import importlib.util
# import json
# import sys
# from pathlib import Path
# from typing import Any

# import cv2
# import yaml

# from src.perception.frame_sampler import sample_video, write_manifest
# from src.perception.detector import run_pose_pipeline
# from src.state.classifier import classify_pose_manifest


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# DEPENDENCIES = {
#     "cv2": "opencv-python",
#     "PIL": "Pillow",
#     "numpy": "numpy",
#     "pandas": "pandas",
#     "sklearn": "scikit-learn",
#     "matplotlib": "matplotlib",
#     "yaml": "PyYAML",
#     "dotenv": "python-dotenv",
#     "ultralytics": "ultralytics",
#     "langgraph": "langgraph",
#     "pytest": "pytest",
# }


# def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
#     """Load YAML configuration and validate that it contains a mapping."""
#     if not path.is_file():
#         raise FileNotFoundError(f"Config file not found: {path}")

#     with path.open("r", encoding="utf-8") as file:
#         config = yaml.safe_load(file)

#     if not isinstance(config, dict):
#         raise ValueError(
#             "config.yaml must contain a YAML mapping at the top level."
#         )

#     return config


# def check_dependencies() -> tuple[list[str], list[str]]:
#     """Return installed and missing dependency names."""
#     installed: list[str] = []
#     missing: list[str] = []

#     for import_name, package_name in DEPENDENCIES.items():
#         try:
#             found = importlib.util.find_spec(import_name) is not None
#         except (ImportError, ValueError):
#             found = False

#         if found:
#             installed.append(package_name)
#         else:
#             missing.append(package_name)

#     return installed, missing


# def check_setup() -> int:
#     """Check Python, dependencies, config, and expected project folders."""
#     print("Elderly Monitor — setup check")
#     print("=" * 32)

#     print(f"Python: {sys.version.split()[0]}")
#     print(f"Project root: {PROJECT_ROOT}")

#     if sys.version_info < (3, 10):
#         print(
#             "WARNING: Python 3.10+ is recommended; "
#             "use Python 3.11 for this project."
#         )

#     # ---------------------------------------------------------
#     # Config check
#     # ---------------------------------------------------------
#     try:
#         config = load_config()

#         print(f"Config: OK ({CONFIG_PATH.name})")

#         video_config = config.get("video", {})
#         perception_config = config.get("perception", {})

#         print(
#             "Video sampling FPS: "
#             f"{video_config.get('sampling_fps', 'not set')}"
#         )

#         print(
#             "Pose model: "
#             f"{perception_config.get('pose_model', 'not set')}"
#         )

#     except (
#         FileNotFoundError,
#         ValueError,
#         yaml.YAMLError,
#     ) as exc:
#         print(f"Config: ERROR — {exc}")
#         return 1

#     # ---------------------------------------------------------
#     # Folder check
#     # ---------------------------------------------------------
#     expected_dirs = [
#         PROJECT_ROOT / "data" / "videos",
#         PROJECT_ROOT / "data" / "ground_truth",
#         PROJECT_ROOT / "data" / "bed_regions",
#         PROJECT_ROOT / "results",
#     ]

#     missing_dirs = [
#         str(folder.relative_to(PROJECT_ROOT))
#         for folder in expected_dirs
#         if not folder.is_dir()
#     ]

#     if missing_dirs:
#         print("Folders missing:")

#         for folder in missing_dirs:
#             print(f"  - {folder}")

#         return 1

#     print("Project folders: OK")

#     # ---------------------------------------------------------
#     # Dependency check
#     # ---------------------------------------------------------
#     installed, missing = check_dependencies()

#     print(
#         f"Dependencies installed: "
#         f"{len(installed)}/{len(DEPENDENCIES)}"
#     )

#     if installed:
#         print("  OK: " + ", ".join(installed))

#     if missing:
#         print("  Missing: " + ", ".join(missing))

#         print(
#             "Install them with: "
#             "python -m pip install -r requirements.txt"
#         )

#         return 1

#     print("\nSetup looks ready.")

#     print(
#         "Available stages:"
#     )
#     print(
#         "  Step 2: --sample-video <path>"
#     )
#     print(
#         "  Step 3A: --pose-video <path>"
#     )
#     print(
#         "  Step 3B: --state-video <path>"
#     )

#     return 0


# def inspect_video(video_path: Path) -> int:
#     """Check that OpenCV can open the requested video and print metadata."""

#     if not video_path.is_absolute():
#         video_path = (Path.cwd() / video_path).resolve()

#     if not video_path.is_file():
#         print(
#             f"ERROR: Video file not found: {video_path}",
#             file=sys.stderr,
#         )
#         return 2

#     capture = cv2.VideoCapture(str(video_path))

#     try:
#         if not capture.isOpened():
#             print(
#                 f"ERROR: OpenCV could not open this video: {video_path}",
#                 file=sys.stderr,
#             )
#             return 2

#         fps = float(
#             capture.get(cv2.CAP_PROP_FPS) or 0.0
#         )

#         frame_count = int(
#             capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
#         )

#         width = int(
#             capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
#         )

#         height = int(
#             capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
#         )

#         duration_sec = (
#             frame_count / fps
#             if fps > 0
#             else 0.0
#         )

#         details = {
#             "video": str(video_path),
#             "width": width,
#             "height": height,
#             "source_fps": round(fps, 3),
#             "frame_count": frame_count,
#             "duration_sec": round(duration_sec, 3),
#         }

#         print(json.dumps(details, indent=2))

#         print(
#             "\nVideo input check passed."
#         )

#         return 0

#     finally:
#         capture.release()


# def sample_video_command(
#     video_path: Path,
#     sampling_fps: float | None = None,
#     max_frames: int | None = None,
# ) -> int:
#     """Sample a video and write JPEG frames plus a JSON manifest."""

#     if not video_path.is_absolute():
#         video_path = (Path.cwd() / video_path).resolve()

#     try:
#         config = load_config()

#         video_config = config.get(
#             "video",
#             {}
#         )

#         output_config = config.get(
#             "output",
#             {}
#         )

#         # -----------------------------------------------------
#         # Sampling configuration
#         # -----------------------------------------------------
#         rate = float(
#             sampling_fps
#             if sampling_fps is not None
#             else video_config.get(
#                 "sampling_fps",
#                 1.0,
#             )
#         )

#         frame_limit = (
#             max_frames
#             if max_frames is not None
#             else video_config.get(
#                 "max_frames"
#             )
#         )

#         if frame_limit is not None:
#             frame_limit = int(frame_limit)

#         # -----------------------------------------------------
#         # Output locations
#         # -----------------------------------------------------
#         output_root_value = output_config.get(
#             "directory",
#             "results",
#         )

#         output_root = Path(
#             output_root_value
#         )

#         if not output_root.is_absolute():
#             output_root = PROJECT_ROOT / output_root

#         run_dir = (
#             output_root / video_path.stem
#         )

#         frames_dir = (
#             run_dir / "frames"
#         )

#         manifest_path = (
#             run_dir / "frame_manifest.json"
#         )

#         # -----------------------------------------------------
#         # Run frame sampler
#         # -----------------------------------------------------
#         manifest = sample_video(
#             video_path=video_path,
#             frames_dir=frames_dir,
#             sampling_fps=rate,
#             max_frames=frame_limit,
#         )

#         # Make manifest paths portable.
#         manifest["frames_directory"] = "frames"

#         write_manifest(
#             manifest,
#             manifest_path,
#         )

#         # -----------------------------------------------------
#         # Print result
#         # -----------------------------------------------------
#         print(
#             "Frame sampling completed successfully."
#         )

#         print(
#             f"Video: {video_path}"
#         )

#         print(
#             f"Duration: "
#             f"{manifest['duration_sec']:.2f} seconds"
#         )

#         print(
#             f"Source FPS: "
#             f"{manifest['source_fps']:.3f}"
#         )

#         print(
#             f"Sampling FPS: "
#             f"{manifest['sampling_fps_effective']:.3f}"
#         )

#         print(
#             f"Frames saved: "
#             f"{manifest['sampled_frame_count']}"
#         )

#         print(
#             f"Frames folder: "
#             f"{frames_dir}"
#         )

#         print(
#             f"Manifest: "
#             f"{manifest_path}"
#         )

#         return 0

#     except (
#         FileNotFoundError,
#         ValueError,
#         OSError,
#         yaml.YAMLError,
#     ) as exc:
#         print(
#             f"ERROR: {exc}",
#             file=sys.stderr,
#         )
#         return 2


# def pose_video_command(
#     video_path: Path,
# ) -> int:
#     """Run YOLO Pose estimation on previously sampled video frames."""

#     if not video_path.is_absolute():
#         video_path = (Path.cwd() / video_path).resolve()

#     if not video_path.is_file():
#         print(
#             f"ERROR: Video file not found: {video_path}",
#             file=sys.stderr,
#         )
#         return 2

#     try:
#         # -----------------------------------------------------
#         # Load config
#         # -----------------------------------------------------
#         config = load_config()

#         perception_config = config.get(
#             "perception",
#             {}
#         )

#         output_config = config.get(
#             "output",
#             {}
#         )

#         # -----------------------------------------------------
#         # Pose configuration
#         # -----------------------------------------------------
#         model_name = perception_config.get(
#             "pose_model",
#             "yolo11n-pose.pt",
#         )

#         confidence = float(
#             perception_config.get(
#                 "pose_confidence",
#                 0.35,
#             )
#         )

#         image_size = int(
#             perception_config.get(
#                 "pose_image_size",
#                 640,
#             )
#         )

#         device = perception_config.get(
#             "pose_device",
#             "cpu",
#         )

#         # -----------------------------------------------------
#         # Output directory
#         # -----------------------------------------------------
#         output_root_value = output_config.get(
#             "directory",
#             "results",
#         )

#         output_root = Path(
#             output_root_value
#         )

#         if not output_root.is_absolute():
#             output_root = (
#                 PROJECT_ROOT / output_root
#             )

#         run_dir = (
#             output_root / video_path.stem
#         )

#         manifest_path = (
#             run_dir / "frame_manifest.json"
#         )

#         # -----------------------------------------------------
#         # Step 2 dependency check
#         # -----------------------------------------------------
#         if not manifest_path.is_file():
#             print(
#                 "ERROR: frame_manifest.json was not found.",
#                 file=sys.stderr,
#             )

#             print(
#                 f"Expected: {manifest_path}",
#                 file=sys.stderr,
#             )

#             print(
#                 "\nRun Step 2 first with:",
#                 file=sys.stderr,
#             )

#             print(
#                 "  python -m src.main "
#                 f"--sample-video {video_path}",
#                 file=sys.stderr,
#             )

#             return 2

#         # -----------------------------------------------------
#         # Run YOLO Pose
#         # -----------------------------------------------------
#         pose_output_dir = (
#             run_dir / "pose"
#         )

#         print(
#             "\nStarting YOLO Pose estimation..."
#         )

#         print(
#             f"Video: {video_path}"
#         )

#         print(
#             f"Model: {model_name}"
#         )

#         print(
#             f"Confidence threshold: {confidence}"
#         )

#         print(
#             f"Image size: {image_size}"
#         )

#         print(
#             f"Device: {device}"
#         )

#         print()

#         result = run_pose_pipeline(
#             manifest_path=manifest_path,
#             output_dir=pose_output_dir,
#             model_name=model_name,
#             confidence=confidence,
#             image_size=image_size,
#             device=device,
#         )

#         # -----------------------------------------------------
#         # Print results
#         # -----------------------------------------------------
#         print(
#             "\nYOLO pose estimation completed successfully."
#         )

#         print(
#             f"Frames processed: "
#             f"{result['frame_count']}"
#         )

#         print(
#             "Frames with person detected: "
#             f"{result['person_detected_count']}"
#         )

#         print(
#             f"Pose output directory: "
#             f"{pose_output_dir}"
#         )

#         print(
#             "Combined pose manifest: "
#             f"{pose_output_dir / 'pose_manifest.json'}"
#         )

#         print(
#             "Annotated frames: "
#             f"{pose_output_dir / 'annotated'}"
#         )

#         print(
#             "Per-frame JSON observations: "
#             f"{pose_output_dir / 'observations'}"
#         )

#         return 0

#     except (
#         FileNotFoundError,
#         ValueError,
#         OSError,
#     ) as exc:
#         print(
#             f"ERROR: {exc}",
#             file=sys.stderr,
#         )
#         return 2


# def state_video_command(
#     video_path: Path,
# ) -> int:
#     """Classify activity and bed occupancy from YOLO Pose results."""

#     if not video_path.is_absolute():
#         video_path = (Path.cwd() / video_path).resolve()

#     if not video_path.is_file():
#         print(
#             f"ERROR: Video file not found: {video_path}",
#             file=sys.stderr,
#         )
#         return 2

#     try:
#         # -----------------------------------------------------
#         # Load configuration
#         # -----------------------------------------------------
#         config = load_config()

#         state_config = config.get(
#             "state",
#             {}
#         )

#         output_config = config.get(
#             "output",
#             {}
#         )

#         # -----------------------------------------------------
#         # Output location
#         # -----------------------------------------------------
#         output_root_value = output_config.get(
#             "directory",
#             "results",
#         )

#         output_root = Path(
#             output_root_value
#         )

#         if not output_root.is_absolute():
#             output_root = (
#                 PROJECT_ROOT / output_root
#             )

#         run_dir = (
#             output_root / video_path.stem
#         )

#         pose_manifest_path = (
#             run_dir
#             / "pose"
#             / "pose_manifest.json"
#         )

#         # -----------------------------------------------------
#         # Step 3A dependency check
#         # -----------------------------------------------------
#         if not pose_manifest_path.is_file():
#             print(
#                 "ERROR: pose_manifest.json was not found.",
#                 file=sys.stderr,
#             )

#             print(
#                 f"Expected: {pose_manifest_path}",
#                 file=sys.stderr,
#             )

#             print(
#                 "\nRun Step 3A first with:",
#                 file=sys.stderr,
#             )

#             print(
#                 "  python -m src.main "
#                 f"--pose-video {video_path}",
#                 file=sys.stderr,
#             )

#             return 2

#         # -----------------------------------------------------
#         # Run classifier
#         # -----------------------------------------------------
#         state_output_dir = (
#             run_dir / "state"
#         )

#         print(
#             "\nStarting activity/state classification..."
#         )

#         print(
#             f"Video: {video_path}"
#         )

#         print(
#             f"Pose manifest: "
#             f"{pose_manifest_path}"
#         )

#         print()

#         result = classify_pose_manifest(
#             pose_manifest_path=pose_manifest_path,
#             output_dir=state_output_dir,
#             thresholds=state_config,
#         )

#         # -----------------------------------------------------
#         # Count activities and occupancy
#         # -----------------------------------------------------
#         activity_counts: dict[str, int] = {}

#         occupancy_counts: dict[str, int] = {}

#         for observation in result["observations"]:

#             activity = observation[
#                 "activity"
#             ]

#             occupancy = observation[
#                 "bed_occupancy"
#             ]

#             activity_counts[activity] = (
#                 activity_counts.get(
#                     activity,
#                     0,
#                 )
#                 + 1
#             )

#             occupancy_counts[occupancy] = (
#                 occupancy_counts.get(
#                     occupancy,
#                     0,
#                 )
#                 + 1
#             )

#         # -----------------------------------------------------
#         # Print result
#         # -----------------------------------------------------
#         print(
#             "\nState classification completed successfully."
#         )

#         print(
#             f"Frames processed: "
#             f"{result['frame_count']}"
#         )

#         print(
#             "\nActivity counts:"
#         )

#         for activity, count in activity_counts.items():
#             print(
#                 f"  {activity}: {count}"
#             )

#         print(
#             "\nBed occupancy counts:"
#         )

#         for occupancy, count in occupancy_counts.items():
#             print(
#                 f"  {occupancy}: {count}"
#             )

#         print(
#             "\nState manifest: "
#             f"{state_output_dir / 'state_manifest.json'}"
#         )

#         return 0

#     except (
#         FileNotFoundError,
#         ValueError,
#         OSError,
#         yaml.YAMLError,
#     ) as exc:
#         print(
#             f"ERROR: {exc}",
#             file=sys.stderr,
#         )
#         return 2


# def build_parser() -> argparse.ArgumentParser:
#     """Build the command-line argument parser."""

#     parser = argparse.ArgumentParser(
#         description=(
#             "Elderly Monitor: setup checks, video processing, "
#             "frame sampling, YOLO Pose, and state recognition."
#         )
#     )

#     group = parser.add_mutually_exclusive_group(
#         required=True
#     )

#     # ---------------------------------------------------------
#     # Step 1
#     # ---------------------------------------------------------
#     group.add_argument(
#         "--check-setup",
#         action="store_true",
#         help=(
#             "Check the Python environment, dependencies, "
#             "config, and folders."
#         ),
#     )

#     # ---------------------------------------------------------
#     # Video inspection
#     # ---------------------------------------------------------
#     group.add_argument(
#         "--video",
#         type=Path,
#         help=(
#             "Check that a video can be opened and inspect "
#             "its basic metadata."
#         ),
#     )

#     # ---------------------------------------------------------
#     # Step 2
#     # ---------------------------------------------------------
#     group.add_argument(
#         "--sample-video",
#         type=Path,
#         help=(
#             "Sample frames from a video and save a JSON "
#             "timestamp manifest."
#         ),
#     )

#     # ---------------------------------------------------------
#     # Step 3A
#     # ---------------------------------------------------------
#     group.add_argument(
#         "--pose-video",
#         type=Path,
#         help=(
#             "Run YOLO Pose estimation on previously "
#             "sampled frames."
#         ),
#     )

#     # ---------------------------------------------------------
#     # Step 3B
#     # ---------------------------------------------------------
#     group.add_argument(
#         "--state-video",
#         type=Path,
#         help=(
#             "Classify activity and bed occupancy "
#             "from YOLO Pose results."
#         ),
#     )

#     # ---------------------------------------------------------
#     # Optional Step 2 arguments
#     # ---------------------------------------------------------
#     parser.add_argument(
#         "--sampling-fps",
#         type=float,
#         help=(
#             "Override video.sampling_fps from config.yaml "
#             "(for example, 1 or 2)."
#         ),
#     )

#     parser.add_argument(
#         "--max-frames",
#         type=int,
#         help=(
#             "Optional maximum number of frames to save "
#             "for a quick test."
#         ),
#     )

#     return parser


# def main() -> int:
#     """CLI entry point."""

#     args = build_parser().parse_args()

#     # ---------------------------------------------------------
#     # Step 1
#     # ---------------------------------------------------------
#     if args.check_setup:
#         return check_setup()

#     # ---------------------------------------------------------
#     # Video inspection
#     # ---------------------------------------------------------
#     if args.video is not None:
#         return inspect_video(
#             args.video
#         )

#     # ---------------------------------------------------------
#     # Step 2
#     # ---------------------------------------------------------
#     if args.sample_video is not None:
#         return sample_video_command(
#             args.sample_video,
#             sampling_fps=args.sampling_fps,
#             max_frames=args.max_frames,
#         )

#     # ---------------------------------------------------------
#     # Step 3A
#     # ---------------------------------------------------------
#     if args.pose_video is not None:
#         return pose_video_command(
#             args.pose_video
#         )

#     # ---------------------------------------------------------
#     # Step 3B
#     # ---------------------------------------------------------
#     if args.state_video is not None:
#         return state_video_command(
#             args.state_video
#         )

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


"""Command-line entry point for the elderly-monitor assignment.

Supported stages:

    Step 1:
        --check-setup

    Video inspection:
        --video <path>

    Step 2:
        --sample-video <path>

    Step 3A:
        --pose-video <path>

    Step 3B:
        --state-video <path>

    Step 4:
        --events-video <path>

Step 4 adds:
    - temporal smoothing
    - stable state segments
    - activity timeline
    - bed exit detection
    - bed return detection

Later stages can add:
    - agentic reasoning
    - NORMAL / MONITOR / ALERT
    - evaluation
    - reporting
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.perception.frame_sampler import (
    sample_video,
    write_manifest,
)

from src.perception.detector import (
    run_pose_pipeline,
)

from src.state.classifier import (
    classify_pose_manifest,
)

from src.state.state_machine import (
    build_timeline,
)

from src.events.bed_events import (
    detect_bed_events,
    save_events,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT / "config.yaml"
)


# ============================================================
# Dependency list
# ============================================================

DEPENDENCIES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "ultralytics": "ultralytics",
    "langgraph": "langgraph",
    "pytest": "pytest",
}


# ============================================================
# Configuration
# ============================================================

def load_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml must contain a YAML "
            "mapping at the top level."
        )

    return config


# ============================================================
# Setup checks
# ============================================================

def check_dependencies() -> tuple[
    list[str],
    list[str],
]:
    """Return installed and missing dependencies."""

    installed: list[str] = []
    missing: list[str] = []

    for import_name, package_name in (
        DEPENDENCIES.items()
    ):
        try:
            found = (
                importlib.util.find_spec(
                    import_name
                )
                is not None
            )
        except (
            ImportError,
            ValueError,
        ):
            found = False

        if found:
            installed.append(
                package_name
            )
        else:
            missing.append(
                package_name
            )

    return installed, missing


def check_setup() -> int:
    """Check Python, dependencies, config,
    and required folders.
    """

    print(
        "Elderly Monitor — setup check"
    )

    print(
        "=" * 32
    )

    print(
        f"Python: "
        f"{sys.version.split()[0]}"
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    if sys.version_info < (
        3,
        10,
    ):
        print(
            "WARNING: Python 3.10+ is recommended; "
            "use Python 3.11 for this project."
        )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    try:
        config = load_config()

        print(
            f"Config: OK "
            f"({CONFIG_PATH.name})"
        )

        video_config = config.get(
            "video",
            {},
        )

        perception_config = config.get(
            "perception",
            {},
        )

        print(
            "Video sampling FPS: "
            f"{video_config.get('sampling_fps', 'not set')}"
        )

        print(
            "Pose model: "
            f"{perception_config.get('pose_model', 'not set')}"
        )

    except (
        FileNotFoundError,
        ValueError,
        yaml.YAMLError,
    ) as exc:

        print(
            f"Config: ERROR — {exc}"
        )

        return 1

    # --------------------------------------------------------
    # Folders
    # --------------------------------------------------------

    expected_dirs = [
        PROJECT_ROOT
        / "data"
        / "videos",

        PROJECT_ROOT
        / "data"
        / "ground_truth",

        PROJECT_ROOT
        / "data"
        / "bed_regions",

        PROJECT_ROOT
        / "results",
    ]

    missing_dirs = [
        str(
            folder.relative_to(
                PROJECT_ROOT
            )
        )
        for folder in expected_dirs
        if not folder.is_dir()
    ]

    if missing_dirs:

        print(
            "Folders missing:"
        )

        for folder in missing_dirs:
            print(
                f"  - {folder}"
            )

        return 1

    print(
        "Project folders: OK"
    )

    # --------------------------------------------------------
    # Dependencies
    # --------------------------------------------------------

    installed, missing = (
        check_dependencies()
    )

    print(
        f"Dependencies installed: "
        f"{len(installed)}/"
        f"{len(DEPENDENCIES)}"
    )

    if installed:
        print(
            "  OK: "
            + ", ".join(installed)
        )

    if missing:

        print(
            "  Missing: "
            + ", ".join(missing)
        )

        print(
            "Install them with: "
            "python -m pip install "
            "-r requirements.txt"
        )

        return 1

    print(
        "\nSetup looks ready."
    )

    print(
        "\nAvailable stages:"
    )

    print(
        "  Step 2: "
        "--sample-video <path>"
    )

    print(
        "  Step 3A: "
        "--pose-video <path>"
    )

    print(
        "  Step 3B: "
        "--state-video <path>"
    )

    print(
        "  Step 4: "
        "--events-video <path>"
    )

    return 0


# ============================================================
# Video inspection
# ============================================================

def inspect_video(
    video_path: Path,
) -> int:
    """Check that OpenCV can open the video."""

    if not video_path.is_absolute():
        video_path = (
            Path.cwd()
            / video_path
        ).resolve()

    if not video_path.is_file():

        print(
            f"ERROR: Video file not found: "
            f"{video_path}",
            file=sys.stderr,
        )

        return 2

    capture = cv2.VideoCapture(
        str(video_path)
    )

    try:

        if not capture.isOpened():

            print(
                "ERROR: OpenCV could not "
                "open this video: "
                f"{video_path}",
                file=sys.stderr,
            )

            return 2

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
            or 0.0
        )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or 0
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or 0
        )

        duration_sec = (
            frame_count / fps
            if fps > 0
            else 0.0
        )

        details = {
            "video": str(video_path),
            "width": width,
            "height": height,
            "source_fps": round(
                fps,
                3,
            ),
            "frame_count": frame_count,
            "duration_sec": round(
                duration_sec,
                3,
            ),
        }

        print(
            json.dumps(
                details,
                indent=2,
            )
        )

        print(
            "\nVideo input check passed."
        )

        return 0

    finally:

        capture.release()


# ============================================================
# Step 2 - Frame sampling
# ============================================================

def sample_video_command(
    video_path: Path,
    sampling_fps: float | None = None,
    max_frames: int | None = None,
) -> int:
    """Sample a video and create frame_manifest.json."""

    if not video_path.is_absolute():

        video_path = (
            Path.cwd()
            / video_path
        ).resolve()

    try:

        config = load_config()

        video_config = config.get(
            "video",
            {},
        )

        output_config = config.get(
            "output",
            {},
        )

        # ----------------------------------------------------
        # Sampling configuration
        # ----------------------------------------------------

        rate = float(
            sampling_fps
            if sampling_fps is not None
            else video_config.get(
                "sampling_fps",
                1.0,
            )
        )

        frame_limit = (
            max_frames
            if max_frames is not None
            else video_config.get(
                "max_frames"
            )
        )

        if frame_limit is not None:
            frame_limit = int(
                frame_limit
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_root_value = (
            output_config.get(
                "directory",
                "results",
            )
        )

        output_root = Path(
            output_root_value
        )

        if not output_root.is_absolute():
            output_root = (
                PROJECT_ROOT
                / output_root
            )

        run_dir = (
            output_root
            / video_path.stem
        )

        frames_dir = (
            run_dir
            / "frames"
        )

        manifest_path = (
            run_dir
            / "frame_manifest.json"
        )

        # ----------------------------------------------------
        # Run sampler
        # ----------------------------------------------------

        manifest = sample_video(
            video_path=video_path,
            frames_dir=frames_dir,
            sampling_fps=rate,
            max_frames=frame_limit,
        )

        manifest[
            "frames_directory"
        ] = "frames"

        write_manifest(
            manifest,
            manifest_path,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print(
            "Frame sampling completed successfully."
        )

        print(
            f"Video: {video_path}"
        )

        print(
            f"Duration: "
            f"{manifest['duration_sec']:.2f} seconds"
        )

        print(
            f"Source FPS: "
            f"{manifest['source_fps']:.3f}"
        )

        print(
            f"Sampling FPS: "
            f"{manifest['sampling_fps_effective']:.3f}"
        )

        print(
            f"Frames saved: "
            f"{manifest['sampled_frame_count']}"
        )

        print(
            f"Frames folder: "
            f"{frames_dir}"
        )

        print(
            f"Manifest: "
            f"{manifest_path}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 2


# ============================================================
# Step 3A - YOLO Pose
# ============================================================

def pose_video_command(
    video_path: Path,
) -> int:
    """Run YOLO Pose estimation."""

    if not video_path.is_absolute():

        video_path = (
            Path.cwd()
            / video_path
        ).resolve()

    if not video_path.is_file():

        print(
            f"ERROR: Video file not found: "
            f"{video_path}",
            file=sys.stderr,
        )

        return 2

    try:

        config = load_config()

        perception_config = (
            config.get(
                "perception",
                {},
            )
        )

        output_config = (
            config.get(
                "output",
                {},
            )
        )

        # ----------------------------------------------------
        # Pose configuration
        # ----------------------------------------------------

        model_name = (
            perception_config.get(
                "pose_model",
                "yolo11n-pose.pt",
            )
        )

        confidence = float(
            perception_config.get(
                "pose_confidence",
                0.35,
            )
        )

        image_size = int(
            perception_config.get(
                "pose_image_size",
                640,
            )
        )

        device = (
            perception_config.get(
                "pose_device",
                "cpu",
            )
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_root_value = (
            output_config.get(
                "directory",
                "results",
            )
        )

        output_root = Path(
            output_root_value
        )

        if not output_root.is_absolute():
            output_root = (
                PROJECT_ROOT
                / output_root
            )

        run_dir = (
            output_root
            / video_path.stem
        )

        manifest_path = (
            run_dir
            / "frame_manifest.json"
        )

        # ----------------------------------------------------
        # Step 2 dependency
        # ----------------------------------------------------

        if not manifest_path.is_file():

            print(
                "ERROR: frame_manifest.json "
                "was not found.",
                file=sys.stderr,
            )

            print(
                f"Expected: {manifest_path}",
                file=sys.stderr,
            )

            print(
                "\nRun Step 2 first with:",
                file=sys.stderr,
            )

            print(
                "  python -m src.main "
                f"--sample-video {video_path}",
                file=sys.stderr,
            )

            return 2

        pose_output_dir = (
            run_dir
            / "pose"
        )

        print(
            "\nStarting YOLO Pose estimation..."
        )

        print(
            f"Video: {video_path}"
        )

        print(
            f"Model: {model_name}"
        )

        print(
            f"Confidence threshold: "
            f"{confidence}"
        )

        print(
            f"Image size: {image_size}"
        )

        print(
            f"Device: {device}"
        )

        print()

        result = run_pose_pipeline(
            manifest_path=manifest_path,
            output_dir=pose_output_dir,
            model_name=model_name,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )

        print(
            "\nYOLO pose estimation "
            "completed successfully."
        )

        print(
            f"Frames processed: "
            f"{result['frame_count']}"
        )

        print(
            "Frames with person detected: "
            f"{result['person_detected_count']}"
        )

        print(
            f"Pose output directory: "
            f"{pose_output_dir}"
        )

        print(
            "Combined pose manifest: "
            f"{pose_output_dir / 'pose_manifest.json'}"
        )

        print(
            "Annotated frames: "
            f"{pose_output_dir / 'annotated'}"
        )

        print(
            "Per-frame JSON observations: "
            f"{pose_output_dir / 'observations'}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 2


# ============================================================
# Step 3B - Activity/state classification
# ============================================================

def state_video_command(
    video_path: Path,
) -> int:
    """Classify activity and bed occupancy."""

    if not video_path.is_absolute():

        video_path = (
            Path.cwd()
            / video_path
        ).resolve()

    if not video_path.is_file():

        print(
            f"ERROR: Video file not found: "
            f"{video_path}",
            file=sys.stderr,
        )

        return 2

    try:

        config = load_config()

        state_config = (
            config.get(
                "state",
                {},
            )
        )

        output_config = (
            config.get(
                "output",
                {},
            )
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_root_value = (
            output_config.get(
                "directory",
                "results",
            )
        )

        output_root = Path(
            output_root_value
        )

        if not output_root.is_absolute():
            output_root = (
                PROJECT_ROOT
                / output_root
            )

        run_dir = (
            output_root
            / video_path.stem
        )

        pose_manifest_path = (
            run_dir
            / "pose"
            / "pose_manifest.json"
        )

        # ----------------------------------------------------
        # Step 3A dependency
        # ----------------------------------------------------

        if not pose_manifest_path.is_file():

            print(
                "ERROR: pose_manifest.json "
                "was not found.",
                file=sys.stderr,
            )

            print(
                f"Expected: "
                f"{pose_manifest_path}",
                file=sys.stderr,
            )

            print(
                "\nRun Step 3A first with:",
                file=sys.stderr,
            )

            print(
                "  python -m src.main "
                f"--pose-video {video_path}",
                file=sys.stderr,
            )

            return 2

        state_output_dir = (
            run_dir
            / "state"
        )

        print(
            "\nStarting activity/state "
            "classification..."
        )

        print(
            f"Video: {video_path}"
        )

        print(
            f"Pose manifest: "
            f"{pose_manifest_path}"
        )

        print()

        result = (
            classify_pose_manifest(
                pose_manifest_path=(
                    pose_manifest_path
                ),
                output_dir=(
                    state_output_dir
                ),
                thresholds=(
                    state_config
                ),
            )
        )

        # ----------------------------------------------------
        # Count activities
        # ----------------------------------------------------

        activity_counts: dict[
            str,
            int,
        ] = {}

        occupancy_counts: dict[
            str,
            int,
        ] = {}

        for observation in (
            result["observations"]
        ):

            activity = (
                observation["activity"]
            )

            occupancy = (
                observation[
                    "bed_occupancy"
                ]
            )

            activity_counts[
                activity
            ] = (
                activity_counts.get(
                    activity,
                    0,
                )
                + 1
            )

            occupancy_counts[
                occupancy
            ] = (
                occupancy_counts.get(
                    occupancy,
                    0,
                )
                + 1
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            "\nState classification "
            "completed successfully."
        )

        print(
            f"Frames processed: "
            f"{result['frame_count']}"
        )

        print(
            "\nActivity counts:"
        )

        for activity, count in (
            activity_counts.items()
        ):

            print(
                f"  {activity}: {count}"
            )

        print(
            "\nBed occupancy counts:"
        )

        for occupancy, count in (
            occupancy_counts.items()
        ):

            print(
                f"  {occupancy}: {count}"
            )

        print(
            "\nState manifest: "
            f"{state_output_dir / 'state_manifest.json'}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 2


# ============================================================
# Step 4 - Temporal tracking and events
# ============================================================

def events_video_command(
    video_path: Path,
) -> int:
    """Run temporal smoothing, timeline generation,
    and bed exit/return detection.
    """

    if not video_path.is_absolute():

        video_path = (
            Path.cwd()
            / video_path
        ).resolve()

    if not video_path.is_file():

        print(
            f"ERROR: Video file not found: "
            f"{video_path}",
            file=sys.stderr,
        )

        return 2

    try:

        config = load_config()

        state_config = (
            config.get(
                "state",
                {},
            )
        )

        events_config = (
            config.get(
                "events",
                {},
            )
        )

        output_config = (
            config.get(
                "output",
                {},
            )
        )

        # ----------------------------------------------------
        # Output root
        # ----------------------------------------------------

        output_root_value = (
            output_config.get(
                "directory",
                "results",
            )
        )

        output_root = Path(
            output_root_value
        )

        if not output_root.is_absolute():
            output_root = (
                PROJECT_ROOT
                / output_root
            )

        run_dir = (
            output_root
            / video_path.stem
        )

        # ----------------------------------------------------
        # Step 3B dependency
        # ----------------------------------------------------

        state_manifest_path = (
            run_dir
            / "state"
            / "state_manifest.json"
        )

        if not state_manifest_path.is_file():

            print(
                "ERROR: state_manifest.json "
                "was not found.",
                file=sys.stderr,
            )

            print(
                f"Expected: "
                f"{state_manifest_path}",
                file=sys.stderr,
            )

            print(
                "\nRun Step 3B first with:",
                file=sys.stderr,
            )

            print(
                "  python -m src.main "
                f"--state-video {video_path}",
                file=sys.stderr,
            )

            return 2

        # ----------------------------------------------------
        # Step 4 configuration
        # ----------------------------------------------------

        smoothing_window = int(
            state_config.get(
                "smoothing_window",
                5,
            )
        )

        min_state_duration_sec = float(
            state_config.get(
                "min_state_duration_sec",
                2.0,
            )
        )

        unknown_confidence_threshold = (
            float(
                state_config.get(
                    "unknown_confidence_threshold",
                    0.4,
                )
            )
        )

        min_event_duration_sec = (
            float(
                events_config.get(
                    "min_stable_duration_sec",
                    min_state_duration_sec,
                )
            )
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        temporal_output_dir = (
            run_dir
            / "timeline"
        )

        print(
            "\nStarting temporal state "
            "tracking..."
        )

        print(
            f"State manifest: "
            f"{state_manifest_path}"
        )

        print(
            f"Smoothing window: "
            f"{smoothing_window}"
        )

        print(
            f"Minimum state duration: "
            f"{min_state_duration_sec:.2f}s"
        )

        print(
            f"Event confirmation duration: "
            f"{min_event_duration_sec:.2f}s"
        )

        print()

        # ----------------------------------------------------
        # Build timeline
        # ----------------------------------------------------

        timeline = build_timeline(
            state_manifest_path=(
                state_manifest_path
            ),
            output_dir=(
                temporal_output_dir
            ),
            smoothing_window=(
                smoothing_window
            ),
            min_state_duration_sec=(
                min_state_duration_sec
            ),
            unknown_confidence_threshold=(
                unknown_confidence_threshold
            ),
        )

        timeline_path = (
            temporal_output_dir
            / "timeline.json"
        )

        # ----------------------------------------------------
        # Detect events
        # ----------------------------------------------------

        events = detect_bed_events(
            timeline=timeline,
            min_stable_duration_sec=(
                min_event_duration_sec
            ),
        )

        events_path = (
            temporal_output_dir
            / "bed_events.json"
        )

        save_events(
            events=events,
            timeline=timeline,
            timeline_path=timeline_path,
            output_path=events_path,
        )

        # ----------------------------------------------------
        # Print timeline
        # ----------------------------------------------------

        print(
            "\nTemporal tracking "
            "completed successfully."
        )

        print(
            f"Stable segments: "
            f"{len(timeline['segments'])}"
        )

        print(
            "\nTimeline:"
        )

        for segment in (
            timeline["segments"]
        ):

            print(
                f"  "
                f"{segment['start_sec']:6.1f}s - "
                f"{segment['end_sec']:6.1f}s  "
                f"{segment['activity']:20s} "
                f"{segment['bed_occupancy']}"
            )

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        print(
            f"\nBed events detected: "
            f"{len(events)}"
        )

        if not events:

            print(
                "  No bed exit/return events "
                "were confirmed."
            )

        for event in events:
            print(
                f"  {event['event_type']} "
                f"start={event['start_time_sec']:.1f}s "
                f"confirmed={event['confirmed_time_sec']:.1f}s "
                f"confidence={event['confidence']:.2f} "
                f"({event['evidence_strength']})"
            )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        print(
            f"\nTimeline file: "
            f"{timeline_path}"
        )

        print(
            f"Events file: "
            f"{events_path}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 2


# ============================================================
# CLI parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Elderly Monitor: video processing, "
            "YOLO Pose, state recognition, "
            "temporal tracking, and bed events."
        )
    )

    group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    group.add_argument(
        "--check-setup",
        action="store_true",
        help=(
            "Check Python environment, "
            "dependencies, config, "
            "and project folders."
        ),
    )

    # --------------------------------------------------------
    # Video inspection
    # --------------------------------------------------------

    group.add_argument(
        "--video",
        type=Path,
        help=(
            "Inspect video metadata."
        ),
    )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    group.add_argument(
        "--sample-video",
        type=Path,
        help=(
            "Sample frames from a video."
        ),
    )

    # --------------------------------------------------------
    # Step 3A
    # --------------------------------------------------------

    group.add_argument(
        "--pose-video",
        type=Path,
        help=(
            "Run YOLO Pose estimation."
        ),
    )

    # --------------------------------------------------------
    # Step 3B
    # --------------------------------------------------------

    group.add_argument(
        "--state-video",
        type=Path,
        help=(
            "Classify activity/state "
            "from pose results."
        ),
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    group.add_argument(
        "--events-video",
        type=Path,
        help=(
            "Run temporal smoothing, "
            "timeline generation, and "
            "bed exit/return detection."
        ),
    )

    # --------------------------------------------------------
    # Optional Step 2 parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--sampling-fps",
        type=float,
        help=(
            "Override video.sampling_fps."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Maximum number of sampled frames."
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> int:
    """CLI entry point."""

    args = (
        build_parser()
        .parse_args()
    )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    if args.check_setup:

        return check_setup()

    # --------------------------------------------------------
    # Video inspection
    # --------------------------------------------------------

    if args.video is not None:

        return inspect_video(
            args.video
        )

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    if args.sample_video is not None:

        return sample_video_command(
            args.sample_video,
            sampling_fps=(
                args.sampling_fps
            ),
            max_frames=(
                args.max_frames
            ),
        )

    # --------------------------------------------------------
    # Step 3A
    # --------------------------------------------------------

    if args.pose_video is not None:

        return pose_video_command(
            args.pose_video
        )

    # --------------------------------------------------------
    # Step 3B
    # --------------------------------------------------------

    if args.state_video is not None:

        return state_video_command(
            args.state_video
        )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    if args.events_video is not None:

        return events_video_command(
            args.events_video
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )