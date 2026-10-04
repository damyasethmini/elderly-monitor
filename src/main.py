"""Command-line entry point for the elderly-monitor assignment.

This module provides the core vision and temporal pipeline:

    Step 1  - environment/setup checks
    Step 2  - frame sampling
    Step 3A - YOLO Pose estimation
    Step 3B - activity and bed-occupancy classification
    Step 4  - temporal state tracking and bed exit/return detection

Steps 5-7 are exposed through their dedicated modules:
    Step 5  - ``python -m src.agent.agentic_analysis``
    Step 6  - ``python -m src.alerts.contextual_alert``
    Step 7  - ``python -m src.evaluation.evaluate``

The project is local-first and does not require an external LLM API.
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

from src.perception.frame_sampler import sample_video, write_manifest
from src.perception.detector import run_pose_pipeline
from src.state.classifier import classify_pose_manifest
from src.state.state_machine import build_timeline
from src.events.bed_events import detect_bed_events, save_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


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


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration and validate that it contains a mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml must contain a YAML mapping at the top level."
        )

    return config


def check_dependencies() -> tuple[list[str], list[str]]:
    """Return installed and missing dependency names."""
    installed: list[str] = []
    missing: list[str] = []

    for import_name, package_name in DEPENDENCIES.items():
        try:
            found = importlib.util.find_spec(import_name) is not None
        except (ImportError, ValueError):
            found = False

        if found:
            installed.append(package_name)
        else:
            missing.append(package_name)

    return installed, missing


def check_setup() -> int:
    """Check Python, dependencies, config, and expected project folders."""
    print("Elderly Monitor — setup check")
    print("=" * 32)
