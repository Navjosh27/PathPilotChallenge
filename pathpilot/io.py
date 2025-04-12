# Utilities for loading and cleaning journey data.

from pathlib import Path
from typing import List, Dict
import json
import pandas as pd

__all__ = ["load_json_folder", "flatten_sessions", "clean"]


def load_json_folder(root: Path) -> List[Dict]:
    """
    Recursively load every *.json file under
    root/upgrades and root/cancellations.

    Adds helper keys:
        filepath  – Path object
        label_dir – 'upgrades' or 'cancellations'
    """
    raw: List[Dict] = []
    for label_dir in ("upgrades", "cancellations"):
        for fp in (root / label_dir).rglob("*.json"):
            with fp.open("r", encoding="utf‑8") as f:
                data = json.load(f)
            data["filepath"] = fp
            data["label_dir"] = label_dir
            raw.append(data)
    return raw


def flatten_sessions(raw: List[Dict]) -> pd.DataFrame:
    """
    Flatten nested recording‑chunk JSON into one row per step.

    Columns:
        session_id • step_idx • label_dir • filename • label
        date • status • start_ms • end_ms • details
    """
    records = []
    for doc in raw:
        label_dir = doc["label_dir"]
        filename = doc["filepath"].name
        
        # Process each session in the document
        for session_id, session_data in doc.items():
            if session_id in ("filepath", "label_dir"):
                continue
                
            if "steps" not in session_data:
                continue
                
            for idx, step in enumerate(session_data["steps"]):
                records.append(
                    {
                        "session_id": session_id,
                        "step_idx": idx,
                        "label_dir": label_dir,
                        "filename": filename,
                        "label": step["label"],
                        "date": step["date"],
                        "status": step["status"],
                        "start_ms": step["recordingReel"]["start_ms_since"],
                        "end_ms": step["recordingReel"]["end_ms_since"],
                        "details": step["recordingReel"]["details"],
                    }
                )
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type‑cast columns and drop rows missing critical fields."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    df["end_ms"] = pd.to_numeric(df["end_ms"], errors="coerce")
    df = df.dropna(subset=["session_id", "status"])
    return df 