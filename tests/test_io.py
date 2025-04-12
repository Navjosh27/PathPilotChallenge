# Tests for data loading and cleaning utilities.

from pathlib import Path
from pathpilot.io import load_json_folder, flatten_sessions, clean


def test_flatten_has_core_columns():
    data_dir = Path("./data")
    raw = load_json_folder(data_dir)

    df = clean(flatten_sessions(raw))
    assert not df.empty
    assert {"session_id", "status", "label_dir"}.issubset(df.columns)

def test_load_journey_data():
    pass 