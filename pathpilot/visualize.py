# Functions for plotting journey timelines and visualizations.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = ["plot_timeline"]

STATUS_COLORS = {
    "non-converted": "#6c757d",       # gray
    "attempted_conversion": "#ffc107",# amber
    "converted": "#28a745",           # green
    "conversion_at_risk": "#dc3545",  # red
}

def plot_timeline(df_session: pd.DataFrame, out_path: Path) -> None:
    """
    Draw a horizontal timeline for one session and save as PNG.
    """
    df = df_session.sort_values("start_ms")
    fig, ax = plt.subplots(figsize=(10, 1 + 0.25 * len(df)))

    for idx, row in df.iterrows():
        ax.barh(
            y=0,
            width=row["end_ms"] - row["start_ms"],
            left=row["start_ms"],
            color=STATUS_COLORS.get(row["status"], "#999999"),
            edgecolor="none",
            height=0.6,
        )

    ax.set_yticks([])
    ax.set_xlabel("milliseconds since session start")
    ax.set_title(f"Session {df['session_id'].iloc[0]}")

    # legend once
    handles = [Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig) 