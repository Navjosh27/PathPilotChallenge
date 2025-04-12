# Command-line interface entry point for PathPilot.

from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

from .io import load_json_folder, flatten_sessions, clean
from .features import build_feature_matrix
from .visualize import plot_timeline
from .model import train, evaluate, save


def main() -> None:
    parser = argparse.ArgumentParser(description="PathPilot user‑journey analysis")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
        help="Folder containing upgrades/ and cancellations/ sub‑folders",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path("./plots"),
        help="Destination folder for timeline PNGs",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to save the trained model (joblib)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip timeline generation step",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    print("🔄  Loading JSON …")
    raw = load_json_folder(args.data)
    df_flat = clean(flatten_sessions(raw))

    if not args.no_plots:
        print("🖼️  Generating timelines …")
        for session_id, group in df_flat.groupby("session_id"):
            out_file = args.plots / f"{session_id}.png"
            plot_timeline(group, out_file)

    print("🧮  Building feature matrix …")
    X, y = build_feature_matrix(df_flat)

    print("🧑‍🏫  Training model …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train(X_tr, y_tr)

    print("📊  Evaluation:")
    metrics = evaluate(model, X_te, y_te)
    print(f"Accuracy: {metrics['accuracy']}")
    print(metrics["report"])

    if args.model_out:
        save(model, args.model_out)
        print(f"💾  Model saved to {args.model_out}")


if __name__ == "__main__":
    main() 