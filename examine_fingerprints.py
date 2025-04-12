from pathlib import Path
import pandas as pd
import json
from pathpilot.io import load_json_folder, flatten_sessions, clean
from pathpilot.features import compute_fingerprint, build_feature_matrix

# Load and process the data
print("1. Loading JSON data...")
data_dir = Path("./data")
raw = load_json_folder(data_dir)
print(f"   Loaded {len(raw)} JSON files")

print("\n2. Flattening sessions...")
df_flat = clean(flatten_sessions(raw))
print(f"   Created DataFrame with {len(df_flat)} rows")
print(f"   Columns: {', '.join(df_flat.columns)}")

print("\n3. Sample of flattened data:")
print(df_flat.head(3).to_string())

print("\n4. Computing fingerprints for each session...")
# Group by session_id and compute fingerprint for each
grouped = df_flat.groupby("session_id")
print(f"   Found {len(grouped)} unique sessions")

# Show a sample fingerprint
sample_session_id = list(grouped.groups.keys())[0]
sample_session = grouped.get_group(sample_session_id)
print(f"\n   Sample session: {sample_session_id}")
print(f"   Label: {sample_session['label_dir'].iloc[0]}")
print(f"   Number of steps: {len(sample_session)}")

fingerprint = compute_fingerprint(sample_session)
print("\n   Fingerprint:")
for feature, value in fingerprint.items():
    print(f"   - {feature}: {value}")

print("\n5. Building feature matrix...")
X, y = build_feature_matrix(df_flat)
print(f"   Feature matrix shape: {X.shape}")
print(f"   Target vector shape: {y.shape}")

print("\n6. Feature matrix sample:")
print(X.head(3).to_string())

print("\n7. Target distribution:")
print(y.value_counts().to_string()) 