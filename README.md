# PathPilot Challenge

## Setup
```bash
git clone <your‑repo>
cd PathPilotChallenge
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline
```bash
python -m pathpilot.cli \
  --data ./data \
  --plots ./plots \
  --model-out pathpilot_model.joblib
```

Generates timeline PNGs in ./plots

Trains a Random‑Forest classifier to predict upgrade (1) vs cancellation (0)

Prints evaluation metrics

Saves the model (optional)

## Project layout
```
PathPilotChallenge/
├── data/               # upgrades/ & cancellations/ JSON
├── pathpilot/          # source code
├── tests/              # pytest unit tests
├── plots/              # generated timelines
└── requirements.txt
```

## Extending
- `features.py` – add richer journey features
- `model.py` – swap in a different ML algorithm
- `visualize.py` – customize timeline aesthetics

## License
MIT 