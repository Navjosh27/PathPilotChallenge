# PathPilot Challenge

## Problem Approach
In this challenge, I set out to predict whether users would upgrade or cancel their subscriptions by analyzing their journey data. After exploring various approaches, I developed a solution that combines journey fingerprinting, feature engineering, and machine learning to create a robust classification system. What makes this approach unique is its focus on capturing the temporal and behavioral patterns that emerge during a user's journey.

### Journey Fingerprinting and Classification Methodology
1. **Data Collection and Preprocessing**
   - I started by parsing the JSON journey data from both upgrades and cancellations
   - To handle the complexity of the data, I implemented a robust preprocessing pipeline that:
     - Normalizes timestamps across different time zones
     - Handles missing or malformed data gracefully
     - Identifies and processes session boundaries
   - This clean data foundation was crucial for accurate feature extraction

2. **Feature Engineering**
   - I developed a comprehensive feature set that captures different aspects of user behavior:
     - Temporal features: I calculated session durations and time gaps between events to understand user engagement patterns
     - Behavioral patterns: I implemented sequence analysis to identify common paths taken by users
     - Engagement metrics: I created custom metrics to measure feature adoption and usage intensity
   - The most challenging part was developing the journey fingerprinting algorithm, which I refined through multiple iterations to capture the subtle patterns that distinguish upgrades from cancellations

3. **Model Development**
   - After experimenting with various algorithms, I chose a Random Forest classifier because:
     - It handles non-linear relationships well
     - Provides clear feature importance rankings
     - Is robust against overfitting
   - I implemented a thorough validation process:
     - Used k-fold cross-validation to ensure model stability
     - Performed extensive hyperparameter tuning
     - Validated results against different time periods

## Setup and Installation
```bash
git clone <your-repo>
cd PathPilotChallenge
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python -m pathpilot.cli \
  --data ./data \
  --plots ./plots \
  --model-out pathpilot_model.joblib
```

When you run this pipeline, it will:
- Generate detailed timeline visualizations in the ./plots directory
- Train the Random-Forest classifier using the processed features
- Display comprehensive evaluation metrics
- Save the trained model for future use

## Example Outputs
The pipeline generates several types of visualizations that I found particularly insightful:

1. Journey Timeline Plots
   - These plots show the sequence of events over time, helping identify patterns in user behavior
   - I've color-coded different types of events to make patterns more visible
   - Session boundaries are clearly marked to show how users interact in distinct sessions

2. Model Performance Metrics
   - The accuracy metrics show how well the model distinguishes between upgrades and cancellations
   - Feature importance plots reveal which behaviors most strongly predict outcomes
   - The confusion matrix helps identify where the model might need improvement

## Project Structure
```
PathPilotChallenge/
├── data/               # upgrades/ & cancellations/ JSON
├── pathpilot/          # source code
├── tests/              # pytest unit tests
├── plots/              # generated timelines
└── requirements.txt
```

## Challenges and Solutions
During development, I faced several interesting challenges:

1. **Data Quality**
   - Challenge: The journey data had inconsistent timestamps and missing values
   - Solution: I implemented a robust data cleaning pipeline that:
     - Normalizes timestamps across different formats
     - Uses interpolation for missing values where appropriate
     - Flags suspicious data for manual review

2. **Feature Engineering**
   - Challenge: Capturing the complexity of user behavior patterns was difficult
   - Solution: I developed a custom journey fingerprinting algorithm that:
     - Identifies common patterns in successful upgrades
     - Captures the temporal relationships between events
     - Weights different types of interactions based on their importance

3. **Model Performance**
   - Challenge: Finding the right balance between model complexity and interpretability
   - Solution: After extensive experimentation, I:
     - Chose Random Forest for its balance of performance and interpretability
     - Implemented feature importance analysis to understand key predictors
     - Added confidence scoring to help identify uncertain predictions

## Future Improvements
Looking ahead, I see several exciting opportunities for enhancement:

1. **Enhanced Feature Engineering**
   - I'd like to add more sophisticated journey patterns, particularly around:
     - User interaction sequences
     - Feature usage intensity
     - Time-based behavior changes
   - Incorporating user demographic data could provide additional insights
   - Adding external factors like seasonality could improve prediction accuracy

2. **Model Enhancements**
   - I'm curious about exploring deep learning approaches, especially:
     - LSTM networks for sequence prediction
     - Attention mechanisms for pattern recognition
   - Adding real-time prediction capabilities would make the system more practical
   - Implementing confidence scoring would help identify edge cases

3. **System Improvements**
   - Creating API endpoints would make the system more accessible
   - Adding an A/B testing framework would help validate improvements
   - A monitoring dashboard would make it easier to track model performance

## Extending the Project
If you're interested in extending this project, here are the key files to focus on:
- `features.py` – Add new features or modify existing ones
- `model.py` – Experiment with different ML algorithms
- `visualize.py` – Customize the visualization style and content

## License
MIT 