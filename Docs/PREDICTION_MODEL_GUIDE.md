# NBA 2K25 Esports Match Prediction Guide

This guide explains how to use the unified match prediction system that predicts both match winners and total scores consistently.

## Overview

The prediction system uses a **score-first approach** where:
1. **Individual player scores** are predicted for each match
2. **Winner** is determined by who has the higher predicted score  
3. **Total score** is the sum of both predicted scores
4. **Win probability** is derived from the score difference

This approach eliminates conflicting predictions between winner and score models.

## Model Architecture

- **Model Type**: Multi-output XGBoost Regressor
- **Targets**: [Home Player Score, Away Player Score]
- **Features**: 40+ engineered features including:
  - Individual player performance metrics
  - Recent form and momentum
  - Head-to-head records
  - Team preferences
  - Relative performance differences

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
# Using CLI
cd backend/app
python cli.py train-model --days 60 --min-matches 5

# Using API
curl -X POST "http://localhost:5000/api/ml/train?days_back=60&min_matches_per_player=5"
```

### 3. Generate Predictions

```bash
# Using CLI
python cli.py predict

# Using API
curl "http://localhost:5000/api/ml/predictions"
```

## CLI Commands

### Train the Model
```bash
python cli.py train-model --days 60 --min-matches 5
```
- `--days`: Number of days of match history to use (default: 60)
- `--min-matches`: Minimum matches required per player (default: 5)

### Generate Predictions
```bash
python cli.py predict
```
Generates predictions for all upcoming matches and displays detailed results.

### Evaluate Model Performance
```bash
python cli.py evaluate-model --test-days 7
```
Tests the model on recent matches to assess accuracy.

### View Feature Importance
```bash
python cli.py feature-importance --top-n 20
```
Shows the most important features used by the model.

## API Endpoints

### Train Model
```http
POST /api/ml/train?days_back=60&min_matches_per_player=5
```

### Get Predictions
```http
GET /api/ml/predictions
```

### Get Predictions Summary
```http
GET /api/ml/predictions/summary
```

### Evaluate Model
```http
GET /api/ml/model-performance?test_days=7
```

### Feature Importance
```http
GET /api/ml/feature-importance
```

### Retrain Model
```http
POST /api/ml/retrain?days_back=60
```

## Key Features

### 1. Consistent Predictions
- Winner is always determined by predicted scores
- No conflicting predictions between winner and score models
- Win probability derived from score difference

### 2. Rich Feature Engineering
- **Individual Stats**: Average score, win rate, recent form, momentum
- **Head-to-Head**: Historical performance against specific opponents  
- **Relative**: Performance differences between players
- **Team**: Performance with specific teams
- **Consistency**: Score variance and reliability metrics

### 3. Model Interpretability
- Feature importance rankings
- Confidence scores for predictions
- Performance metrics and evaluation

## Prediction Output

Each prediction includes:

```json
{
  "match": "Player A vs Player B",
  "predicted_winner": "Player A", 
  "home_win_probability": 0.72,
  "predicted_scores": {
    "home": 85.3,
    "away": 78.1, 
    "total": 163.4
  },
  "confidence": 0.45,
  "fixture_start": "2025-06-21T19:00:00Z"
}
```

## Model Performance Metrics

The system tracks several metrics:
- **Winner Accuracy**: Percentage of correct winner predictions
- **Score MAE**: Mean Absolute Error for individual scores
- **Total Score MAE**: Mean Absolute Error for total match score
- **Confidence**: Measure of prediction certainty

## Best Practices

### 1. Data Quality
- Ensure sufficient historical data (recommended: 60+ days)
- Require minimum matches per player (recommended: 5+)
- Regularly retrain with fresh data

### 2. Model Maintenance
- Retrain weekly or after significant events
- Monitor performance metrics
- Evaluate on recent matches regularly

### 3. Interpretation
- Higher confidence scores indicate more reliable predictions
- Consider head-to-head history for key matchups
- Factor in recent form and momentum trends

## Troubleshooting

### Common Issues

1. **"No training data available"**
   - Run data fetching first: `python cli.py fetch-matches`
   - Ensure player stats are calculated: `python cli.py calculate-stats`

2. **"Model not found"**
   - Train the model first: `python cli.py train-model`

3. **Low prediction accuracy**
   - Increase training data: use more `--days`
   - Check data quality and feature engineering
   - Consider retraining with recent data

### Performance Optimization

- **More Training Data**: Use 60-90 days for better performance
- **Feature Engineering**: The model uses 40+ features automatically
- **Regular Retraining**: Weekly retraining recommended
- **Data Quality**: Ensure clean, consistent data

## Advanced Usage

### Custom Feature Engineering

You can extend the `MatchPredictionFeatureEngineer` class to add new features:

```python
def _get_custom_features(self, home_stats, away_stats):
    features = {}
    # Add your custom features here
    features['custom_metric'] = calculate_custom_metric(home_stats, away_stats)
    return features
```

### Model Tuning

Adjust model parameters in `MatchPredictionModel`:

```python
self.model_params = {
    'n_estimators': 200,      # Increase for better performance
    'max_depth': 6,           # Adjust for complexity
    'learning_rate': 0.1,     # Lower for more stable learning
    'subsample': 0.8,         # Reduce overfitting
    'colsample_bytree': 0.8   # Feature sampling
}
```

## Integration Example

```python
from backend.services.match_prediction_service import MatchPredictionService

# Initialize service
service = MatchPredictionService()

# Train model
training_df = service.prepare_training_data(days_back=60)
metrics = service.train_model(training_df)

# Generate predictions  
predictions = service.predict_upcoming_matches()
summary = service.get_prediction_summary(predictions)

# Display results
for pred in summary['predictions']:
    print(f"{pred['home_player']} vs {pred['away_player']}")
    print(f"Winner: {pred['predicted_winner']} ({pred['home_win_probability']:.1%})")
    print(f"Score: {pred['predicted_scores']['home']:.1f} - {pred['predicted_scores']['away']:.1f}")
```

This unified approach ensures consistent, interpretable predictions for both match outcomes and scoring, eliminating the confusion from conflicting separate models.
