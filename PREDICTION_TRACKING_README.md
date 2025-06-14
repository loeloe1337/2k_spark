# Real Prediction Tracking System

This document explains the new real prediction tracking system that replaces the previous mock data approach.

## Overview

The system now tracks **real predictions** made for upcoming matches and validates them against actual match results once the games are completed. This provides accurate prediction accuracy metrics instead of artificially generated data.

## How It Works

### 1. Prediction Generation
When predictions are generated for upcoming matches:
- Real predictions are saved to `upcoming_match_predictions.json`
- The same predictions are automatically added to `prediction_history.json` with validation fields set to `null`
- Each prediction includes:
  - Match details (players, teams, fixture start time)
  - Win probability predictions
  - Score predictions
  - Generation timestamp

### 2. Prediction Validation
Once matches are completed:
- The validation service fetches actual match results from the API
- It matches completed matches with pending predictions
- Updates prediction history with:
  - Actual scores (`homeScore`, `awayScore`)
  - Actual result (`result`)
  - Prediction correctness (`prediction_correct`)
  - Score prediction errors

### 3. Accuracy Calculation
The frontend displays real accuracy metrics based on validated predictions.

## Key Files

### Scripts
- `scripts/generate_predictions.py` - Generates real predictions for upcoming matches
- `scripts/validate_predictions.py` - Validates predictions against completed matches
- `scripts/clear_mock_predictions.py` - Clears mock data (one-time use)

### Services
- `services/prediction_validation_service.py` - Core validation logic
- `services/refresh_service.py` - Automatically runs validation during data refresh

### Data Files
- `output/upcoming_match_predictions.json` - Current predictions for upcoming matches
- `output/prediction_history.json` - Historical predictions with validation results
- `output/prediction_history.backup.json` - Backup of previous mock data

## Usage

### Command Line Interface

```bash
# Generate new predictions for upcoming matches
python app/cli.py generate

# Manually validate completed predictions
python app/cli.py validate

# Clear mock data (already done)
python app/cli.py clear-mock
```

### Direct Script Usage

```bash
# Generate predictions
python scripts/generate_predictions.py

# Validate predictions
python scripts/validate_predictions.py

# Clear mock data
python scripts/clear_mock_predictions.py
```

### Automatic Validation
Prediction validation runs automatically when the refresh service updates data:
```bash
python app/cli.py refresh
```

## Prediction History Structure

### Before Validation (New Predictions)
```json
{
  "fixtureId": 235955,
  "homePlayer": {"id": 975, "name": "DOMAIN"},
  "awayPlayer": {"id": 2238, "name": "BABYLON"},
  "homeTeam": {"id": 28, "name": "Toronto Raptors"},
  "awayTeam": {"id": 16, "name": "Miami Heat"},
  "fixtureStart": "2025-06-14T07:47:00Z",
  "prediction": {
    "home_win_probability": 0.42,
    "away_win_probability": 0.58,
    "predicted_winner": "away",
    "confidence": 0.58
  },
  "score_prediction": {
    "home_score": 51,
    "away_score": 53,
    "total_score": 104,
    "score_diff": -2
  },
  "generated_at": "2025-06-14T09:05:34.517579",
  "homeScore": null,
  "awayScore": null,
  "result": null,
  "prediction_correct": null,
  "home_score_error": null,
  "away_score_error": null,
  "total_score_error": null
}
```

### After Validation (Completed Matches)
```json
{
  "fixtureId": 235955,
  "homePlayer": {"id": 975, "name": "DOMAIN"},
  "awayPlayer": {"id": 2238, "name": "BABYLON"},
  "homeTeam": {"id": 28, "name": "Toronto Raptors"},
  "awayTeam": {"id": 16, "name": "Miami Heat"},
  "fixtureStart": "2025-06-14T07:47:00Z",
  "prediction": {
    "home_win_probability": 0.42,
    "away_win_probability": 0.58,
    "predicted_winner": "away",
    "confidence": 0.58
  },
  "score_prediction": {
    "home_score": 51,
    "away_score": 53,
    "total_score": 104,
    "score_diff": -2
  },
  "generated_at": "2025-06-14T09:05:34.517579",
  "homeScore": 48,
  "awayScore": 55,
  "result": "away_win",
  "prediction_correct": true,
  "home_score_error": 3,
  "away_score_error": 2,
  "total_score_error": 1
}
```

## Migration from Mock Data

✅ **Completed Steps:**
1. Backed up 334 mock predictions to `prediction_history.backup.json`
2. Cleared mock data from prediction history
3. Generated 37 new real predictions for upcoming matches
4. Updated prediction generation to save to history
5. Enhanced CLI with new commands

## Monitoring Prediction Accuracy

### Real-Time Tracking
- New predictions are automatically added to history when generated
- Validation runs automatically during data refresh
- Frontend displays real accuracy metrics

### Manual Validation
Run validation manually to check for newly completed matches:
```bash
python app/cli.py validate
```

### Logs
Check logs for validation activity:
- `logs/prediction_refresh.log` - Prediction generation and validation logs
- `logs/data_fetcher.log` - Match data fetching logs

## Benefits of Real Tracking

1. **Accurate Metrics**: Real prediction accuracy instead of artificial 70% bias
2. **Transparent Process**: Clear audit trail of when predictions were made
3. **Continuous Improvement**: Ability to analyze prediction performance over time
4. **Real Validation**: Actual match results from live API data
5. **Automated Workflow**: Predictions are validated automatically as matches complete

## Next Steps

1. **Wait for Matches**: Allow upcoming matches to complete
2. **Monitor Validation**: Check logs for automatic validation
3. **Review Accuracy**: View real accuracy metrics in the frontend
4. **Analyze Performance**: Use validated data to improve prediction models

The system is now properly configured to track real prediction performance and provide meaningful accuracy metrics.