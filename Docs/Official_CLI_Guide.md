# 📋 Official CLI Guide - NBA 2K25 eSports Match Prediction System

Welcome to the comprehensive guide for using the NBA 2K25 eSports Match Prediction System CLI. This guide will walk you through all available commands and how to use them effectively.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Required dependencies installed (`pip install -r requirements.txt`)
- Valid authentication token (automatically handled by pipeline)

### ⚡ Fastest Way to Get Started
Run this single command to get everything set up and generate predictions:

```bash
python -m backend.app.cli run-pipeline --train
```

This will automatically:
1. Fetch authentication token
2. Download all necessary data
3. Train a prediction model
4. Generate predictions for upcoming matches

### Basic Usage
All CLI commands follow this pattern:
```bash
python -m backend.app.cli [COMMAND] [OPTIONS]
```

---

## 📖 Available Commands

### 1. 🔐 Authentication Commands

#### `get-token`
**Purpose**: Obtain and store authentication token for API access.

```bash
python -m backend.app.cli get-token
```

**What it does**:
- Fetches a new authentication token from the API
- Saves token to `output/auth_token.json`
- Required before using any data fetching commands

**Example Output**:
```
Fetching authentication token...
Token saved to: output/auth_token.json
```

---

### 2. 📊 Data Collection Commands

#### `fetch-upcoming`
**Purpose**: Fetch upcoming NBA 2K25 eSports matches for the next 24 hours.

```bash
python -m backend.app.cli fetch-upcoming
```

**What it does**:
- Retrieves all scheduled matches for the next 24 hours
- Saves data to `output/upcoming_matches.json`
- Shows summary of matches found

**Example Output**:
```
Fetching upcoming matches...
Total upcoming matches found: 37
Data saved to: output/upcoming_matches.json
```

#### `fetch-history`
**Purpose**: Fetch historical match data for analysis and training.

```bash
python -m backend.app.cli fetch-history
```

**What it does**:
- Retrieves completed match results
- Saves data to `output/match_history.json`
- Provides historical data for model training

#### `fetch-stats`
**Purpose**: Fetch detailed player statistics.

```bash
python -m backend.app.cli fetch-stats
```

**What it does**:
- Collects comprehensive player performance data
- Saves statistics to `output/player_stats.json`
- Essential for accurate predictions

---

### 3. 🤖 Machine Learning Commands

#### `train-model`
**Purpose**: Train a new prediction model with the latest data.

```bash
python -m backend.app.cli train-model
```

**What it does**:
- Creates training features from historical data
- Trains a new Random Forest model
- Automatically versions the model (v1.0.0, v1.0.1, etc.)
- Saves model to `output/models/`
- Automatically activates if it's the best performing model

**Example Output**:
```
Training new prediction model...
Model Performance:
  Accuracy: 68.3%
  Home Team MAE: 7.00
  Away Team MAE: 6.32
Model saved as: v1.0.0
Model activated as best performer
```

#### `predict`
**Purpose**: Generate predictions for upcoming matches using the active model.

```bash
python -m backend.app.cli predict
```

**What it does**:
- Loads the currently active model
- Fetches upcoming matches automatically
- Generates predictions with confidence scores
- Saves results to `output/match_predictions.csv`
- Shows detailed prediction summary

**Example Output**:
```
Generating predictions for upcoming matches...
=== Match Predictions ===
Total matches: 37
Average confidence: 0.079
High confidence matches: 0

=== Individual Predictions ===
KOBRA vs JD
  Predicted Winner: JD
  Win Probability: 17.9%
  Predicted Scores: 47.2 - 62.5
  Confidence: 0.139
```

---

### 4. 🏆 Model Management Commands

#### `list-models`
**Purpose**: View all available model versions and their performance metrics.

```bash
python -m backend.app.cli list-models
```

**What it does**:
- Lists all trained model versions
- Shows performance metrics for each model
- Indicates which model is currently active
- Displays training dates

**Example Output**:
```
=== Available Model Versions ===
Version      Active   Accuracy   Home MAE   Away MAE   Training Date
--------------------------------------------------------------------------------
v1.0.0       YES      0.683      7.00       6.32       2025-06-21T09:16:46
v1.0.1       NO       0.671      7.15       6.45       2025-06-21T08:30:22
```

#### `activate-model [VERSION]`
**Purpose**: Activate a specific model version for predictions.

```bash
python -m backend.app.cli activate-model v1.0.0
```

**Parameters**:
- `VERSION`: The model version to activate (e.g., v1.0.0)

**What it does**:
- Sets the specified model as the active model
- All future predictions will use this model
- Updates model registry in the system

**Example Output**:
```
Activating model version: v1.0.0
Model v1.0.0 is now active
```

#### `compare-models`
**Purpose**: Compare performance metrics between different model versions.

```bash
python -m backend.app.cli compare-models
```

**What it does**:
- Shows detailed comparison of all model versions
- Highlights the best performing model in each category
- Helps you decide which model to activate

**Example Output**:
```
=== Model Performance Comparison ===
Best Accuracy: v1.0.0 (68.3%)
Best Home MAE: v1.0.0 (7.00)
Best Away MAE: v1.0.0 (6.32)
Most Recent: v1.0.1 (2025-06-21T08:30:22)
```

---

### 5. 🚀 Pipeline Commands

#### `run-pipeline`
**Purpose**: Execute the complete prediction workflow with fresh data in one command.

```bash
python -m backend.app.cli run-pipeline
```

**Advanced Usage**:
```bash
# Run pipeline with new model training
python -m backend.app.cli run-pipeline --train

# Run pipeline with custom data ranges
python -m backend.app.cli run-pipeline --train --history-days 120 --training-days 90

# Force refresh authentication token
python -m backend.app.cli run-pipeline --refresh-token
```

**Parameters**:
- `--train`: Train a new model with the fresh data
- `--refresh-token`: Force refresh the authentication token
- `--history-days [DAYS]`: Days of match history to fetch (default: 90)
- `--training-days [DAYS]`: Days of history for model training (default: 60)
- `--min-matches [NUM]`: Minimum matches per player for training (default: 5)

**What it does**:
This is the **most convenient command** that automates the entire prediction workflow:

1. **🔐 Authentication**: Checks/refreshes authentication token
2. **📊 Data Collection**: Fetches latest match history, player stats, and upcoming matches
3. **🤖 Model Training**: Optionally trains a new model with fresh data (with `--train` flag)
4. **🎯 Predictions**: Generates predictions using the best available model
5. **📋 Summary**: Provides a complete summary of all operations

**Example Output**:
```
🚀 Starting Complete Prediction Pipeline...
============================================================

📝 Step 1/6: Checking authentication token...
✅ Authentication token ready

📊 Step 2/6: Fetching latest match history...
✅ Match history updated

🏀 Step 3/6: Fetching latest player statistics...
✅ Player statistics updated

🔮 Step 4/6: Fetching upcoming matches...
✅ Upcoming matches updated

🤖 Step 5/6: Training new prediction model...
✅ New model trained: v1.0.1
   Accuracy: 70.2%
   Home MAE: 6.85
   Away MAE: 6.20
🏆 New model is now active (best performer)

🎯 Step 6/6: Generating predictions...
✅ Predictions generated successfully!
📁 Results saved to: output/match_predictions.csv

============================================================
🎉 Pipeline completed successfully!

📋 Summary:
   ✅ Fresh data fetched (matches, stats, upcoming)
   ✅ New model trained and evaluated
   ✅ Predictions generated with latest data

📁 Check these files:
   • output/match_predictions.csv - Latest predictions
   • output/upcoming_matches.json - Upcoming matches data
   • logs/ - Detailed execution logs
```

**Use Cases**:
- **Daily Operations**: Run once daily for fresh predictions
- **Competition Prep**: Get the most accurate predictions before important matches
- **Model Updates**: Regularly update models with new performance data
- **Quick Setup**: Perfect for new users who want everything set up quickly

---

## 🔄 Common Workflows

### 🚀 **RECOMMENDED: One-Command Pipeline** ⭐
```bash
# Complete workflow with model training (recommended for best accuracy)
python -m backend.app.cli run-pipeline --train

# Quick daily predictions without retraining
python -m backend.app.cli run-pipeline
```

### Complete Setup Workflow (Manual)
```bash
# 1. Get authentication token
python -m backend.app.cli get-token

# 2. Fetch all necessary data
python -m backend.app.cli fetch-history
python -m backend.app.cli fetch-stats
python -m backend.app.cli fetch-upcoming

# 3. Train your first model
python -m backend.app.cli train-model

# 4. Generate predictions
python -m backend.app.cli predict
```

### Daily Prediction Workflow (Manual)
```bash
# 1. Update upcoming matches
python -m backend.app.cli fetch-upcoming

# 2. Generate fresh predictions
python -m backend.app.cli predict
```

### Model Management Workflow
```bash
# 1. Check current models
python -m backend.app.cli list-models

# 2. Train a new model with updated data
python -m backend.app.cli train-model

# 3. Compare performance
python -m backend.app.cli compare-models

# 4. Activate best model if needed
python -m backend.app.cli activate-model v1.0.1
```

### Pipeline Workflow
```bash
# 1. Run the complete pipeline for fresh predictions
python -m backend.app.cli run-pipeline

# 2. Run the pipeline with new model training
python -m backend.app.cli run-pipeline --train

# 3. Run the pipeline with custom history and training days
python -m backend.app.cli run-pipeline --train --history-days 120 --training-days 90

# 4. Force refresh the authentication token in the pipeline
python -m backend.app.cli run-pipeline --refresh-token
```

---

## 📁 Output Files

### Generated Files Location: `output/`

| File | Purpose | Generated By |
|------|---------|--------------|
| `auth_token.json` | API authentication | `get-token` |
| `upcoming_matches.json` | Upcoming match data | `fetch-upcoming` |
| `match_history.json` | Historical match results | `fetch-history` |
| `player_stats.json` | Player statistics | `fetch-stats` |
| `match_predictions.csv` | Prediction results | `predict` |
| `training_features.csv` | Training dataset | `train-model` |
| `models/*.joblib` | Trained models | `train-model` |

---

## ⚠️ Important Notes

### Authentication
- **Always run `get-token` first** before any data fetching commands
- Tokens may expire; re-run `get-token` if you get authentication errors

### Data Dependencies
- **Historical data is required** for training models (`fetch-history`)
- **Player stats are essential** for accurate predictions (`fetch-stats`)
- **Upcoming matches** must be fetched before generating predictions (`fetch-upcoming`)

### Model Versioning
- Models are automatically versioned using semantic versioning
- The system automatically activates the best-performing model
- You can manually activate any previous model version

### Performance Expectations
- **Training time**: 30-60 seconds depending on data size
- **Prediction time**: 5-15 seconds for 30-50 matches
- **Accuracy**: Typically 65-75% with proper training data

---

## 🐛 Troubleshooting

### Common Issues

#### "Authentication token not found"
**Solution**: Run `python -m backend.app.cli get-token`

#### "No training data found"
**Solution**: Run `python -m backend.app.cli fetch-history` and `python -m backend.app.cli fetch-stats`

#### "No active model found"
**Solution**: 
1. Run `python -m backend.app.cli list-models` to see available models
2. If no models exist, run `python -m backend.app.cli train-model`
3. If models exist, run `python -m backend.app.cli activate-model [VERSION]`

#### "No upcoming matches found"
**Solution**: Run `python -m backend.app.cli fetch-upcoming` to refresh match data

### Getting Help
- Check the logs in the `logs/` directory for detailed error information
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify your internet connection for API calls

---

## 🎯 Best Practices

1. **Regular Updates**: Fetch new data daily for best prediction accuracy
2. **Model Retraining**: Retrain models weekly or after significant data updates
3. **Backup Models**: Keep multiple model versions for comparison
4. **Monitor Performance**: Use `compare-models` to track model performance over time
5. **Log Monitoring**: Check logs regularly for any issues or warnings

---

## 📞 Support

For additional help or questions:
- Check the main `README.md` for system overview
- Review logs in the `logs/` directory
- Ensure you're using the latest version from the `official_backend` branch

---

*Last updated: June 21, 2025*
*Compatible with: NBA 2K25 eSports Prediction System v1.0.0+*
