# 2K Flash - NBA 2K25 eSports Match Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://docker.com)
[![Supabase](https://img.shields.io/badge/Supabase-Database-green.svg)](https://supabase.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

2K Flash is a comprehensive prediction system for NBA 2K25 eSports matches in the H2H GG League. The system collects real data from the H2H GG League API, processes player statistics, and uses a unified machine learning approach to predict both match winners and scores consistently, eliminating conflicting predictions.

## Features

- 🏀 **Real-time NBA Data**: Fetch live scores and match updates from H2H GG League API
- 🤖 **Unified ML Model**: Single XGBoost multi-output model predicting both home and away scores, with winner derived from score predictions
- 🔄 **Model Versioning**: Automatic model versioning with best model selection and performance tracking
- 🎯 **Consistent Predictions**: No conflicting winner vs. score predictions - all outcomes derived from the same unified model
- 🧠 **Advanced Features**: 52+ engineered features including player stats, head-to-head history, recent form, team performance
- 📊 **Player Analytics**: Comprehensive player statistics and performance metrics with team-specific analysis
- 🔮 **Smart Predictions**: Historical data analysis with ~67% winner accuracy and ~6.1 MAE for scores
- 🎛️ **Model Management**: CLI and API endpoints for listing, comparing, and activating model versions
- 🔧 **Model Optimization**: Bayesian optimization for hyperparameter tuning and feature selection
- ✅ **Real-time Validation**: Automatic prediction tracking and accuracy measurement against actual results
- 🚀 **FastAPI Backend**: High-performance REST API with automatic documentation
- 🐳 **Docker Containerized**: Full containerization for easy deployment
- 🗄️ **Supabase Database**: Persistent data storage with PostgreSQL
- 🌐 **CORS Support**: Ready for web frontend integration
- 📝 **Structured Logging**: Comprehensive logging for monitoring and debugging
- 🔧 **CLI Tools**: Command-line interface for data management
- 🔐 **Token Management**: Secure authentication token handling
- 📈 **Health Monitoring**: Built-in health checks and system monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline │    │   API Service   │
│                 │    │                 │    │                 │
│ • H2H GG League │───▶│ • Token Fetcher │───▶│ • FastAPI Server│
│ • Live Scores   │    │ • Data Fetchers │    │ • Docker Container│
│   API           │    │ • Processors    │    │ • Auto Docs     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │ Supabase Database│    │   Client Apps   │
         │              │                 │    │                 │
         └──────────────▶│ • Match History │    │ • Web Frontend  │
                        │ • Player Stats  │    │ • Mobile Apps   │
                        │ • Upcoming      │    │ • Third Party   │
                        │   Matches       │    │   Integrations  │
                        └─────────────────┘    └─────────────────┘
```

## Unified Prediction System

2K Flash uses a revolutionary **unified machine learning approach** that solves the common problem of conflicting predictions between winner and score models. Unlike traditional systems that use separate models, our approach ensures consistency:

### How It Works

1. **Single Multi-Output Model**: Uses XGBoost with `MultiOutputRegressor` to predict both home and away scores simultaneously
2. **Derived Winner**: Winner is automatically determined by comparing predicted scores (higher score wins)
3. **Consistent Metrics**: Total score, score differential, and win probabilities all stem from the same predictions
4. **No Conflicts**: Impossible to have contradicting winner vs. score predictions

### Performance Metrics

- **Winner Prediction Accuracy**: ~67% (significantly above random chance)
- **Home Score MAE**: ~6.2 points
- **Away Score MAE**: ~6.1 points  
- **Total Score MAE**: ~9.6 points
- **Training Data**: 5,000+ historical matches with 52+ features per match

### Feature Engineering

The model uses 52+ engineered features including:

- **Player Statistics**: Win rate, average score, match count, form trends
- **Head-to-Head History**: Direct matchup records and performance
- **Team Performance**: Player performance with specific NBA teams
- **Recent Form**: Rolling averages and streaks over recent matches
- **Temporal Features**: Day of week, time patterns, seasonal trends
- **Advanced Metrics**: Score differentials, clutch performance, consistency ratings

### Model Architecture

```python
# Unified prediction approach
features = extract_features(player_stats, match_data)
predictions = xgb_multi_output_model.predict(features)

home_score = predictions[0]
away_score = predictions[1]

# All metrics derived consistently
winner = "home" if home_score > away_score else "away"
total_score = home_score + away_score
score_diff = home_score - away_score
win_probability = sigmoid_transform(score_diff)
```

## Project Structure

```
2k_spark/
├── backend/                    # Backend application code
│   ├── app/                    # Main application entry points
│   │   ├── api.py             # FastAPI server with Docker health checks
│   │   ├── cli.py             # Command-line interface for unified ML pipeline
│   │   └── fetch_data_explore.py  # Data exploration utilities
│   ├── config/                # Configuration management
│   │   ├── logging_config.py  # Logging configuration
│   │   └── settings.py        # Application settings with Supabase config
│   ├── core/                  # Core business logic
│   │   └── data/             # Data processing pipeline
│   │       ├── fetchers/     # Data fetching modules
│   │       │   ├── __init__.py      # Token fetcher with environment detection
│   │       │   ├── token.py         # Standard token fetcher (Selenium)
│   │       │   ├── token_render.py  # Render-compatible token fetcher
│   │       │   ├── match_history.py # Historical match data fetcher
│   │       │   └── upcoming_matches.py # Upcoming match data fetcher
│   │       ├── processors/   # Data processing modules  
│   │       │   ├── player_stats.py          # Player statistics calculator
│   │       │   ├── match_prediction_features.py # Feature engineering (52+ features)
│   │       │   └── match_prediction_model.py    # Unified XGBoost multi-output model
│   │       └── storage.py    # Data storage utilities
│   ├── services/              # Business services
│   │   ├── data_service.py   # Data management service with DB integration
│   │   ├── enhanced_prediction_service.py # Enhanced prediction service with versioning
│   │   ├── match_prediction_service.py # Base prediction service
│   │   ├── supabase_service.py # Supabase database operations
│   │   └── live_scores_service.py # Live scores integration
│   └── utils/                 # Utility functions
│       ├── logging.py        # Logging utilities and decorators
│       ├── time.py           # Time utilities
│       └── validation.py     # Data validation
├── output/                    # Generated data files (backup/local)
│   ├── auth_token.json       # Authentication tokens
│   ├── match_history.json    # Historical match data (5,000+ matches)
│   ├── player_stats.json     # Player statistics (100+ players)
│   ├── upcoming_matches.json # Upcoming matches
│   ├── training_features.csv # Engineered features for ML training
│   └── models/               # Trained model storage
│       └── nba2k_match_predictor.joblib # Unified prediction model
├── logs/                      # Application logs
│   ├── api.log               # API server logs
│   └── data_fetcher.log      # Data pipeline logs
├── Docs/                      # Documentation
│   ├── pipeline_functionality.md
│   ├── PREDICTION_MODEL_GUIDE.md
│   ├── SETUP.md
│   ├── supabase_docker_plan.md
│   └── SUPABASE_SETUP.md
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose setup with health checks
├── .dockerignore             # Docker ignore file
├── .env                      # Environment variables (not in git)
├── .env.template             # Environment template
├── schema.sql                # Database schema for Supabase
├── migrate_data.py           # Data migration script
└── requirements.txt          # Python dependencies with ML libraries
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Chrome browser (for Selenium automation)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 2k_spark
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Chrome WebDriver**
   The application uses `webdriver-manager` to automatically manage Chrome WebDriver installation.

## Usage

### Docker (Recommended)

The easiest way to run the application is using Docker:

```bash
# Build and start the services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the services
docker-compose down
```

The API will be available at:
- **API Base URL**: `http://localhost:5000`
- **Interactive Docs**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`
- **Health Check**: `http://localhost:5000/api/health`

### Local Development

For local development without Docker:

1. **Create virtual environment and install dependencies** (see Installation section)

2. **Start the FastAPI server**:
   ```bash
   cd backend/app
   python api.py
   ```

The API will be available at:
- **API Base URL**: `http://localhost:5000`
- **Interactive Docs**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

### CLI Commands

The application provides comprehensive CLI commands for data management and machine learning:

```bash
cd backend

# Data Collection
python app/cli.py fetch-token              # Fetch authentication token
python app/cli.py fetch-history           # Fetch match history
python app/cli.py fetch-upcoming          # Fetch upcoming matches  
python app/cli.py calculate-stats         # Calculate player statistics

# Machine Learning - Enhanced Model with Versioning
python app/cli.py train-model --days 30   # Train unified prediction model with versioning
python app/cli.py predict-matches         # Generate predictions using best model
python app/cli.py list-models             # List all model versions
python app/cli.py activate-model v1.0.2   # Activate specific model version
python app/cli.py compare-models v1.0.1 v1.0.2  # Compare model versions
python app/cli.py evaluate-model --test-days 7   # Evaluate model performance
python app/cli.py feature-importance --top-n 20  # Show feature importance

# Server Management
python app/api.py                         # Start FastAPI server
```

### Model Training Examples

```bash
# Train with 60 days of data, requiring 10 matches per player
python app/cli.py train-model --days 60 --min-matches 10

# Quick training with recent data
python app/cli.py train-model --days 14 --min-matches 3

# Generate predictions after training
python app/cli.py predict-matches
```

## API Endpoints

### Core Endpoints

- `GET /api/health` - Health check and API status
- `GET /api/system/status` - System status including database connectivity
- `POST /api/system/setup-database` - Database setup verification
- `GET /api/upcoming-matches` - Get upcoming matches (live from H2H GG League)
- `GET /api/player-stats` - Get player statistics (from database)
- `GET /api/live-scores` - Get live NBA scores 

### Machine Learning Endpoints

- `POST /api/ml/train` - Train the unified prediction model with versioning
- `GET /api/ml/predictions` - Get predictions for upcoming matches (uses best model)
- `GET /api/ml/model-performance` - Get model performance metrics
- `GET /api/ml/feature-importance` - Get feature importance from trained model
- `POST /api/ml/retrain` - Retrain model with fresh data

### Model Versioning Endpoints

- `GET /api/ml/models` - List all available model versions
- `POST /api/ml/models/{version}/activate` - Activate a specific model version
- `GET /api/ml/models/compare/{version1}/{version2}` - Compare two model versions
- `GET /api/ml/models/active` - Get information about the currently active model

### Data Management Endpoints

- `POST /api/data/fetch-matches` - Fetch fresh match data
- `POST /api/data/calculate-stats` - Recalculate player statistics
- `POST /api/data/refresh-all` - Complete data refresh pipeline

### Prediction API Usage Examples

```python
import requests

# Train the model with 30 days of data
train_response = requests.post(
    "http://localhost:5000/api/ml/train",
    params={"days_back": 30, "min_matches_per_player": 5}
)
training_results = train_response.json()
print(f"Model accuracy: {training_results['metrics']['winner_accuracy']}")

# Get predictions for upcoming matches
pred_response = requests.get("http://localhost:5000/api/ml/predictions")
predictions = pred_response.json()

for match in predictions['summary']['predictions']:
    print(f"{match['home_player']} vs {match['away_player']}")
    print(f"  Winner: {match['predicted_winner']} ({match['confidence']:.1%})")
    print(f"  Score: {match['predicted_scores']['home']}-{match['predicted_scores']['away']}")
    print(f"  Total: {match['predicted_scores']['total']}")

# Get player statistics
response = requests.get("http://localhost:5000/api/player-stats")
player_stats = response.json()
```

## Configuration

The application uses centralized configuration in `backend/config/settings.py`:

- **API Settings**: Host, port, CORS origins
- **Data Sources**: H2H GG League URL, API endpoints
- **File Paths**: Output directories, log locations
- **Selenium**: Browser automation settings
- **Timezone**: Default timezone for date handling

## Data Pipeline

### Data Sources

1. **H2H GG League**: Primary source for match data and statistics
2. **Live Scores API**: Real-time score updates

### Data Processing

1. **Token Fetcher**: Manages authentication tokens with caching
2. **Match History Fetcher**: Retrieves historical match data
3. **Upcoming Matches Fetcher**: Gets scheduled matches
4. **Player Stats Processor**: Calculates comprehensive player analytics
5. **Feature Engineer**: Creates 52+ ML features from raw match data
6. **Unified Prediction Model**: XGBoost multi-output model for consistent predictions

### Machine Learning Pipeline

```
Raw Match Data → Feature Engineering → Model Training → Predictions
      ↓                    ↓                ↓             ↓
[Historical Data]  →  [52+ Features]  →  [XGBoost]  →  [Winner + Scores]
      ↓                    ↓                ↓             ↓
[Player Stats]     →  [Preprocessing]  →  [Training]  →  [Validation]
      ↓                    ↓                ↓             ↓
[Live Matches]     →  [Standardization] → [Prediction] → [API Response]
```

### Generated Data

- **Match History**: Complete match records with scores and metadata
- **Player Statistics**: Win rates, averages, performance metrics with team-specific breakdowns
- **Upcoming Matches**: Scheduled games with ML-powered predictions
- **Live Scores**: Real-time match updates
- **Feature Datasets**: Engineered features for model training (52+ per match)
- **Model Artifacts**: Trained models with performance metrics and feature importance

## Development

### Code Structure

- **Modular Design**: Separated concerns with clear interfaces
- **Error Handling**: Comprehensive exception handling and logging
- **Type Hints**: Full type annotation for better code quality
- **Documentation**: Inline documentation and docstrings

### Logging

The application uses structured logging with separate loggers:
- **API Logger**: Web server operations
- **Data Fetcher Logger**: Data pipeline operations

### Testing

To run the validation script:

```bash
python validate_stats.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation in the `Docs/` directory
- Review the API documentation at `/docs` endpoint

## Roadmap

- [x] Docker containerization
- [x] Health check endpoint
- [x] Database integration (Supabase PostgreSQL)
- [x] Real-time data fetching from H2H GG League API
- [x] System monitoring and status endpoints
- [x] **Machine learning predictions with unified model approach**
- [x] **Advanced feature engineering (52+ features)**
- [x] **Comprehensive player statistics and analytics**
- [x] **Real-time prediction validation and tracking**
- [ ] Data migration from JSON to database (script available)
- [ ] Real-time WebSocket updates
- [ ] Comprehensive test suite
- [ ] Performance monitoring
- [ ] Rate limiting and caching
- [ ] User authentication system
- [ ] Frontend dashboard
- [ ] CI/CD pipeline

## Technologies Used

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Database**: Supabase (PostgreSQL)
- **Machine Learning**: XGBoost, scikit-learn, NumPy, Pandas
- **Feature Engineering**: Custom feature extraction with 52+ engineered features
- **Model Architecture**: Multi-output regression with unified prediction approach
- **Containerization**: Docker, Docker Compose
- **Data Sources**: H2H GG League API, Live Scores API
- **Web Scraping**: Selenium, Chrome WebDriver
- **Data Processing**: Pandas, NumPy, advanced statistical calculations
- **Monitoring**: Built-in health checks, structured logging, prediction validation

---

**Built with ❤️ for NBA 2K25 eSports enthusiasts**

### 🎯 **Why This Prediction System is Different**

Unlike other prediction systems that use separate models for winner and score predictions (leading to conflicts), 2K Flash uses a **unified approach** where:
- A single XGBoost model predicts both home and away scores
- The winner is derived from comparing these scores  
- Total scores, differentials, and probabilities are all consistent
- **No more contradictory predictions!**

**Result**: ~67% winner accuracy with ~6.1 point score accuracy, all from one coherent model.

## Quick Start with Docker + Supabase

1. **Clone and build the project**:
   ```bash
   git clone <your-repo-url>
   cd 2k_spark
   docker-compose build
   ```

2. **Set up Supabase** (see [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for detailed instructions):
   - Create a Supabase project
   - Copy `.env.template` to `.env` and configure your Supabase credentials
   - Run the database schema from `schema.sql`

3. **Start the application**:
   ```bash
   docker-compose up -d
   ```

4. **Access the API**:
   - API Base URL: http://localhost:5000
   - Interactive Docs: http://localhost:5000/docs
   - Health Check: http://localhost:5000/api/health

## Model Versioning

The enhanced prediction system includes automatic model versioning and best model selection:

### CLI Commands

```bash
# Train a new model version with automatic versioning
python app/cli.py train-model --days 60 --min-matches 5

# List all available model versions
python app/cli.py list-models

# Activate a specific model version
python app/cli.py activate-model v1.0.2

# Compare two model versions
python app/cli.py compare-models v1.0.1 v1.0.2
```

### Features

- **Automatic Versioning**: Each trained model gets a unique version (v1.0.0, v1.0.1, etc.)
- **Best Model Selection**: System automatically selects the best performing model based on validation accuracy
- **Performance Tracking**: Detailed metrics stored for each model version
- **Easy Switching**: Activate any previous model version with a single command
- **Model Comparison**: Compare performance between different model versions

### API Endpoints for Model Management

- `GET /api/ml/models` - List all available model versions
- `POST /api/ml/models/{version}/activate` - Activate a specific model version
- `GET /api/ml/models/compare/{version1}/{version2}` - Compare two model versions

## Database Schema

The project uses Supabase (PostgreSQL) with a comprehensive schema that supports both core data storage and ML model versioning:

### Schema Files

- **`schema.sql`** - Complete database schema with all tables
- **`migrations/`** - Database migration history for tracking changes
  - `000_original_schema.sql` - Original base schema (historical reference)
  - `001_prediction_system_enhancement.sql` - Model versioning tables
  - `002_core_tables_player_enhancement.sql` - Player column additions

### Core Tables

- `matches` - Historical match data with player information
- `player_stats` - Player statistics and performance metrics
- `upcoming_matches` - Scheduled matches for prediction

### ML Model Tables

- `model_registry` - Model versions and metadata
- `match_predictions` - Prediction results and tracking
- `feature_importance` - Feature importance per model version
- `model_performance` - Model performance over time