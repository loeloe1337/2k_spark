# Component Documentation - Backend

## Overview

The 2K Spark backend is built with a modular, layered architecture that separates concerns and promotes maintainability. This document provides detailed information about each major component and how they work together.

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Application   │────▶│    Services     │────▶│      Core       │
│     Layer       │     │     Layer       │     │    Business     │
│   (API/CLI)     │     │                 │     │     Logic       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Configuration  │     │     Utils       │     │   Data Layer    │
│   & Logging     │     │   & Helpers     │     │   (Storage)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📁 Directory Structure

```
backend/
├── app/                    # Application entry points
├── config/                 # Configuration management
├── core/                   # Core business logic
│   ├── data/              # Data access and processing
│   ├── models/            # Machine learning models
│   └── optimization/      # Model optimization
├── services/              # Service layer
├── utils/                 # Common utilities
└── scripts/               # Maintenance scripts
```

## 🎯 Application Layer (`app/`)

### API Server (`api.py`)

The main Flask application providing RESTful endpoints.

**Key Features**:
- CORS-enabled for frontend integration
- Comprehensive error handling
- Execution time logging
- Background refresh capabilities

**Main Endpoints**:
```python
@app.route('/api/predictions', methods=['GET'])
@app.route('/api/score-predictions', methods=['GET'])
@app.route('/api/stats', methods=['GET'])
@app.route('/api/refresh', methods=['POST'])
```

**Usage**:
```bash
# Start the API server
python app/api.py

# The server runs on http://localhost:5000
```

### Command Line Interface (`cli.py`)

Comprehensive CLI for system administration and operations.

**Available Commands**:
```bash
# Data operations
python app/cli.py fetch-token
python app/cli.py fetch-history
python app/cli.py fetch-upcoming
python app/cli.py calculate-stats

# Model operations
python app/cli.py train-winner-model
python app/cli.py train-score-model
python app/cli.py optimize-winner-model
python app/cli.py optimize-score-model

# System maintenance
python app/cli.py clean-model-registry
python app/cli.py refresh
```

### Model Optimization Scripts

Standalone scripts for Bayesian optimization:

- `optimize_winner_model.py`: Optimizes classification models
- `optimize_score_model.py`: Optimizes regression models

## ⚙️ Configuration Layer (`config/`)

### Settings (`settings.py`)

Central configuration management with environment variable support.

**Key Configuration Areas**:
```python
# API Configuration
API_HOST = "localhost"
API_PORT = 5000
CORS_ORIGINS = ["http://localhost:3000"]

# H2H GG League API
H2H_BASE_URL = "https://api-sis-stats.hudstats.com/v1"
H2H_DEFAULT_TOURNAMENT_ID = 1

# Model Settings
DEFAULT_RANDOM_STATE = 42
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# File Paths
PREDICTIONS_FILE = OUTPUT_DIR / "upcoming_match_predictions.json"
PLAYER_STATS_FILE = OUTPUT_DIR / "player_stats.json"
```

### Logging Configuration (`logging_config.py`)

Structured logging setup with component-specific loggers.

**Logger Types**:
- API logger with request/response tracking
- Data fetcher logger for external API calls
- Model training logger for ML operations
- General application logger

**Usage**:
```python
from config.logging_config import get_api_logger
logger = get_api_logger()
logger.info("API request processed")
```

## 🧠 Core Business Logic (`core/`)

### Data Layer (`core/data/`)

#### Fetchers (`core/data/fetchers/`)

Specialized classes for retrieving data from external sources:

**Token Fetcher** (`token.py`):
```python
fetcher = TokenFetcher()
token = fetcher.get_token()  # Handles browser automation
headers = fetcher.get_auth_headers()
```

**Match History Fetcher** (`match_history.py`):
```python
fetcher = MatchHistoryFetcher()
matches = fetcher.fetch_match_history(days_back=90)
```

**Upcoming Matches Fetcher** (`upcoming_matches.py`):
```python
fetcher = UpcomingMatchesFetcher()
upcoming = fetcher.fetch_upcoming_matches(days_forward=30)
```

#### Processors (`core/data/processors/`)

**Player Statistics Processor** (`player_stats.py`):
Calculates comprehensive player metrics from match history.

```python
processor = PlayerStatsProcessor()
stats = processor.calculate_stats(match_history)

# Generated statistics include:
# - Win/loss rates
# - Average scores
# - Performance trends
# - Consistency metrics
# - Recent form analysis
```

#### Storage (`storage.py`)

Unified interface for data persistence:

```python
from core.data.storage import save_json, load_json

# Save data
save_json(data, "output/predictions.json")

# Load data
predictions = load_json("output/predictions.json")
```

### Models Layer (`core/models/`)

#### Base Model (`base.py`)

Abstract base class defining the interface for all prediction models:

```python
class BasePredictionModel(ABC):
    @abstractmethod
    def train(self, features, targets):
        pass
    
    @abstractmethod
    def predict(self, features):
        pass
    
    @abstractmethod
    def evaluate(self, features, targets):
        pass
```

#### Winner Prediction Model (`winner_prediction.py`)

Classification model for predicting match winners:

```python
model = WinnerPredictionModel()
metrics = model.train(features, winners)
predictions = model.predict(upcoming_features)

# Features used:
# - Player win rates
# - Average scores
# - Head-to-head history
# - Recent form
# - Performance variance
```

#### Score Prediction Model (`score_prediction.py`)

Regression model for predicting match scores:

```python
model = ScorePredictionModel()
metrics = model.train(features, scores)
score_predictions = model.predict(upcoming_features)

# Predicts:
# - Home team score
# - Away team score
# - Total score
# - Confidence intervals
```

#### Feature Engineering (`feature_engineering.py`)

Advanced feature creation and transformation:

```python
engineer = FeatureEngineer()
features = engineer.create_features(player_stats, match_data)

# Generated features:
# - Rolling averages
# - Performance momentum
# - Variance metrics
# - Head-to-head statistics
# - Recent form indicators
```

#### Model Registry (`registry.py`)

Centralized model management with versioning:

```python
registry = ModelRegistry()

# Register a new model
registry.register_model(model, metrics, metadata)

# Get the best model
best_model = registry.get_best_model()

# List all models
models = registry.list_models()
```

### Optimization Layer (`core/optimization/`)

#### Bayesian Optimizer (`bayesian_optimizer.py`)

Advanced hyperparameter optimization using Gaussian processes:

```python
optimizer = BayesianOptimizer(
    model_class=WinnerPredictionModel,
    param_space={
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20)
    }
)

best_params, best_score = optimizer.optimize(
    features, targets, n_trials=100
)
```

#### Tuner (`tuner.py`)

High-level interface for model optimization:

```python
tuner = ModelTuner()
optimized_model = tuner.tune_model(
    model_type='winner_prediction',
    data=training_data,
    n_trials=50
)
```

## 🔧 Service Layer (`services/`)

### Data Service (`data_service.py`)

Orchestrates data operations across multiple sources:

```python
service = DataService()

# Refresh all data
service.refresh_all_data()

# Get processed player stats
stats = service.get_player_statistics()

# Get upcoming matches
matches = service.get_upcoming_matches()
```

### Prediction Service (`prediction_service.py`)

Manages prediction generation and formatting:

```python
service = PredictionService()

# Generate predictions for upcoming matches
predictions = service.generate_predictions()

# Get prediction history
history = service.get_prediction_history()

# Validate predictions against results
validation = service.validate_predictions()
```

### Refresh Service (`refresh_service.py`)

Coordinates system-wide data updates:

```python
service = RefreshService()

# Full system refresh
success = service.refresh_system()

# Check refresh status
status = service.get_refresh_status()
```

### Live Scores Service (`live_scores_service.py`)

Handles real-time match data:

```python
service = LiveScoresService()

# Get live match data
live_matches = service.get_live_matches()

# Get live matches with predictions
enriched_data = service.get_live_matches_with_predictions()
```

## 🛠️ Utilities (`utils/`)

### Logging Utilities (`logging.py`)

Decorators and helpers for consistent logging:

```python
from utils.logging import log_execution_time, log_exceptions

@log_execution_time(logger)
@log_exceptions(logger)
def my_function():
    # Function automatically logs execution time and exceptions
    pass
```

### Time Utilities (`time.py`)

Time zone handling and date operations:

```python
from utils.time import convert_timezone, format_datetime

# Convert to specific timezone
local_time = convert_timezone(utc_time, 'America/New_York')

# Format for display
formatted = format_datetime(datetime_obj, 'human')
```

### Validation Utilities (`validation.py`)

Input validation and data sanitization:

```python
from utils.validation import validate_match_data, sanitize_input

# Validate match data structure
is_valid = validate_match_data(match_dict)

# Sanitize user input
clean_input = sanitize_input(user_data)
```

## 🔄 Data Flow

### 1. Data Collection Flow
```
H2H API → Token Fetcher → Match History Fetcher → Raw Match Data
                      → Upcoming Matches Fetcher → Future Matches
```

### 2. Data Processing Flow
```
Raw Match Data → Player Stats Processor → Feature Engineering → ML Features
```

### 3. Model Training Flow
```
ML Features → Model Training → Model Evaluation → Model Registry
```

### 4. Prediction Flow
```
Upcoming Matches → Feature Engineering → Trained Models → Predictions → API
```

## 🚀 Getting Started with Components

### Adding a New Data Fetcher

1. Create a new fetcher class in `core/data/fetchers/`:
```python
class NewDataFetcher:
    def __init__(self, config):
        self.config = config
    
    def fetch_data(self):
        # Implementation here
        pass
```

2. Add configuration in `config/settings.py`
3. Integrate with the data service
4. Add CLI commands if needed

### Adding a New Model

1. Inherit from `BasePredictionModel`:
```python
class NewPredictionModel(BasePredictionModel):
    def train(self, features, targets):
        # Training implementation
        pass
    
    def predict(self, features):
        # Prediction implementation
        pass
```

2. Add to model registry
3. Create optimization parameters
4. Add CLI commands for training

### Adding a New API Endpoint

1. Add route in `app/api.py`:
```python
@app.route('/api/new-endpoint', methods=['GET'])
@log_execution_time(logger)
@log_exceptions(logger)
def new_endpoint():
    # Implementation here
    return jsonify(result)
```

2. Add corresponding service method
3. Update API documentation
4. Add frontend integration

## 🔍 Monitoring and Debugging

### Logging Levels
- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about system operation
- **WARNING**: Something unexpected happened but system continues
- **ERROR**: Serious problem that prevented function execution

### Log Files
- `logs/api.log`: API requests and responses
- `logs/data_fetcher.log`: Data collection operations
- `logs/model_tuning.log`: Model training and optimization
- `logs/prediction_refresh.log`: System refresh operations

### Performance Monitoring
All major functions are decorated with `@log_execution_time` to track performance and identify bottlenecks.

---

**Last Updated**: June 17, 2025
