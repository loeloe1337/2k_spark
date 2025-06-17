# 2K Flash - NBA 2K25 eSports Match Prediction System

2K Flash is a comprehensive prediction system for NBA 2K25 eSports matches in the H2H GG League. The system collects real data from the H2H GG League API, processes player statistics, and uses advanced machine learning models to predict match winners and scores with high accuracy.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd 2k_spark

# Backend setup
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r ../requirements.txt
python app/api.py

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to see the application!

📖 **Detailed Setup Guide**: [docs/development/getting-started.md](docs/development/getting-started.md)

## ✨ Features

- **🎯 Intelligent Predictions**: Advanced ML models for match winners and scores
- **📊 Real-time Analytics**: Live player statistics and performance metrics
- **🔬 Advanced ML**: Bayesian optimization and ensemble methods
- **🌐 Modern Web Interface**: React/Next.js with responsive design
- **⚡ Fast API**: RESTful backend with comprehensive endpoints

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React/Next.js │────▶│   Flask API     │────▶│  ML Models &    │
│    Frontend     │     │    Backend      │     │  Data Pipeline  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Vercel        │     │    Render       │     │  H2H GG League  │
│  (Frontend)     │     │   (Backend)     │     │      API        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions` | GET | Get winner predictions for upcoming matches |
| `/api/score-predictions` | GET | Get score predictions for upcoming matches |
| `/api/stats` | GET | Get prediction accuracy and performance metrics |
| `/api/player-stats` | GET | Get comprehensive player statistics |
| `/api/refresh` | POST | Trigger data refresh and prediction update |

📖 **Complete API Documentation**: [docs/development/api-docs.md](docs/development/api-docs.md)

## 🛠️ Technology Stack

### Backend
- **Python 3.10+** with Flask
- **scikit-learn** for machine learning
- **Selenium** for data collection
- **Bayesian optimization** for model tuning

### Frontend
- **Next.js 13+** with App Router
- **TypeScript** for type safety
- **Tailwind CSS + Shadcn UI** for styling
- **Chart.js** for data visualization

### Infrastructure
- **Vercel** (Frontend deployment)
- **Render** (Backend deployment)
- **GitHub Actions** (CI/CD)

## 📁 Project Structure

```
2k_spark/
├── docs/                      # 📚 Comprehensive documentation
│   ├── development/           # Developer guides and API docs
│   ├── architecture/          # System architecture overview
│   └── components/            # Component-specific documentation
├── backend/                   # 🐍 Python API backend
│   ├── app/                  # Application entry points
│   ├── core/                 # Business logic & ML models
│   ├── services/             # Service layer
│   └── config/               # Configuration & settings
├── frontend/                  # ⚛️ Next.js React frontend
│   ├── src/app/              # App router pages
│   ├── src/components/       # React components
│   └── src/hooks/            # Custom React hooks
├── H2HGGL-Player-Photos/     # 🖼️ Player profile images
├── logs/                     # 📝 Application logs
├── models/                   # 🤖 Trained ML models
└── output/                   # 📊 Generated data & predictions
```

## 📚 Documentation

Our documentation is organized for easy navigation:

### 🚀 For Developers
- **[Quick Start Guide](docs/development/getting-started.md)** - Get up and running in 5 minutes
- **[API Documentation](docs/development/api-docs.md)** - Complete API reference
- **[Testing Guide](docs/development/testing.md)** - Testing strategies and examples

### 🏗️ Architecture
- **[Project Overview](docs/architecture/project-overview.md)** - System architecture and design
- **[Backend Components](docs/components/backend/README.md)** - Detailed backend documentation

### 🔧 Operations
- **[Deployment Guide](docs/deployment/)** - Production deployment instructions
- **[Maintenance Guide](docs/operations/)** - System maintenance and troubleshooting

## 🔄 CLI Usage

The backend provides a comprehensive CLI for data management and model operations:

```bash
# Data Collection
python app/cli.py fetch-token        # Fetch authentication token
python app/cli.py fetch-history      # Fetch match history
python app/cli.py fetch-upcoming     # Fetch upcoming matches
python app/cli.py calculate-stats    # Calculate player statistics

# Model Training
python app/cli.py train-winner-model # Train winner prediction model
python app/cli.py train-score-model  # Train score prediction model
python app/cli.py optimize-winner-model   # Optimize with Bayesian optimization
python app/cli.py optimize-score-model    # Optimize score prediction model

# System Maintenance
python app/cli.py refresh            # Full system refresh
python app/cli.py clean-model-registry    # Clean model registry
```

## 🚀 Deployment

The application is configured for easy deployment:

### Frontend (Vercel)
1. Connect your GitHub repository to Vercel
2. Set root directory to `frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy!

### Backend (Render)
1. Connect your GitHub repository to Render
2. Set root directory to `backend`
3. Add environment variable: `CORS_ORIGINS`
4. Deploy!

📖 **Detailed Deployment Guide**: [docs/deployment/deployment.md](docs/deployment/deployment.md)

## 🤝 Contributing

1. **Read the Documentation**: Start with [docs/development/getting-started.md](docs/development/getting-started.md)
2. **Follow Testing Guidelines**: See [docs/development/testing.md](docs/development/testing.md)
3. **Check the Architecture**: Review [docs/architecture/project-overview.md](docs/architecture/project-overview.md)
4. **Submit Pull Requests**: With tests and documentation updates

## 📈 Current Status

### ✅ Completed
- Complete backend architecture with advanced ML models
- React frontend with responsive design  
- Real-time data integration with H2H GG League API
- Comprehensive logging and monitoring
- Bayesian optimization for model tuning

### 🚧 In Progress
- Testing infrastructure implementation
- Performance optimization
- UI/UX improvements

### 📋 Planned
- Database migration from file-based storage
- Enhanced security and authentication
- Advanced data visualizations
- Mobile app development

## 📊 Performance Metrics

- **Prediction Accuracy**: ~74% for winner predictions
- **Score Prediction MAE**: ~3.2 points average error
- **API Response Time**: <500ms for all endpoints
- **System Uptime**: 99.9% target availability

## 🆘 Need Help?

- **Quick Issues**: Check [docs/operations/maintenance.md](docs/operations/maintenance.md)
- **API Questions**: See [docs/development/api-docs.md](docs/development/api-docs.md)
- **Setup Problems**: Follow [docs/development/getting-started.md](docs/development/getting-started.md)
- **Architecture Questions**: Review [docs/components/backend/README.md](docs/components/backend/README.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the NBA 2K25 eSports community**  
**Last Updated**: June 17, 2025

## Contents

### Files

Various application files that may include legacy implementations, experimental features, or alternative application structures.

### Subdirectories

Subdirectories containing organized legacy or alternative application components.

## Key Components

*   **Legacy Code:** Previous versions of application components maintained for reference or rollback purposes.
*   **Experimental Features:** Development and testing implementations for new functionality before integration into main codebase.
*   **Alternative Architecture:** Different structural approaches to application organization and implementation.
*   **Backup Components:** Preserved implementations that may be needed for specific scenarios or compatibility.
*   **Development Tools:** Utility scripts and tools used during development but not part of the main application.
*   **Configuration Files:** Alternative or legacy configuration setups for different deployment scenarios.

## Usage

This directory may be used for development reference, testing alternative implementations, or maintaining backward compatibility. Components from this directory should be carefully evaluated before integration into the main application structure.

## Dependencies

Dependencies vary based on the specific contents and may include:
*   **Legacy framework versions** for compatibility with older implementations
*   **Development tools** for testing and experimentation
*   **Alternative libraries** for different implementation approaches
*   **Backup configurations** for various deployment scenarios

---

## backend/app

# Folder: app

## Folder Purpose

The app directory serves as the main application entry point for the 2k_spark backend system. It contains the primary executable scripts and interfaces that users and systems interact with directly. This includes the RESTful API server, command-line interface for administrative tasks, model optimization scripts, and system refresh utilities. These components provide the primary access points for both programmatic and manual interaction with the prediction system.

## Contents

### Files

*   **api.py:** Flask-based RESTful API server providing endpoints for predictions, statistics, and data refresh operations.
*   **auth_token.json:** Local authentication token storage for H2H GG League API access within the app context.
*   **clean_model_registry.py:** Utility script for cleaning and maintaining the model registry by removing problematic or outdated models.
*   **cli.py:** Comprehensive command-line interface providing administrative commands for data fetching, model training, and system maintenance.
*   **optimize_score_model.py:** Standalone script for optimizing score prediction models using Bayesian optimization techniques.
*   **optimize_winner_model.py:** Standalone script for optimizing winner prediction models with hyperparameter tuning.
*   **refresh_script.py:** Automated script for refreshing predictions and updating system data on a scheduled basis.
*   **refresh_status.json:** Status tracking file for monitoring the refresh process and maintaining system state.

### Subdirectories

None - this directory contains only executable files and configuration.

## Key Components

*   **API Server (api.py):** Flask application with CORS support providing RESTful endpoints for frontend communication and external integrations.
*   **CLI Interface (cli.py):** Comprehensive command-line tool with subcommands for token fetching, data collection, model training, and system maintenance.
*   **Model Optimization Scripts:** Dedicated scripts for Bayesian optimization of both winner and score prediction models with advanced hyperparameter tuning.
*   **Refresh System:** Automated data refresh mechanism with status tracking for maintaining up-to-date predictions and statistics.
*   **Registry Maintenance:** Tools for cleaning and maintaining the model registry to ensure optimal performance and storage efficiency.

## Usage

Start the API server with `python api.py` for web service functionality. Use the CLI with `python cli.py [command]` for administrative tasks such as `fetch-token`, `train-winner-model`, or `optimize-score-model`. Run optimization scripts directly for model tuning, and use the refresh script for automated system updates.

## Dependencies

*   **Flask and Flask-CORS** for RESTful API implementation with cross-origin support
*   **argparse** for command-line interface parsing and subcommand handling
*   **Core modules** from the parent backend directory for data processing and model operations
*   **Configuration modules** for settings, logging, and environment management
*   **Service layer modules** for business logic and data operations

---

## backend/config

# Folder: config

## Folder Purpose

The config directory centralizes all configuration management for the 2k_spark backend system. It provides a unified approach to handling application settings, environment variables, logging configuration, and system parameters. This directory ensures consistent configuration across all backend components and facilitates easy deployment and environment-specific customization without code changes.

## Contents

### Files

*   **logging_config.py:** Comprehensive logging configuration module providing structured logging setup for different components with file rotation and console output.
*   **settings.py:** Central configuration file containing all application settings including API endpoints, file paths, model parameters, and environment-specific variables.

### Subdirectories

None - this directory contains only configuration modules.

## Key Components

*   **Application Settings:** Centralized configuration for API host/port, CORS origins, file paths, and directory structures.
*   **H2H GG League Integration:** API endpoints, authentication settings, and tournament configuration for external data source integration.
*   **Selenium Configuration:** Web scraping settings including headless mode, timeouts, and browser automation parameters.
*   **Model Configuration:** Machine learning model settings including random states, registry file locations, and training parameters.
*   **Logging System:** Multi-logger configuration with file rotation, different log levels, and component-specific log files.
*   **Refresh Settings:** Automated system refresh intervals and data retention policies for maintaining current predictions.

## Usage

Import settings from `settings.py` using `from config.settings import SETTING_NAME` and configure loggers using functions from `logging_config.py`. Environment variables can override default settings for deployment-specific customization. The configuration supports both development and production environments through environment variable overrides.

## Dependencies

*   **os and pathlib** for environment variable access and path management
*   **logging and logging.handlers** for advanced logging configuration with file rotation
*   **Environment variables** for deployment-specific configuration overrides
*   **Parent directory structure** for relative path calculations and file organization

---

## backend/core

# Folder: core

## Folder Purpose

The core directory contains the fundamental business logic and algorithmic components of the 2k_spark prediction system. It houses the essential data processing pipeline, machine learning models, and optimization frameworks that power the prediction capabilities. This directory represents the heart of the application, implementing sophisticated data fetching, feature engineering, model training, and Bayesian optimization techniques for accurate eSports match predictions.

## Contents

### Files

None - this directory contains only subdirectories with specialized functionality.

### Subdirectories

*   **data/:** Data access layer containing fetchers for external API integration, processors for statistical calculations, and storage management utilities.
*   **models/:** Machine learning model implementations including base classes, feature engineering, model registry, and specialized prediction models for winners and scores.
*   **optimization/:** Advanced optimization framework implementing Bayesian optimization and hyperparameter tuning for model performance enhancement.

## Key Components

*   **Data Pipeline:** Comprehensive data fetching and processing system for H2H GG League API integration with robust error handling and data validation.
*   **Machine Learning Framework:** Sophisticated prediction models using ensemble methods, cross-validation, and feature selection for accurate match outcome predictions.
*   **Feature Engineering:** Advanced statistical feature creation including player performance metrics, team dynamics, and historical trend analysis.
*   **Model Registry:** Centralized model management system with versioning, metadata tracking, and performance monitoring capabilities.
*   **Bayesian Optimization:** State-of-the-art hyperparameter tuning using Gaussian processes for optimal model performance.
*   **Storage Abstraction:** Unified data storage interface supporting multiple backends and ensuring data consistency across the system.

## Usage

The core modules are imported and used by higher-level application components. Data fetchers collect information from external sources, processors transform raw data into features, models generate predictions, and optimizers enhance model performance. The modular design allows for easy testing, maintenance, and extension of core functionality.

## Dependencies

*   **scikit-learn** for machine learning algorithms and model evaluation
*   **pandas and numpy** for data manipulation and numerical computations
*   **Bayesian optimization libraries** for hyperparameter tuning
*   **requests** for HTTP API communication
*   **Selenium** for web scraping and browser automation
*   **joblib** for model serialization and parallel processing

---

## backend/core/data

# Folder: data

## Folder Purpose

The data directory implements the complete data access layer for the 2k_spark prediction system. It provides a comprehensive pipeline for collecting, processing, and storing eSports data from external sources, particularly the H2H GG League API. This directory contains specialized fetchers for different data types, processors for statistical calculations, and storage management utilities that ensure reliable and efficient data operations throughout the system.

## Contents

### Files

*   **storage.py:** Data storage abstraction layer providing unified interfaces for reading, writing, and managing data files across different storage backends.

### Subdirectories

*   **fetchers/:** Collection of specialized data fetchers for retrieving information from external APIs, including authentication token management and match data collection.
*   **processors/:** Data processing modules that transform raw data into meaningful statistics and features for machine learning model consumption.

## Key Components

*   **Data Fetching Framework:** Robust API integration system with error handling, rate limiting, and retry mechanisms for reliable external data collection.
*   **Authentication Management:** Secure token handling and refresh mechanisms for maintaining API access to H2H GG League services.
*   **Statistical Processing:** Advanced player and team statistics calculation including performance metrics, trends, and comparative analysis.
*   **Storage Abstraction:** Unified data persistence layer supporting JSON files, databases, and other storage backends with consistent interfaces.
*   **Data Validation:** Comprehensive validation and sanitization of incoming data to ensure quality and consistency.
*   **Caching System:** Intelligent caching mechanisms to reduce API calls and improve system performance.

## Usage

Fetchers are used to collect data from external sources using methods like `fetch_match_history()` and `fetch_upcoming_matches()`. Processors transform this raw data into statistical features using functions like `calculate_player_stats()`. The storage module provides consistent data persistence across the application with methods for saving and loading various data formats.

## Dependencies

*   **requests** for HTTP API communication and data fetching
*   **Selenium WebDriver** for browser automation and web scraping
*   **pandas** for data manipulation and statistical calculations
*   **json** for data serialization and file operations
*   **datetime and pytz** for time zone handling and date operations
*   **pathlib** for cross-platform file path management

---

## backend/core/data/fetchers

# Folder: fetchers

## Folder Purpose

The fetchers directory contains specialized data collection modules responsible for retrieving information from external sources, primarily the H2H GG League API. These components implement robust API integration with authentication management, error handling, and data validation. Each fetcher is designed for specific data types and provides reliable, efficient access to real-time eSports data essential for the prediction system's accuracy.

## Contents

### Files

*   **__init__.py:** Package initialization file exposing the main fetcher classes and providing convenient imports for other modules.
*   **match_history.py:** Fetcher for retrieving historical match data including results, player performances, and game statistics from the H2H GG League.
*   **token.py:** Authentication token fetcher implementing secure token retrieval and management for API access using Selenium browser automation.
*   **token_render.py:** Alternative token fetcher optimized for server environments and deployment platforms like Render with headless browser support.
*   **upcoming_matches.py:** Fetcher for collecting upcoming match schedules, team information, and tournament data for prediction generation.

### Subdirectories

None - this directory contains only fetcher implementation files.

## Key Components

*   **Authentication System:** Secure token management with automatic refresh capabilities and multiple fetching strategies for different deployment environments.
*   **Match History Fetcher:** Comprehensive historical data collection with filtering, pagination, and data validation for training machine learning models.
*   **Upcoming Matches Fetcher:** Real-time schedule and team data collection for generating current predictions and maintaining up-to-date match information.
*   **Error Handling:** Robust exception management with retry mechanisms, timeout handling, and graceful degradation for unreliable network conditions.
*   **Data Validation:** Input sanitization and output validation ensuring data quality and consistency across all fetched information.
*   **Browser Automation:** Selenium-based web scraping for accessing data not available through direct API endpoints.

## Usage

Fetchers are instantiated and used to collect specific data types: `TokenFetcher().get_token()` for authentication, `MatchHistoryFetcher().fetch_matches()` for historical data, and `UpcomingMatchesFetcher().fetch_upcoming()` for future matches. Each fetcher handles its own error management and data formatting.

## Dependencies

*   **requests** for HTTP API communication and RESTful data access
*   **Selenium WebDriver** for browser automation and JavaScript-heavy page interaction
*   **BeautifulSoup** for HTML parsing and data extraction
*   **json and datetime** for data serialization and time handling
*   **logging** for comprehensive error tracking and debugging
*   **time and random** for retry mechanisms and rate limiting

---

## backend/core/data/processors

# Folder: processors

## Folder Purpose

The processors directory contains data transformation and statistical calculation modules that convert raw eSports data into meaningful features for machine learning models. These components implement sophisticated statistical analysis, player performance metrics, and feature engineering techniques that transform basic match data into rich, predictive features used by the prediction models to generate accurate forecasts.

## Contents

### Files

*   **player_stats.py:** Comprehensive player statistics processor calculating advanced performance metrics, trends, and comparative analysis from historical match data.

### Subdirectories

None - this directory contains only processing implementation files.

## Key Components

*   **Player Performance Analysis:** Advanced statistical calculations including win rates, average scores, performance trends, and consistency metrics for individual players.
*   **Team Dynamics Processing:** Analysis of team composition effects, player synergies, and collaborative performance patterns.
*   **Historical Trend Analysis:** Time-series analysis of player and team performance with trend identification and momentum calculations.
*   **Feature Engineering:** Creation of derived features including rolling averages, performance ratios, and comparative metrics for machine learning consumption.
*   **Statistical Aggregation:** Multi-level aggregation of match data across different time periods, opponents, and game conditions.
*   **Data Normalization:** Standardization and normalization of statistics to ensure fair comparison across different players and time periods.

## Usage

Processors are used to transform raw match data into statistical features: `PlayerStatsProcessor().calculate_stats(match_data)` processes historical matches and generates comprehensive player statistics. The processed data is then used by machine learning models for training and prediction generation.

## Dependencies

*   **pandas** for data manipulation, aggregation, and statistical calculations
*   **numpy** for numerical computations and array operations
*   **datetime** for time-based analysis and trend calculations
*   **statistics** for advanced statistical measures and distributions
*   **collections** for data structure management and counting operations
*   **logging** for process tracking and debugging information

---

## backend/core/models

# Folder: models

## Folder Purpose

The models directory contains the complete machine learning framework for the 2k_spark prediction system. It implements sophisticated prediction models for both match winners and scores, utilizing advanced techniques including ensemble methods, cross-validation, feature selection, and Bayesian optimization. This directory represents the core intelligence of the system, transforming processed eSports data into accurate predictions through state-of-the-art machine learning algorithms.

## Contents

### Files

*   **base.py:** Abstract base class defining the common interface and shared functionality for all prediction models in the system.
*   **feature_engineering.py:** Advanced feature engineering module creating derived features, statistical transformations, and predictive variables from raw player and match data.
*   **registry.py:** Model registry system providing versioning, metadata management, performance tracking, and model lifecycle management for both winner and score prediction models.
*   **score_prediction.py:** Specialized regression model for predicting match scores using ensemble methods and advanced feature selection techniques.
*   **winner_prediction.py:** Classification model for predicting match winners using cross-validated ensemble algorithms with probability calibration.

### Subdirectories

None - this directory contains only model implementation files.

## Key Components

*   **Ensemble Learning Framework:** Advanced ensemble methods combining multiple algorithms for improved prediction accuracy and robustness.
*   **Cross-Validation System:** Comprehensive model validation using stratified k-fold cross-validation with performance metrics tracking.
*   **Feature Selection:** Automated feature selection using statistical tests, recursive feature elimination, and importance-based filtering.
*   **Model Registry:** Centralized model management with versioning, performance comparison, and automatic model selection based on validation metrics.
*   **Hyperparameter Optimization:** Integration with Bayesian optimization for automated hyperparameter tuning and model performance enhancement.
*   **Prediction Calibration:** Probability calibration for winner predictions and confidence interval estimation for score predictions.

## Usage

Models are trained using `WinnerPredictionModel().train(features, targets)` and `ScorePredictionModel().train(features, scores)`. Predictions are generated with `model.predict(new_features)` and models are managed through the registry system for versioning and performance tracking.

## Dependencies

*   **scikit-learn** for machine learning algorithms, ensemble methods, and model evaluation
*   **pandas and numpy** for data manipulation and numerical computations
*   **joblib** for model serialization and parallel processing
*   **datetime** for timestamp management and model versioning
*   **json** for metadata storage and configuration management
*   **logging** for training progress tracking and error handling

---

## backend/core/optimization

# Folder: optimization

## Folder Purpose

The optimization directory implements advanced hyperparameter tuning and model optimization techniques for the 2k_spark prediction system. It provides a sophisticated Bayesian optimization framework that automatically searches for optimal model configurations, improving prediction accuracy through intelligent hyperparameter exploration. This directory contains the algorithmic components that enhance model performance beyond manual tuning capabilities.

## Contents

### Files

*   **__init__.py:** Package initialization file exposing optimization classes and providing convenient imports for the optimization framework.
*   **bayesian_optimizer.py:** Core Bayesian optimization implementation using Gaussian processes for intelligent hyperparameter search and model performance optimization.
*   **tuner.py:** High-level tuning interface that coordinates optimization processes and provides simplified access to complex optimization algorithms.

### Subdirectories

None - this directory contains only optimization implementation files.

## Key Components

*   **Bayesian Optimization Engine:** Advanced optimization using Gaussian processes and acquisition functions for efficient hyperparameter space exploration.
*   **Hyperparameter Search Space:** Configurable parameter spaces with support for continuous, discrete, and categorical hyperparameters.
*   **Acquisition Functions:** Multiple acquisition strategies including Expected Improvement, Upper Confidence Bound, and Probability of Improvement.
*   **Cross-Validation Integration:** Seamless integration with model validation for robust performance estimation during optimization.
*   **Optimization History:** Comprehensive tracking of optimization progress with convergence analysis and performance visualization.
*   **Multi-Objective Optimization:** Support for optimizing multiple metrics simultaneously with Pareto frontier analysis.

## Usage

Optimization is performed using `BayesianOptimizer().optimize(model, parameter_space, objective_function)` for automated hyperparameter tuning. The tuner provides high-level interfaces like `Tuner().tune_model(model_type, data)` for simplified optimization workflows.

## Dependencies

*   **scikit-optimize (skopt)** for Bayesian optimization algorithms and acquisition functions
*   **scipy** for statistical distributions and optimization utilities
*   **numpy** for numerical computations and array operations
*   **matplotlib** for optimization progress visualization and convergence plots
*   **joblib** for parallel optimization and model evaluation
*   **logging** for optimization progress tracking and debugging

---

## backend

# Folder: backend

## Folder Purpose

The backend directory contains the complete server-side implementation of the 2k_spark prediction system. It serves as the core engine for data collection, processing, machine learning model training, and API services. This directory houses the RESTful API server, data fetchers for H2H GG League integration, advanced machine learning models with Bayesian optimization, and various utility services for maintaining the prediction system's accuracy and reliability.

## Contents

### Files

*   **Procfile:** Heroku/Render deployment configuration specifying the application startup command.
*   **auth_token.json:** Authentication token storage for H2H GG League API access.
*   **render.yaml:** Render deployment configuration with service specifications and environment settings.
*   **runtime.txt:** Python runtime version specification for deployment platforms.

### Subdirectories

*   **.venv/:** Python virtual environment containing isolated dependencies for the backend application.
*   **app/:** Application entry points including API server, CLI interface, and model optimization scripts.
*   **config/:** Configuration management modules for logging, settings, and environment variables.
*   **core/:** Core business logic containing data processing, machine learning models, and optimization algorithms.
*   **scripts/:** Utility scripts for data cleanup, prediction generation, and system maintenance tasks.
*   **services/:** Service layer implementing business logic for data operations, predictions, and system refresh.
*   **utils/:** Common utility functions for logging, time handling, and data validation.

## Key Components

*   **API Server:** RESTful API implementation providing endpoints for predictions, statistics, and data refresh operations.
*   **Machine Learning Pipeline:** Advanced prediction models with feature engineering, cross-validation, and Bayesian optimization.
*   **Data Processing Engine:** Comprehensive data fetchers and processors for H2H GG League API integration.
*   **Model Registry:** Centralized system for managing trained models with versioning and metadata tracking.
*   **Optimization Framework:** Bayesian optimization system for hyperparameter tuning and model performance enhancement.
*   **Service Architecture:** Modular service layer separating concerns for data, predictions, and system maintenance.

## Usage

The backend can be operated through multiple interfaces: the API server for web requests, the CLI for administrative tasks, and individual scripts for specific operations. Start the API server with `python app/api.py` or use the CLI with `python app/cli.py` followed by specific commands for data fetching, model training, or optimization.

## Dependencies

*   **Python 3.10+** as the primary runtime environment
*   **FastAPI/Flask** for RESTful API implementation
*   **scikit-learn** for machine learning model development
*   **pandas and numpy** for data processing and numerical computations
*   **Bayesian optimization libraries** for hyperparameter tuning
*   **Selenium** for web scraping and data collection
*   **H2H GG League API** for real-time eSports data access

---

## backend/scripts

# Folder: scripts

## Folder Purpose

The scripts directory contains utility and maintenance scripts that support the operation and maintenance of the 2k_spark prediction system. These standalone scripts handle data cleanup, prediction generation, validation, and system maintenance tasks that are typically run periodically or as needed for system health. They provide essential housekeeping functionality and data integrity operations.

## Contents

### Files

*   **cleanup_duplicate_history.py:** Data cleanup script that identifies and removes duplicate entries from match history data to maintain data integrity.
*   **clear_mock_predictions.py:** Utility script for removing test or mock prediction data from the system to ensure clean production data.
*   **generate_prediction_history.py:** Script for generating historical prediction records and maintaining prediction tracking over time.
*   **generate_predictions.py:** Standalone prediction generation script that creates predictions for upcoming matches using trained models.
*   **validate_predictions.py:** Validation script that checks prediction accuracy against actual match results and generates performance reports.

### Subdirectories

None - this directory contains only utility script files.

## Key Components

*   **Data Cleanup Tools:** Scripts for maintaining data quality by removing duplicates, invalid entries, and test data from production datasets.
*   **Prediction Generation:** Standalone prediction creation tools that can be run independently or scheduled for automated prediction updates.
*   **Validation Framework:** Comprehensive prediction accuracy validation with performance metrics calculation and reporting.
*   **History Management:** Tools for maintaining prediction history and tracking model performance over time.
*   **System Maintenance:** Utility scripts for routine system maintenance, data cleanup, and integrity checks.
*   **Batch Processing:** Scripts designed for batch operations and scheduled execution for system automation.

## Usage

Scripts are executed directly from the command line: `python cleanup_duplicate_history.py` for data cleanup, `python generate_predictions.py` for prediction creation, and `python validate_predictions.py` for accuracy checking. These scripts can be scheduled using cron jobs or task schedulers for automated maintenance.

## Dependencies

*   **pandas** for data manipulation and analysis operations
*   **json** for data file reading and writing
*   **datetime** for time-based operations and date handling
*   **pathlib** for cross-platform file path management
*   **logging** for script execution tracking and error reporting
*   **Core modules** from the parent backend directory for model and data access

---

## backend/services

# Folder: services

## Folder Purpose

The services directory implements the service layer of the 2k_spark prediction system, providing high-level business logic and orchestrating interactions between different system components. These services abstract complex operations into clean, reusable interfaces that coordinate data fetching, prediction generation, validation, and system refresh operations. The service layer ensures separation of concerns and provides a stable API for the application layer.

## Contents

### Files

*   **data_service.py:** Core data service orchestrating data collection, processing, and storage operations across multiple data sources and processors.
*   **live_scores_service.py:** Real-time score tracking service that monitors ongoing matches and updates live game statistics.
*   **prediction_service.py:** Central prediction service coordinating model execution, prediction generation, and result formatting for API consumption.
*   **prediction_validation_service.py:** Validation service that compares predictions against actual results and calculates accuracy metrics and performance statistics.
*   **refresh_service.py:** System refresh service that coordinates periodic data updates, model retraining, and prediction regeneration.

### Subdirectories

None - this directory contains only service implementation files.

## Key Components

*   **Data Orchestration:** Coordinated data collection and processing workflows that manage the entire data pipeline from fetching to storage.
*   **Prediction Pipeline:** End-to-end prediction generation process including feature preparation, model execution, and result formatting.
*   **Live Data Integration:** Real-time data processing for ongoing matches with live score updates and dynamic prediction adjustments.
*   **Validation Framework:** Comprehensive prediction accuracy tracking with statistical analysis and performance reporting.
*   **System Automation:** Automated refresh and maintenance processes that keep the system current with minimal manual intervention.
*   **Error Handling:** Robust error management and recovery mechanisms ensuring system reliability and graceful degradation.

## Usage

Services are used by the API layer and CLI: `DataService().refresh_all_data()` for data updates, `PredictionService().generate_predictions()` for creating predictions, and `RefreshService().refresh_system()` for complete system updates. Services provide clean interfaces that hide implementation complexity.

## Dependencies

*   **Core modules** from the backend for data access, models, and optimization
*   **datetime and pytz** for time zone handling and scheduling
*   **logging** for service operation tracking and error reporting
*   **json** for data serialization and configuration management
*   **threading** for concurrent operations and background processing
*   **pathlib** for file system operations and path management

---

## backend/utils

# Folder: utils

## Folder Purpose

The utils directory contains common utility functions and helper modules that provide shared functionality across the entire 2k_spark backend system. These utilities handle cross-cutting concerns such as logging, time management, data validation, and other common operations that are used throughout the application. This directory promotes code reuse and maintains consistency in how common tasks are performed across different components.

## Contents

### Files

*   **logging.py:** Logging utilities providing decorators and helper functions for execution time tracking, exception logging, and performance monitoring.
*   **time.py:** Time and date utility functions for timezone handling, date formatting, and time-based calculations used throughout the system.
*   **validation.py:** Data validation utilities providing input sanitization, data type checking, and consistency validation for API inputs and data processing.

### Subdirectories

None - this directory contains only utility implementation files.

## Key Components

*   **Logging Decorators:** Function decorators for automatic execution time logging, exception handling, and performance monitoring across all system components.
*   **Time Zone Management:** Comprehensive timezone handling utilities for consistent time representation across different geographical locations and data sources.
*   **Data Validation Framework:** Input validation and sanitization functions ensuring data quality and preventing invalid data from entering the system.
*   **Performance Monitoring:** Utilities for tracking function execution times, memory usage, and system performance metrics.
*   **Error Handling Helpers:** Common error handling patterns and exception management utilities used throughout the application.
*   **Format Converters:** Data format conversion utilities for consistent data representation across different system components.

## Usage

Utilities are imported and used across the system: `@log_execution_time(logger)` decorator for performance tracking, `validate_input_data(data)` for data validation, and `convert_timezone(datetime, timezone)` for time handling. These utilities provide consistent behavior across all system components.

## Dependencies

*   **logging** for comprehensive logging functionality and handler management
*   **datetime and pytz** for time zone handling and date operations
*   **functools** for decorator implementation and function wrapping
*   **json and typing** for data validation and type checking
*   **inspect** for function introspection and metadata extraction
*   **time** for performance measurement and timing operations

---

## frontend/src

# Folder: src

## Folder Purpose

The src directory contains the complete source code for the 2k_spark React application built with Next.js 13+ app router. It implements a modern, component-based architecture with TypeScript, providing a comprehensive user interface for viewing predictions, live matches, player statistics, and historical data. The directory follows Next.js best practices with organized components, custom hooks, context providers, and page-based routing.

## Contents

### Files

None - this directory contains only subdirectories with organized source code.

### Subdirectories

*   **app/:** Next.js app router pages and layouts implementing the main application routes and global styling.
*   **components/:** Reusable React components organized by feature area including predictions, scores, live matches, and UI elements.
*   **contexts/:** React context providers for global state management and cross-component data sharing.
*   **hooks/:** Custom React hooks for data fetching, state management, and reusable component logic.

## Key Components

*   **App Router Architecture:** Modern Next.js 13+ app directory structure with file-based routing and layout components.
*   **Component Library:** Comprehensive collection of reusable components built with Shadcn UI and Tailwind CSS.
*   **State Management:** React context and custom hooks for managing application state and API data.
*   **TypeScript Integration:** Full TypeScript implementation with strict type checking and interface definitions.
*   **Responsive Design:** Mobile-first components with responsive layouts and adaptive user interfaces.
*   **Dark Mode Support:** Complete dark mode theming with system preference detection and manual toggle.

## Usage

The src directory follows Next.js conventions where pages are defined in the app directory, components are organized by feature, hooks provide reusable logic, and contexts manage global state. Components are imported and used throughout the application with consistent props and TypeScript interfaces.

## Dependencies

*   **React 18+** for component-based UI development with hooks and context
*   **Next.js 13+** for app router, server components, and routing
*   **TypeScript** for type safety and enhanced development experience
*   **Tailwind CSS** for utility-first styling and responsive design
*   **Shadcn UI** for pre-built component library and design system
*   **Lucide React** for consistent iconography and visual elements

---

## Summary

This master documentation provides a comprehensive overview of the entire 2K Spark project structure. The project is a sophisticated eSports prediction system that combines:

- **Frontend**: Modern React/Next.js application with TypeScript
- **Backend**: Python-based API with machine learning capabilities
- **Data Pipeline**: Robust data fetching and processing from H2H GG League API
- **Machine Learning**: Advanced prediction models with Bayesian optimization
- **Infrastructure**: GitHub Actions CI/CD, deployment configurations, and utility scripts

Each component is designed with modularity, scalability, and maintainability in mind, creating a comprehensive system for eSports match prediction and analysis.

---

# Project Directory Structure Overview

## Project Directory Structure

```
2k_spark/
├── .github/
│   ├── readme.md
│   └── workflows/
│       ├── ci.yml
│       └── deploy-frontend.yml
├── .gitignore
├── .venv/
├── PREDICTION_TRACKING_README.md
├── Project-Overview/
│   └── Project_Readme.md
├── README.md
├── TOKEN_FETCHER_README.md
├── app/
│   ├── optimize_winner_model.py
│   └── readme.md
├── auth_token.json
├── backend/
│   ├── .venv/
│   ├── Procfile
│   ├── app/
│   │   ├── api.py
│   │   ├── auth_token.json
│   │   ├── clean_model_registry.py
│   │   ├── cli.py
│   │   ├── optimize_score_model.py
│   │   ├── optimize_winner_model.py
│   │   ├── readme.md
│   │   ├── refresh_script.py
│   │   └── refresh_status.json
│   ├── auth_token.json
│   ├── config/
│   │   ├── logging_config.py
│   │   ├── readme.md
│   │   └── settings.py
│   ├── core/
│   │   ├── data/
│   │   ├── models/
│   │   ├── optimization/
│   │   └── readme.md
│   ├── readme.md
│   ├── render.yaml
│   ├── runtime.txt
│   ├── scripts/
│   │   ├── cleanup_duplicate_history.py
│   │   ├── clear_mock_predictions.py
│   │   ├── generate_prediction_history.py
│   │   ├── generate_predictions.py
│   │   ├── readme.md
│   │   └── validate_predictions.py
│   ├── services/
│   │   ├── data_service.py
│   │   ├── live_scores_service.py
│   │   ├── prediction_service.py
│   │   ├── prediction_validation_service.py
│   │   ├── readme.md
│   │   └── refresh_service.py
│   └── utils/
│       ├── logging.py
│       ├── readme.md
│       ├── time.py
│       └── validation.py
├── frontend/
│   ├── .gitignore
│   ├── README.md
│   ├── build-script.js
│   ├── components.json
│   ├── eslint.config.mjs
│   ├── jsconfig.json
│   ├── next.config.js
│   ├── next.config.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.mjs
│   ├── public/
│   │   ├── .nojekyll
│   │   ├── H2HGGL-Player-Photos/
│   │   ├── file.svg
│   │   ├── globe.svg
│   │   ├── next.svg
│   │   ├── vercel.svg
│   │   └── window.svg
│   ├── resolve-imports.js
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── contexts/
│   │   ├── hooks/
│   │   └── readme.md
│   ├── tsconfig.json
│   └── vercel.json
├── logs/
│   ├── api.log
│   ├── data_fetcher.log
│   ├── data_fetcher.log.1
│   ├── model_tuning.log
│   ├── prediction_refresh.log
│   ├── readme.md
│   └── score_model_training.log
├── models/
│   └── [machine learning model files]
├── output/
│   └── [generated output files]
├── requirements.txt

```

## Directory Overview

### Root Level
- **Configuration Files**: `.gitignore`, `package.json`, `requirements.txt`, `vercel.json`
- **Authentication**: `auth_token.json`, `fetch_auth_token.py`
- **Documentation**: `README.md`, `PREDICTION_TRACKING_README.md`, `TOKEN_FETCHER_README.md`

### Main Directories

#### `.github/`
Contains GitHub Actions workflows and repository configuration

#### `H2HGGL-Player-Photos/`
Player profile images in WebP format for the NBA 2K25 eSports players

#### `Project-Overview/`
Project documentation and overview files

#### `app/`
Legacy application files and model optimization scripts

#### `backend/`
Python backend application with the following structure:
- `app/`: Main application files including API and CLI
- `config/`: Configuration and settings
- `core/`: Core business logic (data, models, optimization)
- `scripts/`: Utility and maintenance scripts
- `services/`: Business services layer
- `utils/`: Common utilities and helpers

#### `frontend/`
Next.js React frontend application:
- `src/app/`: Next.js app router pages
- `src/components/`: Reusable React components
- `src/contexts/`: React context providers
- `src/hooks/`: Custom React hooks
- `public/`: Static assets

#### `logs/`
Application log files for monitoring and debugging



#### `models/`
Trained machine learning models and artifacts

#### `output/`
Generated files, processed data, and prediction results

## Technology Stack

- **Backend**: Python, Flask/FastAPI
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Machine Learning**: scikit-learn, pandas, numpy
- **Deployment**: Vercel (frontend), Render (backend)
- **CI/CD**: GitHub Actions
- **Data Storage**: JSON files, model artifacts