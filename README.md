# 2K Flash - NBA Data Analytics Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The 2K Flash project is a comprehensive NBA data analytics pipeline that fetches, processes, and serves basketball match data through a modern FastAPI-based web service. The system integrates multiple data sources and provides real-time access to match history, upcoming games, player statistics, and live scores.

## Features

- 🏀 **Real-time NBA Data**: Fetch live scores and match updates
- 📊 **Player Analytics**: Comprehensive player statistics and performance metrics
- 🔮 **Match Predictions**: Historical data analysis for upcoming games
- 🚀 **FastAPI Backend**: High-performance REST API with automatic documentation
- 🌐 **CORS Support**: Ready for web frontend integration
- 📝 **Structured Logging**: Comprehensive logging for monitoring and debugging
- 🔧 **CLI Tools**: Command-line interface for data management
- 🔐 **Token Management**: Secure authentication token handling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline │    │   API Service   │
│                 │    │                 │    │                 │
│ • H2H GG League │───▶│ • Token Fetcher │───▶│ • FastAPI Server│
│ • Live Scores   │    │ • Data Fetchers │    │ • CORS Support  │
│   API           │    │ • Processors    │    │ • Auto Docs     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
│              │  File Storage   │    │   Client Apps   │
         │              │                 │    │                 │
         └──────────────▶│ • JSON Files    │    │ • Web Frontend  │
                        │ • Logs          │    │ • Mobile Apps   │
                        │ • Auth Tokens   │    │ • Third Party   │
                        └─────────────────┘    └─────────────────┘
```

## Project Structure

```
2k_spark/
├── backend/                    # Backend application code
│   ├── app/                    # Main application entry points
│   │   ├── api.py             # FastAPI server
│   │   ├── cli.py             # Command-line interface
│   │   └── fetch_data_explore.py
│   ├── config/                # Configuration management
│   │   ├── logging_config.py  # Logging configuration
│   │   └── settings.py        # Application settings
│   ├── core/                  # Core business logic
│   │   └── data/             # Data processing pipeline
│   │       ├── fetchers/     # Data fetching modules
│   │       └── processors/   # Data processing modules
│   ├── services/              # Business services
│   │   ├── data_service.py   # Data management service
│   │   └── live_scores_service.py
│   └── utils/                 # Utility functions
│       ├── logging.py        # Logging utilities
│       ├── time.py           # Time utilities
│       └── validation.py     # Data validation
├── output/                    # Generated data files
│   ├── auth_token.json       # Authentication tokens
│   ├── match_history.json    # Historical match data
│   ├── player_stats.json     # Player statistics
│   └── upcoming_matches.json # Upcoming matches
├── logs/                      # Application logs
├── Docs/                      # Documentation
└── requirements.txt           # Python dependencies
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

### API Server

Start the FastAPI server:

```bash
cd backend/app
python api.py
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### CLI Commands

The application provides several CLI commands for data management:

```bash
cd backend/app

# Fetch authentication token
python cli.py fetch-token

# Fetch match history
python cli.py fetch-matches

# Fetch upcoming matches
python cli.py fetch-upcoming

# Calculate player statistics
python cli.py calculate-stats

# Start API server
python cli.py start-server
```

## API Endpoints

### Core Endpoints

- `GET /` - Health check and API information
- `GET /matches/history` - Get historical match data
- `GET /matches/upcoming` - Get upcoming matches
- `GET /players/stats` - Get player statistics
- `GET /scores/live` - Get live scores

### Example API Usage

```python
import requests

# Get player statistics
response = requests.get("http://localhost:8000/players/stats")
player_stats = response.json()

# Get upcoming matches
response = requests.get("http://localhost:8000/matches/upcoming")
upcoming = response.json()
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

### Generated Data

- **Match History**: Complete match records with scores and metadata
- **Player Statistics**: Win rates, averages, performance metrics
- **Upcoming Matches**: Scheduled games with predictions
- **Live Scores**: Real-time match updates

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

- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] Real-time WebSocket updates
- [ ] Machine learning predictions
- [ ] Docker containerization
- [ ] Comprehensive test suite
- [ ] Performance monitoring
- [ ] Rate limiting and caching
- [ ] User authentication system

---

**Built with ❤️ for NBA analytics enthusiasts**