# 2K Flash - NBA Data Analytics Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://docker.com)
[![Supabase](https://img.shields.io/badge/Supabase-Database-green.svg)](https://supabase.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The 2K Flash project is a comprehensive NBA data analytics pipeline that fetches, processes, and serves basketball match data through a modern FastAPI-based web service. The system integrates multiple data sources, provides real-time access to match data, and includes full database persistence with Supabase.

## Features

- 🏀 **Real-time NBA Data**: Fetch live scores and match updates from H2H GG League API
- 📊 **Player Analytics**: Comprehensive player statistics and performance metrics
- 🔮 **Match Predictions**: Historical data analysis for upcoming games
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

## Project Structure

```
2k_spark/
├── backend/                    # Backend application code
│   ├── app/                    # Main application entry points
│   │   ├── api.py             # FastAPI server with Docker health checks
│   │   ├── cli.py             # Command-line interface
│   │   └── fetch_data_explore.py
│   ├── config/                # Configuration management
│   │   ├── logging_config.py  # Logging configuration
│   │   └── settings.py        # Application settings with Supabase config
│   ├── core/                  # Core business logic
│   │   └── data/             # Data processing pipeline
│   │       ├── fetchers/     # Data fetching modules
│   │       └── processors/   # Data processing modules
│   ├── services/              # Business services
│   │   ├── data_service.py   # Data management service with DB integration
│   │   ├── supabase_service.py # Supabase database operations
│   │   └── live_scores_service.py
│   └── utils/                 # Utility functions
│       ├── logging.py        # Logging utilities
│       ├── time.py           # Time utilities
│       └── validation.py     # Data validation
├── output/                    # Generated data files (backup/local)
│   ├── auth_token.json       # Authentication tokens
│   ├── match_history.json    # Historical match data
│   ├── player_stats.json     # Player statistics
│   └── upcoming_matches.json # Upcoming matches
├── logs/                      # Application logs
├── Docs/                      # Documentation
│   ├── pipeline_functionality.md
│   └── supabase_docker_plan.md
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose setup
├── .dockerignore             # Docker ignore file
├── .env                      # Environment variables (not in git)
├── .env.template             # Environment template
├── schema.sql                # Database schema
├── migrate_data.py           # Data migration script
├── SUPABASE_SETUP.md         # Supabase setup guide
└── requirements.txt          # Python dependencies with Supabase
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

- `GET /api/health` - Health check and API status
- `GET /api/system/status` - System status including database connectivity
- `POST /api/system/setup-database` - Database setup verification
- `GET /api/upcoming-matches` - Get upcoming matches (live from H2H GG League)
- `GET /api/player-stats` - Get player statistics (from database)
- `GET /api/live-scores` - Get live NBA scores 
- `GET /api/live-scores` - Get live NBA scores

### Example API Usage

```python
import requests

# Get player statistics
response = requests.get("http://localhost:5000/api/player-stats")
player_stats = response.json()

# Get upcoming matches
response = requests.get("http://localhost:5000/api/upcoming-matches")
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

- [x] Docker containerization
- [x] Health check endpoint
- [x] Database integration (Supabase PostgreSQL)
- [x] Real-time data fetching from H2H GG League API
- [x] System monitoring and status endpoints
- [ ] Data migration from JSON to database (script available)
- [ ] Real-time WebSocket updates
- [ ] Machine learning predictions
- [ ] Comprehensive test suite
- [ ] Performance monitoring
- [ ] Rate limiting and caching
- [ ] User authentication system
- [ ] Frontend dashboard
- [ ] CI/CD pipeline

## Technologies Used

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Database**: Supabase (PostgreSQL)
- **Containerization**: Docker, Docker Compose
- **Data Sources**: H2H GG League API, Live Scores API
- **Web Scraping**: Selenium, Chrome WebDriver
- **Data Processing**: Pandas, NumPy
- **Monitoring**: Built-in health checks, structured logging

---

**Built with ❤️ for NBA analytics enthusiasts**

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