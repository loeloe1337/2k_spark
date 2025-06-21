# 2K Flash Pipeline Functionality Documentation

## Overview

The 2K Flash project is a comprehensive NBA data analytics pipeline that fetches, processes, and serves basketball match data through a modern FastAPI-based web service. The system integrates multiple data sources and provides real-time access to match history, upcoming games, player statistics, and live scores.

## Architecture Overview

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

## Core Components

### 1. Configuration Management (`config/`)

#### Settings (`settings.py`)
- **Purpose**: Centralized configuration for the entire application
- **Key Features**:
  - Environment-based configuration (development/production)
  - File path management for data storage
  - API endpoints and authentication settings
  - Selenium browser automation settings
  - Timezone and date format configurations

#### Logging Configuration (`logging_config.py`)
- **Purpose**: Structured logging across all components
- **Features**:
  - Separate loggers for API and data fetching operations
  - File-based logging with rotation
  - Configurable log levels

### 2. Data Pipeline (`core/data/`)

#### Token Management (`fetchers/token.py`)
- **Purpose**: Handles authentication with H2H GG League API
- **Functionality**:
  - Uses Selenium WebDriver to extract authentication tokens from browser localStorage
  - Supports both headless and GUI modes
  - Token caching and refresh mechanisms
  - Fallback to Render-compatible token fetcher for deployment

#### Match History Fetcher (`fetchers/match_history.py`)
- **Purpose**: Retrieves historical match data
- **Process**:
  1. Obtains authentication token
  2. Constructs API requests with date ranges
  3. Fetches match data from H2H GG League API
  4. Validates and stores data in JSON format
  5. Implements retry logic for failed requests

#### Upcoming Matches Fetcher (`fetchers/upcoming_matches.py`)
- **Purpose**: Retrieves scheduled future matches
- **Process**:
  1. Calculates future date ranges
  2. Fetches upcoming match schedules
  3. Processes and validates match data
  4. Stores results for API consumption

#### Player Statistics Processor (`processors/player_stats.py`)
- **Purpose**: Calculates comprehensive player performance metrics
- **Calculations**:
  - Points per game averages
  - Shooting percentages (FG%, 3P%, FT%)
  - Rebounds, assists, steals, blocks
  - Advanced metrics (efficiency ratings)
  - Trend analysis over time periods

### 3. Services Layer (`services/`)

#### Data Service (`data_service.py`)
- **Purpose**: Orchestrates all data operations
- **Responsibilities**:
  - Coordinates between different fetchers
  - Manages data refresh cycles
  - Provides unified interface for data access
  - Handles error recovery and fallback strategies

#### Live Scores Service (`live_scores_service.py`)
- **Purpose**: Fetches real-time NBA game data
- **Features**:
  - Connects to H2H HudStats live API
  - Processes live game states (pre-game, in-progress, final)
  - Standardizes data format for consistent API responses
  - Real-time score updates and game status

### 4. API Layer (`app/api.py`)

#### FastAPI Application
- **Framework**: Modern FastAPI with automatic OpenAPI documentation
- **Features**:
  - CORS middleware for cross-origin requests
  - Automatic request/response validation
  - Interactive API documentation at `/docs`
  - JSON schema generation

#### API Endpoints

##### `/api/upcoming-matches`
- **Purpose**: Returns scheduled future games
- **Data Flow**:
  1. Checks for cached data in `upcoming_matches.json`
  2. If no data exists, triggers fresh data fetch
  3. Returns standardized match objects with team info, dates, venues

##### `/api/player-stats`
- **Purpose**: Provides player performance analytics
- **Data Flow**:
  1. Attempts to load pre-calculated stats from `player_stats.json`
  2. If unavailable, calculates stats from match history
  3. Falls back to fetching fresh match data if needed
  4. Returns comprehensive player metrics

##### `/api/live-scores`
- **Purpose**: Real-time game scores and status
- **Data Flow**:
  1. Directly queries live scores API
  2. Processes and standardizes response format
  3. Returns current game states and scores

### 5. Command Line Interface (`app/cli.py`)

#### Available Commands
- **`fetch-token`**: Manually refresh authentication tokens
- **`fetch-history --days N`**: Retrieve match history for N days
- **`fetch-upcoming --days N`**: Get upcoming matches for N days
- **`calculate-stats`**: Process player statistics from match data

### 6. Utilities (`utils/`)

#### Logging Utilities (`logging.py`)
- **Decorators**:
  - `@log_execution_time`: Measures and logs function execution time
  - `@log_exceptions`: Catches and logs exceptions with stack traces

#### Time Utilities (`time.py`)
- **Functions**:
  - Timezone-aware datetime handling
  - Date range calculations for API requests
  - Format conversions between different date standards

#### Validation Utilities (`validation.py`)
- **Purpose**: Data integrity and format validation
- **Features**:
  - Match data structure validation
  - API response format checking
  - Data type and range validations

## Data Flow Pipeline

### 1. Authentication Flow
```
Website Visit → Selenium WebDriver → Extract Token → Cache Token → API Requests
```

### 2. Data Fetching Flow
```
CLI Command → Token Validation → API Request → Data Processing → File Storage → API Serving
```

### 3. API Request Flow
```
Client Request → FastAPI Router → Data Service → File Check → Fresh Fetch (if needed) → Response
```

## File Storage Structure

```
2k_spark/
├── output/                    # Data storage
│   ├── match_history.json     # Historical match data
│   ├── player_stats.json      # Calculated player statistics
│   └── upcoming_matches.json  # Scheduled future matches
├── logs/                      # Application logs
│   ├── api.log               # API server logs
│   └── data_fetcher.log      # Data pipeline logs
└── auth_token.json           # Cached authentication token
```

## Error Handling and Resilience

### 1. Network Failures
- Automatic retry mechanisms with exponential backoff
- Graceful degradation to cached data
- Comprehensive error logging

### 2. Authentication Issues
- Automatic token refresh
- Fallback authentication methods
- Manual token override capabilities

### 3. Data Validation
- Schema validation for all API responses
- Data integrity checks before storage
- Malformed data filtering and logging

## Performance Optimizations

### 1. Caching Strategy
- File-based caching for expensive API calls
- Token caching to minimize authentication overhead
- Configurable refresh intervals

### 2. Asynchronous Operations
- FastAPI's async capabilities for concurrent requests
- Non-blocking data fetching operations
- Efficient resource utilization

### 3. Data Processing
- Batch processing for large datasets
- Incremental updates for player statistics
- Memory-efficient data structures

## Deployment Considerations

### 1. Environment Compatibility
- Selenium WebDriver configuration for different environments
- Render-specific adaptations for cloud deployment
- Environment variable configuration

### 2. Dependencies
- Core libraries: FastAPI, Selenium, NumPy, Pandas
- Browser automation: Chrome/Chromium WebDriver
- HTTP client: Requests library

### 3. Monitoring
- Structured logging for operational visibility
- Performance metrics tracking
- Error rate monitoring

## Security Features

### 1. Token Management
- Secure token storage and transmission
- Token expiration handling
- Authentication state management

### 2. API Security
- CORS configuration for controlled access
- Input validation and sanitization
- Rate limiting considerations

### 3. Data Protection
- Secure file storage permissions
- Log data sanitization
- Environment-based configuration

## Future Enhancement Opportunities

### 1. Real-time Features
- WebSocket connections for live updates
- Push notifications for game events
- Real-time player performance tracking

### 2. Advanced Analytics
- Machine learning predictions
- Advanced statistical modeling
- Trend analysis and forecasting

### 3. Scalability Improvements
- Database integration for larger datasets
- Microservices architecture
- Horizontal scaling capabilities

## Conclusion

The 2K Flash pipeline represents a robust, scalable solution for NBA data analytics. Its modular architecture, comprehensive error handling, and modern API design make it suitable for both development and production environments. The system successfully bridges multiple data sources while providing a unified, reliable interface for basketball data consumption.