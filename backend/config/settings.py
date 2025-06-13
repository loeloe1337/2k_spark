"""
Configuration settings for the 2K Flash application.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR.parent / "output"
MODELS_DIR = BASE_DIR.parent / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data files
MATCH_HISTORY_FILE = OUTPUT_DIR / "match_history.json"
PLAYER_STATS_FILE = OUTPUT_DIR / "player_stats.json"
UPCOMING_MATCHES_FILE = OUTPUT_DIR / "upcoming_matches.json"
PREDICTIONS_FILE = OUTPUT_DIR / "upcoming_match_predictions.json"
PREDICTION_HISTORY_FILE = OUTPUT_DIR / "prediction_history.json"

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 5000))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# H2H GG League API settings
H2H_BASE_URL = "https://api-sis-stats.hudstats.com/v1"
H2H_WEBSITE_URL = "https://h2hggl.com/en/match/NB122120625"
H2H_TOKEN_LOCALSTORAGE_KEY = "sis-hudstats-token"
H2H_DEFAULT_TOURNAMENT_ID = 1

# Selenium settings
SELENIUM_HEADLESS = True
SELENIUM_TIMEOUT = 10  # seconds

# Model settings
DEFAULT_RANDOM_STATE = 42
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"
SCORE_MODEL_REGISTRY_FILE = MODELS_DIR / "score_model_registry.json"

# Refresh settings
REFRESH_INTERVAL = 3600  # seconds (1 hour)
MATCH_HISTORY_DAYS = 90  # days of match history to fetch
UPCOMING_MATCHES_DAYS = 30  # days of upcoming matches to fetch

# Date format settings
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
API_DATE_FORMAT = "%Y-%m-%d %H:%M"  # Format for API requests

# Timezone settings
DEFAULT_TIMEZONE = "US/Eastern"
