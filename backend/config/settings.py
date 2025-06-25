"""
Configuration settings for the 2K Flash application.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR.parent / "output"
LOG_DIR = BASE_DIR.parent / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Data files
MATCH_HISTORY_FILE = OUTPUT_DIR / "match_history.json"
PLAYER_STATS_FILE = OUTPUT_DIR / "player_stats.json"
UPCOMING_MATCHES_FILE = OUTPUT_DIR / "upcoming_matches.json"
AUTH_TOKEN_FILE = OUTPUT_DIR / "auth_token.json"

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", os.environ.get("API_PORT", 5000)))  # Use PORT for Render compatibility
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# H2H GG League API settings
H2H_BASE_URL = "https://api-sis-stats.hudstats.com/v1"
H2H_WEBSITE_URL = "https://h2hggl.com/en/match/NB122120625"
H2H_TOKEN_LOCALSTORAGE_KEY = "sis-hudstats-token"
H2H_DEFAULT_TOURNAMENT_ID = 1

# Selenium settings - optimized for container environments
SELENIUM_HEADLESS = os.environ.get("SELENIUM_HEADLESS", "True").lower() == "true"
SELENIUM_TIMEOUT = int(os.environ.get("SELENIUM_TIMEOUT", 30))  # Reduced default for containers

# Container environment detection
IS_CONTAINER = os.environ.get("CONTAINER", "false").lower() == "true" or os.path.exists("/.dockerenv")
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true" or "render.com" in os.environ.get("RENDER_EXTERNAL_URL", "")

# Adjust settings for container environments
if IS_CONTAINER or IS_RENDER:
    SELENIUM_HEADLESS = True
    SELENIUM_TIMEOUT = min(SELENIUM_TIMEOUT, 30)  # Cap at 30 seconds for containers

# Refresh settings
REFRESH_INTERVAL = 3600  # seconds (1 hour)
MATCH_HISTORY_DAYS = 90  # days of match history to fetch
UPCOMING_MATCHES_DAYS = int(os.environ.get("UPCOMING_MATCHES_DAYS", 30))  # days of upcoming matches to fetch

# Date format settings
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
API_DATE_FORMAT = "%Y-%m-%d %H:%M"  # Format for API requests

# Timezone settings
DEFAULT_TIMEZONE = "US/Eastern"

# Supabase settings
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
