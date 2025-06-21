"""
Data fetchers package initialization.
"""

import os
import importlib.util

# Check if we're in a deployment environment (Render)
if os.environ.get("RENDER", "0") == "1" or os.environ.get("RAILWAY_ENVIRONMENT", "") or os.environ.get("VERCEL", ""):
    # Use the Render-compatible token fetcher
    from .token_render import TokenFetcher
else:
    # Use the standard token fetcher with Selenium
    from .token import TokenFetcher

# Export the TokenFetcher class
__all__ = ["TokenFetcher"]
