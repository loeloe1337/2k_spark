"""
Data fetchers package initialization.
"""

# Always use the standard token fetcher with Selenium
from .token import TokenFetcher

# Export the TokenFetcher class
__all__ = ["TokenFetcher"]
