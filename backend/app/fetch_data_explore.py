"""
Standalone script to fetch and print match history and upcoming matches for data exploration.
"""

import sys
from pathlib import Path

# Add backend root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher

if __name__ == "__main__":
    print("Fetching match history...")
    match_fetcher = MatchHistoryFetcher()
    matches = match_fetcher.fetch_match_history(save_to_file=False)
    print(f"Fetched {len(matches)} matches. Sample:")
    for m in matches[:3]:
        print(m)
    print("\n---\n")

    print("Fetching upcoming matches...")
    upcoming_fetcher = UpcomingMatchesFetcher()
    upcoming = upcoming_fetcher.fetch_upcoming_matches(save_to_file=False)
    print(f"Fetched {len(upcoming)} upcoming matches. Sample:")
    for m in upcoming[:3]:
        print(m)
