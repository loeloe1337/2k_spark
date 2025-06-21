"""
Command-line interface for the 2K Flash application.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

from config.logging_config import get_data_fetcher_logger
from utils.logging import log_execution_time, log_exceptions
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor

logger = get_data_fetcher_logger()


@log_execution_time(logger)
@log_exceptions(logger)
def fetch_token(args):
    """
    Fetch authentication token from H2H GG League.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    token_fetcher = TokenFetcher()
    token = token_fetcher.get_token(force_refresh=args.force_refresh)
    print(f"Token: {token}")


@log_execution_time(logger)
@log_exceptions(logger)
def fetch_match_history(args):
    """
    Fetch match history data.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    match_fetcher = MatchHistoryFetcher(days_back=args.days)
    matches = match_fetcher.fetch_match_history()
    print(f"Fetched {len(matches)} matches")


@log_execution_time(logger)
@log_exceptions(logger)
def fetch_upcoming_matches(args):
    """
    Fetch upcoming matches data.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    match_fetcher = UpcomingMatchesFetcher(days_forward=args.days)
    matches = match_fetcher.fetch_upcoming_matches()
    print(f"Fetched {len(matches)} upcoming matches")


@log_execution_time(logger)
@log_exceptions(logger)
def calculate_player_stats(args):
    """
    Calculate player statistics from match history.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Load match history
    match_fetcher = MatchHistoryFetcher()
    matches = match_fetcher.load_from_file()

    if not matches:
        print("No match history data found. Please fetch match history first.")
        return

    # Calculate player stats
    processor = PlayerStatsProcessor()
    player_stats = processor.calculate_player_stats(matches)
    print(f"Calculated statistics for {len(player_stats)} players")


def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description='2K Flash CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Token fetcher
    token_parser = subparsers.add_parser('fetch-token', help='Fetch authentication token')
    token_parser.add_argument('--force-refresh', action='store_true', help='Force token refresh')

    # Match history fetcher
    history_parser = subparsers.add_parser('fetch-matches', help='Fetch match history')
    history_parser.add_argument('--days', type=int, default=90, help='Number of days of history to fetch')

    # Upcoming matches fetcher
    upcoming_parser = subparsers.add_parser('fetch-upcoming', help='Fetch upcoming matches')
    upcoming_parser.add_argument('--days', type=int, default=7, help='Number of days to look ahead')

    # Player stats calculator
    stats_parser = subparsers.add_parser('calculate-stats', help='Calculate player statistics')

    args = parser.parse_args()

    if args.command == 'fetch-token':
        fetch_token(args)
    elif args.command == 'fetch-matches':
        fetch_match_history(args)
    elif args.command == 'fetch-upcoming':
        fetch_upcoming_matches(args)
    elif args.command == 'calculate-stats':
        calculate_player_stats(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
