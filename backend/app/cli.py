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
from config.settings import DEFAULT_RANDOM_STATE
from utils.logging import log_execution_time, log_exceptions
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor
from core.models.winner_prediction import WinnerPredictionModel
from core.models.score_prediction import ScorePredictionModel
from core.models.registry import ModelRegistry, ScoreModelRegistry
from services.refresh_service import refresh_predictions
from services.prediction_service import PredictionService
from services.prediction_validation_service import validate_predictions
from scripts.validate_predictions import main as validate_predictions_script
from scripts.clear_mock_predictions import main as clear_mock_predictions

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


@log_execution_time(logger)
@log_exceptions(logger)
def train_winner_model(args):
    """
    Train winner prediction model.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Load match history
    match_fetcher = MatchHistoryFetcher()
    matches = match_fetcher.load_from_file()

    if not matches:
        print("No match history data found. Please fetch match history first.")
        return

    # Load player stats
    processor = PlayerStatsProcessor()
    player_stats = processor.load_from_file()

    if not player_stats:
        print("No player statistics found. Please calculate player statistics first.")
        return

    # Train model
    model = WinnerPredictionModel()
    model.train(player_stats, matches)

    # Save model
    model_path, info_path = model.save()

    # Register model
    registry = ModelRegistry()
    registry.register_model(model.get_info())

    print(f"Trained and saved winner prediction model with ID: {model.model_id}")
    print(f"Model accuracy: {model.model_info.get('accuracy', 0):.4f}")


@log_execution_time(logger)
@log_exceptions(logger)
def train_score_model(args):
    """
    Train score prediction model.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Load match history
    match_fetcher = MatchHistoryFetcher()
    matches = match_fetcher.load_from_file()

    if not matches:
        print("No match history data found. Please fetch match history first.")
        return

    # Load player stats
    processor = PlayerStatsProcessor()
    player_stats = processor.load_from_file()

    if not player_stats:
        print("No player statistics found. Please calculate player statistics first.")
        return

    # Train model
    model = ScorePredictionModel()
    model.train(player_stats, matches)

    # Save model
    model_path, info_path = model.save()

    # Register model
    registry = ScoreModelRegistry()
    registry.register_model(model.get_info())

    print(f"Trained and saved score prediction model with ID: {model.model_id}")
    print(f"Model total score MAE: {model.model_info.get('total_score_mae', 0):.4f}")


@log_execution_time(logger)
@log_exceptions(logger)
def list_models(args):
    """
    List trained models.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    if args.type == 'winner':
        registry = ModelRegistry()
        models = registry.list_models()
        print(f"Winner prediction models ({len(models)}):")
        for model in models:
            print(f"  - ID: {model.get('model_id')}, Accuracy: {model.get('accuracy', 0):.4f}")

        best_model = registry.get_best_model_info()
        if best_model:
            print(f"Best model: {best_model.get('model_id')} (Accuracy: {best_model.get('accuracy', 0):.4f})")

    elif args.type == 'score':
        registry = ScoreModelRegistry()
        models = registry.list_models()
        print(f"Score prediction models ({len(models)}):")
        for model in models:
            print(f"  - ID: {model.get('model_id')}, MAE: {model.get('total_score_mae', 0):.4f}")

        best_model = registry.get_best_model_info()
        if best_model:
            print(f"Best model: {best_model.get('model_id')} (MAE: {best_model.get('total_score_mae', 0):.4f})")


@log_execution_time(logger)
@log_exceptions(logger)
def optimize_score_model(args):
    """
    Optimize score prediction model using Bayesian optimization.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    script_path = os.path.join(os.path.dirname(__file__), "optimize_score_model.py")

    # Run the script as a subprocess
    cmd = [
        sys.executable,
        script_path,
        "--n-trials", str(args.n_trials),
        "--test-size", str(args.test_size),
        "--random-state", str(args.random_state)
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


@log_execution_time(logger)
@log_exceptions(logger)
def optimize_winner_model(args):
    """
    Optimize winner prediction model using Bayesian optimization.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    script_path = os.path.join(os.path.dirname(__file__), "optimize_winner_model.py")

    # Run the script as a subprocess
    cmd = [
        sys.executable,
        script_path,
        "--n-trials", str(args.n_trials),
        "--test-size", str(args.test_size),
        "--random-state", str(args.random_state)
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


@log_execution_time(logger)
@log_exceptions(logger)
def clean_model_registry(args):
    """
    Clean model registry by removing problematic models.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    script_path = os.path.join(os.path.dirname(__file__), "clean_model_registry.py")

    # Run the script as a subprocess
    cmd = [
        sys.executable,
        script_path,
        "--min-samples", str(args.min_samples)
    ]

    if args.keep_files:
        cmd.append("--keep-files")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def refresh_data(args):
    """
    Refresh all data and predictions.
    """
    print("Refreshing data and predictions...")
    success = refresh_predictions()
    if success:
        print("✅ Data refresh completed successfully")
    else:
        print("❌ Data refresh failed")
        sys.exit(1)


def validate_prediction_results(args):
    """
    Validate predictions against completed match results.
    """
    print("Validating predictions against completed matches...")
    success = validate_predictions()
    if success:
        print("✅ Prediction validation completed successfully")
    else:
        print("❌ Prediction validation failed")
        sys.exit(1)


def clear_mock_prediction_data(args):
    """
    Clear mock prediction data and prepare for real prediction tracking.
    """
    clear_mock_predictions()


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
    history_parser = subparsers.add_parser('fetch-history', help='Fetch match history')
    history_parser.add_argument('--days', type=int, default=90, help='Number of days of history to fetch')

    # Upcoming matches fetcher
    upcoming_parser = subparsers.add_parser('fetch-upcoming', help='Fetch upcoming matches')
    upcoming_parser.add_argument('--days', type=int, default=7, help='Number of days to look ahead')

    # Player stats calculator
    stats_parser = subparsers.add_parser('calculate-stats', help='Calculate player statistics')

    # Winner model trainer
    winner_model_parser = subparsers.add_parser('train-winner-model', help='Train winner prediction model')

    # Score model trainer
    score_model_parser = subparsers.add_parser('train-score-model', help='Train score prediction model')

    # Model lister
    list_models_parser = subparsers.add_parser('list-models', help='List trained models')
    list_models_parser.add_argument('--type', choices=['winner', 'score'], default='winner', help='Model type')

    # Score model optimizer
    optimize_score_parser = subparsers.add_parser('optimize-score-model', help='Optimize score prediction model')
    optimize_score_parser.add_argument('--n-trials', type=int, default=20, help='Number of optimization trials (minimum 10)')
    optimize_score_parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    optimize_score_parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE, help='Random state for reproducibility')

    # Winner model optimizer
    optimize_winner_parser = subparsers.add_parser('optimize-winner-model', help='Optimize winner prediction model')
    optimize_winner_parser.add_argument('--n-trials', type=int, default=20, help='Number of optimization trials (minimum 10)')
    optimize_winner_parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    optimize_winner_parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE, help='Random state for reproducibility')

    # Model registry cleaner
    clean_registry_parser = subparsers.add_parser('clean-model-registry', help='Clean model registry by removing problematic models')
    clean_registry_parser.add_argument('--min-samples', type=int, default=100, help='Minimum number of samples required for a model to be considered valid')
    clean_registry_parser.add_argument('--keep-files', action='store_true', help='Keep model files on disk')

    # Add subcommands
    refresh_parser = subparsers.add_parser('refresh', help='Refresh data and predictions')
    refresh_parser.set_defaults(func=refresh_data)
    
    validate_parser = subparsers.add_parser('validate', help='Validate predictions against completed matches')
    validate_parser.set_defaults(func=validate_prediction_results)
    
    clear_mock_parser = subparsers.add_parser('clear-mock', help='Clear mock prediction data and prepare for real tracking')
    clear_mock_parser.set_defaults(func=clear_mock_prediction_data)

    args = parser.parse_args()

    if args.command == 'fetch-token':
        fetch_token(args)
    elif args.command == 'fetch-history':
        fetch_match_history(args)
    elif args.command == 'fetch-upcoming':
        fetch_upcoming_matches(args)
    elif args.command == 'calculate-stats':
        calculate_player_stats(args)
    elif args.command == 'train-winner-model':
        train_winner_model(args)
    elif args.command == 'train-score-model':
        train_score_model(args)
    elif args.command == 'list-models':
        list_models(args)
    elif args.command == 'optimize-score-model':
        optimize_score_model(args)
    elif args.command == 'optimize-winner-model':
        optimize_winner_model(args)
    elif args.command == 'clean-model-registry':
        clean_model_registry(args)
    elif args.command == 'refresh':
        refresh_data(args)
    elif args.command == 'validate':
        validate_prediction_results(args)
    elif args.command == 'clear-mock':
        clear_mock_prediction_data(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
