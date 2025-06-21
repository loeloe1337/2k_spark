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
from services.match_prediction_service import MatchPredictionService

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
def train_prediction_model(args):
    """
    Train the match prediction model.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = MatchPredictionService()
    
    print(f"Preparing training data with {args.days} days of history...")
    training_df = prediction_service.prepare_training_data(
        days_back=args.days,
        min_matches_per_player=args.min_matches
    )
    
    print(f"Training model on {len(training_df)} samples...")
    metrics = prediction_service.train_model(training_df=training_df, save_model=True)
    
    print("\n=== Training Results ===")
    print(f"Winner Prediction Accuracy: {metrics['val_winner_accuracy']:.3f}")
    print(f"Home Score MAE: {metrics['val_home_mae']:.2f}")
    print(f"Away Score MAE: {metrics['val_away_mae']:.2f}")
    print(f"Total Score MAE: {metrics['val_total_mae']:.2f}")
    print(f"Model saved successfully!")


@log_execution_time(logger)
@log_exceptions(logger)
def predict_matches(args):
    """
    Predict upcoming matches.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = MatchPredictionService()
    
    print("Generating predictions for upcoming matches...")
    predictions_df = prediction_service.predict_upcoming_matches(load_model=True)
    
    if predictions_df.empty:
        print("No upcoming matches to predict")
        return
    
    # Get and display summary
    summary = prediction_service.get_prediction_summary(predictions_df)
    
    print(f"\n=== Match Predictions ===")
    print(f"Total matches: {summary['total_matches']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    print(f"High confidence matches: {summary['high_confidence_matches']}")
    print(f"Average predicted total score: {summary['predicted_total_score_avg']:.1f}")
    
    print(f"\n=== Individual Predictions ===")
    for pred in summary['predictions']:
        print(f"\n{pred['home_player']} vs {pred['away_player']}")
        print(f"  Predicted Winner: {pred['predicted_winner']}")
        print(f"  Win Probability: {pred['home_win_probability']:.1%}")
        print(f"  Predicted Scores: {pred['predicted_scores']['home']:.1f} - {pred['predicted_scores']['away']:.1f}")
        print(f"  Total Score: {pred['predicted_scores']['total']:.1f}")
        print(f"  Confidence: {pred['confidence']:.3f}")


@log_execution_time(logger)
@log_exceptions(logger)
def evaluate_model(args):
    """
    Evaluate model performance.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = MatchPredictionService()
    
    print(f"Evaluating model on last {args.test_days} days of matches...")
    metrics = prediction_service.evaluate_model(test_days_back=args.test_days)
    
    print(f"\n=== Model Evaluation ===")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Winner accuracy: {metrics['winner_accuracy']:.3f}")
    print(f"Home score MAE: {metrics['home_score_mae']:.2f}")
    print(f"Away score MAE: {metrics['away_score_mae']:.2f}")
    print(f"Total score MAE: {metrics['total_score_mae']:.2f}")


@log_execution_time(logger)
@log_exceptions(logger)
def show_feature_importance(args):
    """
    Show feature importance from trained model.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = MatchPredictionService()
    
    print("Getting feature importance...")
    importance_df = prediction_service.get_feature_importance()
    
    print(f"\n=== Top {args.top_n} Most Important Features ===")
    for i, row in importance_df.head(args.top_n).iterrows():
        print(f"{i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")


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

    # ML Model commands
    train_parser = subparsers.add_parser('train-model', help='Train the match prediction model')
    train_parser.add_argument('--days', type=int, default=60, help='Number of days of history to use for training')
    train_parser.add_argument('--min-matches', type=int, default=5, help='Minimum matches per player')

    predict_parser = subparsers.add_parser('predict', help='Predict upcoming matches')

    evaluate_parser = subparsers.add_parser('evaluate-model', help='Evaluate model performance')
    evaluate_parser.add_argument('--test-days', type=int, default=7, help='Number of recent days for testing')

    importance_parser = subparsers.add_parser('feature-importance', help='Show feature importance')
    importance_parser.add_argument('--top-n', type=int, default=20, help='Number of top features to show')

    args = parser.parse_args()

    if args.command == 'fetch-token':
        fetch_token(args)
    elif args.command == 'fetch-matches':
        fetch_match_history(args)
    elif args.command == 'fetch-upcoming':
        fetch_upcoming_matches(args)
    elif args.command == 'calculate-stats':
        calculate_player_stats(args)
    elif args.command == 'train-model':
        train_prediction_model(args)
    elif args.command == 'predict':
        predict_matches(args)
    elif args.command == 'evaluate-model':
        evaluate_model(args)
    elif args.command == 'feature-importance':
        show_feature_importance(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
