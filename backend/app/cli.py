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
from services.enhanced_prediction_service import EnhancedMatchPredictionService

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
    Train the match prediction model with versioning.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = EnhancedMatchPredictionService()
    
    print(f"Preparing training data with {args.days} days of history...")
    training_df = prediction_service.prepare_training_data(
        days_back=args.days,
        min_matches_per_player=args.min_matches
    )
    
    print(f"Training model on {len(training_df)} samples...")
    version, metrics = prediction_service.train_model_with_versioning(
        training_df=training_df, 
        auto_activate=True,
        performance_threshold=0.6
    )
    
    print(f"\n=== Model Training Results ===")
    print(f"Model Version: {version}")
    print(f"Validation Accuracy: {metrics.get('val_winner_accuracy', 0):.3f}")
    print(f"Home Score MAE: {metrics.get('val_home_mae', 0):.2f}")
    print(f"Away Score MAE: {metrics.get('val_away_mae', 0):.2f}")
    print(f"Training Samples: {len(training_df)}")
    
    # List all model versions
    models_df = prediction_service.list_model_versions()
    if not models_df.empty:
        print(f"\n=== Available Model Versions ===")
        print(models_df.to_string(index=False))
    
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
    prediction_service = EnhancedMatchPredictionService()
    
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
    prediction_service = EnhancedMatchPredictionService()
    
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
    prediction_service = EnhancedMatchPredictionService()
    
    print("Getting feature importance...")
    importance_df = prediction_service.get_feature_importance()
    
    print(f"\n=== Top {args.top_n} Most Important Features ===")
    for i, row in importance_df.head(args.top_n).iterrows():
        print(f"{i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")


@log_execution_time(logger)
@log_exceptions(logger)
def list_model_versions(args):
    """
    List all available model versions with their performance metrics.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = EnhancedMatchPredictionService()
    
    print("Listing all available model versions...")
    versions_df = prediction_service.list_model_versions()
    
    if versions_df.empty:
        print("No trained models found.")
        return
    
    print(f"\n=== Available Model Versions ===")
    print(f"{'Version':<12} {'Active':<8} {'Accuracy':<10} {'Home MAE':<10} {'Away MAE':<10} {'Training Date':<20}")
    print("-" * 80)
    
    for _, row in versions_df.iterrows():
        active_mark = "YES" if row['is_active'] else "NO"
        accuracy = f"{row['val_winner_accuracy']:.3f}" if row['val_winner_accuracy'] is not None else "N/A"
        home_mae = f"{row['val_home_mae']:.2f}" if row['val_home_mae'] is not None else "N/A"
        away_mae = f"{row['val_away_mae']:.2f}" if row['val_away_mae'] is not None else "N/A"
        training_date = row['training_date'][:19] if row['training_date'] != "Unknown" else "Unknown"
        
        print(f"{row['version']:<12} {active_mark:<8} {accuracy:<10} {home_mae:<10} {away_mae:<10} {training_date:<20}")


@log_execution_time(logger)
@log_exceptions(logger)
def activate_model_version(args):
    """
    Activate a specific model version.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = EnhancedMatchPredictionService()
    
    try:
        prediction_service.activate_model_version(args.version)
        print(f"Successfully activated model version: {args.version}")
        
        # Show updated versions list
        print("\nUpdated model versions:")
        list_model_versions(args)
        
    except ValueError as e:
        print(f"Error: {e}")


@log_execution_time(logger)
@log_exceptions(logger)
def compare_model_versions(args):
    """
    Compare performance between two model versions.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    prediction_service = EnhancedMatchPredictionService()
    
    try:
        comparison = prediction_service.compare_model_versions(args.version1, args.version2)
        
        print(f"\n=== Model Comparison: {args.version1} vs {args.version2} ===")
        print(f"Better Model: {comparison['better_model']}")
        print(f"Winner Accuracy Difference: {comparison['winner_accuracy_diff']:+.3f}")
        print(f"Home MAE Difference: {comparison['home_mae_diff']:+.2f}")
        print(f"Away MAE Difference: {comparison['away_mae_diff']:+.2f}")
        
        if comparison['winner_accuracy_diff'] > 0:
            print(f"\n{args.version1} has higher accuracy by {comparison['winner_accuracy_diff']:.3f}")
        elif comparison['winner_accuracy_diff'] < 0:
            print(f"\n{args.version2} has higher accuracy by {-comparison['winner_accuracy_diff']:.3f}")
        else:
            print(f"\nBoth models have the same accuracy")
            
    except ValueError as e:
        print(f"Error: {e}")


@log_execution_time(logger)
@log_exceptions(logger)
def run_pipeline(args):
    """
    Run the complete prediction pipeline with fresh data.
    
    This command executes the full workflow:
    1. Fetch authentication token (if needed)
    2. Fetch latest match history
    3. Fetch latest player statistics
    4. Fetch upcoming matches
    5. Train new model (if requested)
    6. Generate predictions using the best model
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    print("🚀 Starting Complete Prediction Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Ensure we have a valid token
        print("\n📝 Step 1/6: Checking authentication token...")
        try:
            token_fetcher = TokenFetcher()
            token = token_fetcher.get_token(force_refresh=args.refresh_token)
            print("✅ Authentication token ready")
        except Exception as e:
            print(f"❌ Token fetch failed: {e}")
            return
          # Step 2: Fetch latest match history
        print("\n📊 Step 2/6: Fetching latest match history...")
        try:
            match_fetcher = MatchHistoryFetcher(days_back=args.history_days)
            match_fetcher.fetch_match_history(save_to_file=True)
            print("✅ Match history updated")
        except Exception as e:
            print(f"❌ Match history fetch failed: {e}")
            return
        
        # Step 3: Fetch latest player statistics
        print("\n🏀 Step 3/6: Fetching latest player statistics...")
        try:
            stats_processor = PlayerStatsProcessor()
            # Load match history and calculate stats
            match_fetcher = MatchHistoryFetcher()
            matches = match_fetcher.load_from_file()
            stats_processor.calculate_player_stats(matches, save_to_file=True)
            print("✅ Player statistics updated")
        except Exception as e:
            print(f"❌ Player stats fetch failed: {e}")
            return
        
        # Step 4: Fetch upcoming matches
        print("\n🔮 Step 4/6: Fetching upcoming matches...")
        try:
            upcoming_fetcher = UpcomingMatchesFetcher()
            upcoming_fetcher.fetch_upcoming_matches(save_to_file=True)
            print("✅ Upcoming matches updated")
        except Exception as e:
            print(f"❌ Upcoming matches fetch failed: {e}")
            return
        
        # Step 5: Train new model (if requested)
        if args.train_new_model:
            print("\n🤖 Step 5/6: Training new prediction model...")
            try:
                prediction_service = EnhancedMatchPredictionService()
                
                # Train new model
                result = prediction_service.train_model(
                    days_back=args.training_days,
                    min_matches_per_player=args.min_matches
                )
                
                print(f"✅ New model trained: {result['version']}")
                print(f"   Accuracy: {result['accuracy']:.1%}")
                print(f"   Home MAE: {result['home_mae']:.2f}")
                print(f"   Away MAE: {result['away_mae']:.2f}")
                
                if result.get('is_best_model'):
                    print(f"🏆 New model is now active (best performer)")
                else:
                    print(f"📊 New model saved but previous model remains active")
                    
            except Exception as e:
                print(f"❌ Model training failed: {e}")
                print("⚠️  Continuing with existing model...")
        else:
            print("\n⏭️  Step 5/6: Skipping model training (use --train to enable)")
        
        # Step 6: Generate predictions
        print("\n🎯 Step 6/6: Generating predictions...")
        try:
            prediction_service = EnhancedMatchPredictionService()
            
            # Generate predictions
            prediction_service.predict_upcoming_matches()
            
            print("✅ Predictions generated successfully!")
            print(f"📁 Results saved to: output/match_predictions.csv")
            
        except Exception as e:
            print(f"❌ Prediction generation failed: {e}")
            return
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 Pipeline completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Fresh data fetched (matches, stats, upcoming)")
        if args.train_new_model:
            print("   ✅ New model trained and evaluated")
        print("   ✅ Predictions generated with latest data")
        print("\n📁 Check these files:")
        print("   • output/match_predictions.csv - Latest predictions")
        print("   • output/upcoming_matches.json - Upcoming matches data")
        print("   • logs/ - Detailed execution logs")
        
        if not args.train_new_model:
            print("\n💡 Tip: Use --train flag to train a new model with the fresh data")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("📋 Check the logs directory for detailed error information")


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

    # Model versioning commands
    list_versions_parser = subparsers.add_parser('list-models', help='List all available model versions')
    
    activate_parser = subparsers.add_parser('activate-model', help='Activate a specific model version')
    activate_parser.add_argument('version', help='Model version to activate (e.g., v1.0.1)')
    
    compare_parser = subparsers.add_parser('compare-models', help='Compare two model versions')
    compare_parser.add_argument('version1', help='First model version (e.g., v1.0.1)')
    compare_parser.add_argument('version2', help='Second model version (e.g., v1.0.2)')

    # Pipeline command - Run complete workflow
    pipeline_parser = subparsers.add_parser('run-pipeline', help='Run complete prediction pipeline with fresh data')
    pipeline_parser.add_argument('--train', action='store_true', dest='train_new_model', 
                                help='Train a new model with the fresh data')
    pipeline_parser.add_argument('--refresh-token', action='store_true', 
                                help='Force refresh authentication token')
    pipeline_parser.add_argument('--history-days', type=int, default=90, 
                                help='Number of days of match history to fetch')
    pipeline_parser.add_argument('--training-days', type=int, default=60, 
                                help='Number of days of history to use for training')
    pipeline_parser.add_argument('--min-matches', type=int, default=5, 
                                help='Minimum matches per player for training')

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
    elif args.command == 'list-models':
        list_model_versions(args)
    elif args.command == 'activate-model':
        activate_model_version(args)
    elif args.command == 'compare-models':
        compare_model_versions(args)
    elif args.command == 'run-pipeline':
        run_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
