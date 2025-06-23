"""
Data migration script to transfer data from JSON files to Supabase database.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the parent directory to the Python path so we can import our modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from backend.config.settings import (
    MATCH_HISTORY_FILE, PLAYER_STATS_FILE, UPCOMING_MATCHES_FILE,
    SUPABASE_URL, SUPABASE_KEY
)
from backend.config.logging_config import get_data_fetcher_logger
from backend.services.supabase_service import SupabaseService

logger = get_data_fetcher_logger()


class DataMigration:
    """
    Handles migration of data from JSON files to Supabase database.
    """

    def __init__(self):
        """
        Initialize the data migration service.
        """
        self.supabase_service = SupabaseService()

    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            List[Dict]: Loaded data or empty list if file doesn't exist
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # If the JSON contains metadata, extract the actual data
                    if 'matches' in data:
                        return data['matches']
                    elif 'players' in data:
                        return data['players']
                    elif 'stats' in data:
                        return data['stats']
                    else:
                        # Return the dict as a single item list
                        return [data]
                elif isinstance(data, list):
                    return data
                else:
                    logger.error(f"Unexpected data format in {file_path}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []

    def migrate_match_history(self) -> bool:
        """
        Migrate match history from JSON to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting match history migration...")
        
        matches = self.load_json_file(MATCH_HISTORY_FILE)
        if not matches:
            logger.warning("No match history data found to migrate")
            return True

        # Transform data if needed
        processed_matches = []
        for match in matches:
            # Ensure each match has the required fields
            processed_match = {
                'id': match.get('id', match.get('match_id', f"migrated_{len(processed_matches)}")),
                'home_team': match.get('home_team', match.get('homeTeam', 'Unknown')),
                'away_team': match.get('away_team', match.get('awayTeam', 'Unknown')),
                'home_score': match.get('home_score', match.get('homeScore', 0)),
                'away_score': match.get('away_score', match.get('awayScore', 0)),
                'date': match.get('date', match.get('match_date', datetime.now().isoformat())),
                'tournament_id': match.get('tournament_id', match.get('tournamentId', 1)),
                'status': match.get('status', 'completed')
            }
            processed_matches.append(processed_match)

        success = self.supabase_service.save_match_history(processed_matches)
        if success:
            logger.info(f"Successfully migrated {len(processed_matches)} matches")
        else:
            logger.error("Failed to migrate match history")
        
        return success

    def migrate_player_stats(self) -> bool:
        """
        Migrate player statistics from JSON to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting player stats migration...")
        
        stats = self.load_json_file(PLAYER_STATS_FILE)
        if not stats:
            logger.warning("No player stats data found to migrate")
            return True

        # Transform data if needed
        processed_stats = []
        for stat in stats:
            # Ensure each stat has the required fields
            processed_stat = {
                'player_name': stat.get('player_name', stat.get('name', 'Unknown Player')),
                'team': stat.get('team', 'Unknown Team'),
                'games_played': stat.get('games_played', stat.get('gamesPlayed', 0)),
                'wins': stat.get('wins', 0),
                'losses': stat.get('losses', 0),
                'win_rate': stat.get('win_rate', stat.get('winRate', 0.0)),
                'total_score': stat.get('total_score', stat.get('totalScore', 0)),
                'avg_score': stat.get('avg_score', stat.get('avgScore', 0.0))
            }
            processed_stats.append(processed_stat)

        success = self.supabase_service.save_player_stats(processed_stats)
        if success:
            logger.info(f"Successfully migrated {len(processed_stats)} player stats")
        else:
            logger.error("Failed to migrate player stats")
        
        return success

    def migrate_upcoming_matches(self) -> bool:
        """
        Migrate upcoming matches from JSON to database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting upcoming matches migration...")
        
        matches = self.load_json_file(UPCOMING_MATCHES_FILE)
        if not matches:
            logger.warning("No upcoming matches data found to migrate")
            return True

        # Transform data if needed
        processed_matches = []
        for match in matches:
            # Ensure each match has the required fields
            processed_match = {
                'id': match.get('id', match.get('match_id', f"upcoming_{len(processed_matches)}")),
                'home_team': match.get('home_team', match.get('homeTeam', 'Unknown')),
                'away_team': match.get('away_team', match.get('awayTeam', 'Unknown')),
                'date': match.get('date', match.get('scheduled_date', datetime.now().isoformat())),
                'tournament_id': match.get('tournament_id', match.get('tournamentId', 1)),
                'status': match.get('status', 'scheduled')
            }
            processed_matches.append(processed_match)

        success = self.supabase_service.save_upcoming_matches(processed_matches)
        if success:
            logger.info(f"Successfully migrated {len(processed_matches)} upcoming matches")
        else:
            logger.error("Failed to migrate upcoming matches")
        
        return success

    def run_migration(self) -> bool:
        """
        Run the complete data migration.
        
        Returns:
            bool: True if all migrations successful, False otherwise
        """
        logger.info("Starting data migration to Supabase...")

        # Check if Supabase is configured and connected
        if not self.supabase_service.is_connected():
            logger.error("Supabase is not configured or connected. Please check your environment variables.")
            logger.error("Required: SUPABASE_URL and SUPABASE_KEY")
            return False

        # Test database connection
        if not self.supabase_service.test_connection():
            logger.error("Failed to connect to Supabase database")
            return False

        # Run migrations
        results = []
        
        # Migrate match history
        results.append(self.migrate_match_history())
        
        # Migrate player stats
        results.append(self.migrate_player_stats())
        
        # Migrate upcoming matches
        results.append(self.migrate_upcoming_matches())

        # Check if all migrations were successful
        success = all(results)
        
        if success:
            logger.info("All data migrations completed successfully!")
            # Print database stats
            stats = self.supabase_service.get_database_stats()
            logger.info(f"Database stats: {stats}")
        else:
            logger.error("Some migrations failed. Please check the logs for details.")

        return success


def main():
    """
    Main function to run the data migration.
    """
    print("2K Flash - Data Migration to Supabase")
    print("=" * 40)
    
    # Check if Supabase is configured
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Supabase configuration missing!")
        print("Please set the following environment variables:")
        print("- SUPABASE_URL")
        print("- SUPABASE_KEY")
        print("\nOr configure them in your docker-compose.yml file.")
        return 1

    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Migration files:")
    print(f"  - Match History: {MATCH_HISTORY_FILE}")
    print(f"  - Player Stats: {PLAYER_STATS_FILE}")
    print(f"  - Upcoming Matches: {UPCOMING_MATCHES_FILE}")
    print()

    # Confirm migration
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Migration cancelled.")
        return 0

    # Run migration
    migration = DataMigration()
    success = migration.run_migration()
    
    if success:
        print("\n✅ Migration completed successfully!")
        return 0
    else:
        print("\n❌ Migration failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())
