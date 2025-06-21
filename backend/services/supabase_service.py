"""
Supabase service for database operations.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY
from config.logging_config import get_data_fetcher_logger

logger = get_data_fetcher_logger()


class SupabaseService:
    """
    Service for managing Supabase database operations.
    """

    def __init__(self):
        """
        Initialize the Supabase service.
        """
        self.client: Optional[Client] = None
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the Supabase client.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("Supabase URL or Key not configured. Database operations will be disabled.")
            return

        try:
            # Use service role key if available for admin operations, otherwise use anon key
            key = SUPABASE_SERVICE_ROLE_KEY if SUPABASE_SERVICE_ROLE_KEY else SUPABASE_KEY
            self.client = create_client(SUPABASE_URL, key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self.client = None

    def is_connected(self) -> bool:
        """
        Check if the Supabase client is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None

    # Match History Operations
    def save_match_history(self, matches: List[Dict[str, Any]]) -> bool:
        """
        Save match history to the database.
        
        Args:
            matches (List[Dict]): List of match data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False

        try:
            # Prepare match data for insertion
            match_records = []
            for match in matches:
                record = {
                    'match_id': match.get('id'),
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'home_score': match.get('home_score'),
                    'away_score': match.get('away_score'),
                    'match_date': match.get('date'),
                    'tournament_id': match.get('tournament_id'),
                    'status': match.get('status'),
                    'raw_data': match,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                match_records.append(record)

            # Insert or update match records
            result = self.client.table('matches').upsert(
                match_records,
                on_conflict='match_id'
            ).execute()

            logger.info(f"Successfully saved {len(match_records)} matches to database")
            return True

        except Exception as e:
            logger.error(f"Error saving match history: {str(e)}")
            return False

    def get_match_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get match history from the database.
        
        Args:
            limit (int): Maximum number of matches to return
            
        Returns:
            List[Dict]: List of match data
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return []

        try:
            result = self.client.table('matches').select('*').order(
                'match_date', desc=True
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error fetching match history: {str(e)}")
            return []

    # Player Statistics Operations
    def save_player_stats(self, stats: List[Dict[str, Any]]) -> bool:
        """
        Save player statistics to the database.
        
        Args:
            stats (List[Dict]): List of player statistics
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False

        try:
            # Prepare player stats for insertion
            stat_records = []
            for stat in stats:
                record = {
                    'player_name': stat.get('player_name'),
                    'team': stat.get('team'),
                    'games_played': stat.get('games_played'),
                    'wins': stat.get('wins'),
                    'losses': stat.get('losses'),
                    'win_rate': stat.get('win_rate'),
                    'total_score': stat.get('total_score'),
                    'avg_score': stat.get('avg_score'),
                    'raw_data': stat,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                stat_records.append(record)

            # Insert or update player stats
            result = self.client.table('player_stats').upsert(
                stat_records,
                on_conflict='player_name'
            ).execute()

            logger.info(f"Successfully saved {len(stat_records)} player stats to database")
            return True

        except Exception as e:
            logger.error(f"Error saving player stats: {str(e)}")
            return False

    def get_player_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get player statistics from the database.
        
        Args:
            limit (int): Maximum number of player stats to return
            
        Returns:
            List[Dict]: List of player statistics
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return []

        try:
            result = self.client.table('player_stats').select('*').order(
                'win_rate', desc=True
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            return []

    # Upcoming Matches Operations
    def save_upcoming_matches(self, matches: List[Dict[str, Any]]) -> bool:
        """
        Save upcoming matches to the database.
        
        Args:
            matches (List[Dict]): List of upcoming matches
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False

        try:
            # Prepare upcoming matches for insertion
            match_records = []
            for match in matches:
                record = {
                    'match_id': match.get('id'),
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'scheduled_date': match.get('date'),
                    'tournament_id': match.get('tournament_id'),
                    'status': match.get('status', 'scheduled'),
                    'raw_data': match,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                match_records.append(record)

            # Insert or update upcoming matches
            result = self.client.table('upcoming_matches').upsert(
                match_records,
                on_conflict='match_id'
            ).execute()

            logger.info(f"Successfully saved {len(match_records)} upcoming matches to database")
            return True

        except Exception as e:
            logger.error(f"Error saving upcoming matches: {str(e)}")
            return False

    def get_upcoming_matches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get upcoming matches from the database.
        
        Args:
            limit (int): Maximum number of matches to return
            
        Returns:
            List[Dict]: List of upcoming matches
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return []

        try:
            result = self.client.table('upcoming_matches').select('*').order(
                'scheduled_date', desc=False
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {str(e)}")
            return []

    # Utility Methods
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self.client:
            return False

        try:
            # Try to fetch a simple query to test connection
            result = self.client.table('matches').select('count').limit(1).execute()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict: Database statistics
        """
        if not self.client:
            return {'error': 'Database not connected'}

        try:
            stats = {}
            
            # Get table counts
            matches_count = self.client.table('matches').select('count').execute()
            stats['matches_count'] = len(matches_count.data) if matches_count.data else 0
            
            player_stats_count = self.client.table('player_stats').select('count').execute()
            stats['player_stats_count'] = len(player_stats_count.data) if player_stats_count.data else 0
            
            upcoming_matches_count = self.client.table('upcoming_matches').select('count').execute()
            stats['upcoming_matches_count'] = len(upcoming_matches_count.data) if upcoming_matches_count.data else 0
            
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {'error': str(e)}
