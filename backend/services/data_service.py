"""
Data service for managing data operations.
"""

from config.settings import MATCH_HISTORY_DAYS, UPCOMING_MATCHES_DAYS
from config.logging_config import get_data_fetcher_logger
from utils.logging import log_execution_time, log_exceptions
from core.data.fetchers import TokenFetcher
from core.data.fetchers.match_history import MatchHistoryFetcher
from core.data.fetchers.upcoming_matches import UpcomingMatchesFetcher
from core.data.processors.player_stats import PlayerStatsProcessor
from services.supabase_service import SupabaseService

logger = get_data_fetcher_logger()


class DataService:
    """
    Service for managing data operations.
    """

    def __init__(self):
        """
        Initialize the data service.
        """
        self.token_fetcher = TokenFetcher()
        self.match_history_fetcher = MatchHistoryFetcher(days_back=MATCH_HISTORY_DAYS)
        self.upcoming_matches_fetcher = UpcomingMatchesFetcher(days_forward=UPCOMING_MATCHES_DAYS)
        self.player_stats_processor = PlayerStatsProcessor()
        self.supabase_service = SupabaseService()

    @log_execution_time(logger)
    @log_exceptions(logger)
    def fetch_token(self, force_refresh=False):
        """
        Fetch authentication token.

        Args:
            force_refresh (bool): Whether to force a token refresh

        Returns:
            str: Authentication token or None if failed
        """
        logger.info("Fetching authentication token")

        try:
            token = self.token_fetcher.get_token(force_refresh=force_refresh)
            if not token:
                logger.error("Failed to retrieve authentication token")
                return None

            logger.info("Successfully fetched authentication token")
            return token

        except Exception as e:
            logger.error(f"Error fetching authentication token: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def fetch_match_history(self, days_back=None):
        """
        Fetch match history data.

        Args:
            days_back (int): Number of days of history to fetch

        Returns:
            list: List of match data dictionaries or None if failed
        """
        logger.info(f"Fetching match history for the past {days_back or MATCH_HISTORY_DAYS} days")

        try:
            # Update days_back if provided
            if days_back is not None:
                self.match_history_fetcher.days_back = days_back

            # Fetch match history
            matches = self.match_history_fetcher.fetch_match_history()
            if not matches:
                logger.error("Failed to fetch match history")
                return None

            logger.info(f"Successfully fetched {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error fetching match history: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def fetch_upcoming_matches(self, days_forward=None):
        """
        Fetch upcoming matches data.

        Args:
            days_forward (int): Number of days to look ahead

        Returns:
            list: List of upcoming match data dictionaries or None if failed
        """
        logger.info(f"Fetching upcoming matches for the next {days_forward or UPCOMING_MATCHES_DAYS} days")

        try:
            # Update days_forward if provided
            if days_forward is not None:
                self.upcoming_matches_fetcher.days_forward = days_forward

            # Fetch upcoming matches
            matches = self.upcoming_matches_fetcher.fetch_upcoming_matches()
            if not matches:
                logger.error("Failed to fetch upcoming matches")
                return None

            logger.info(f"Successfully fetched {len(matches)} upcoming matches")
            return matches

        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def calculate_player_stats(self, matches=None):
        """
        Calculate player statistics.

        Args:
            matches (list): List of match data dictionaries

        Returns:
            dict: Dictionary of player statistics or None if failed
        """
        logger.info("Calculating player statistics")

        try:
            # Load match history if not provided
            if matches is None:
                matches = self.match_history_fetcher.load_from_file()
                if not matches:
                    logger.error("Failed to load match history")
                    return None

            # Calculate player statistics
            player_stats = self.player_stats_processor.calculate_player_stats(matches)
            if not player_stats:
                logger.error("Failed to calculate player statistics")
                return None

            logger.info(f"Successfully calculated statistics for {len(player_stats)} players")
            return player_stats

        except Exception as e:
            logger.error(f"Error calculating player statistics: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_player_stats(self):
        """
        Get player statistics from file.

        Returns:
            dict: Dictionary of player statistics or None if failed
        """
        logger.info("Getting player statistics")

        try:
            # Load player statistics
            player_stats = self.player_stats_processor.load_from_file()
            if not player_stats:
                logger.error("Failed to load player statistics")
                return None

            logger.info(f"Successfully loaded statistics for {len(player_stats)} players")
            return player_stats

        except Exception as e:
            logger.error(f"Error getting player statistics: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_match_history(self):
        """
        Get match history from file.

        Returns:
            list: List of match data dictionaries or None if failed
        """
        logger.info("Getting match history")

        try:
            # Load match history
            matches = self.match_history_fetcher.load_from_file()
            if not matches:
                logger.error("Failed to load match history")
                return None

            logger.info(f"Successfully loaded {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error getting match history: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_upcoming_matches(self):
        """
        Get upcoming matches from file.

        Returns:
            list: List of upcoming match data dictionaries or None if failed
        """
        logger.info("Getting upcoming matches")

        try:
            # Load upcoming matches
            matches = self.upcoming_matches_fetcher.load_from_file()
            if not matches:
                logger.error("Failed to load upcoming matches")
                return None

            logger.info(f"Successfully loaded {len(matches)} upcoming matches")
            return matches

        except Exception as e:
            logger.error(f"Error getting upcoming matches: {str(e)}")
            return None

    # Database-integrated methods
    @log_execution_time(logger)
    @log_exceptions(logger)
    def fetch_and_save_match_history(self, force_refresh=False):
        """
        Fetch match history and save to both file and database.
        
        Args:
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            list: Match history data or None if failed
        """
        logger.info("Fetching and saving match history")
        
        try:
            # Fetch match history
            matches = self.fetch_match_history(force_refresh=force_refresh)
            if not matches:
                return None
            
            # Save to database if available
            if self.supabase_service.is_connected():
                success = self.supabase_service.save_match_history(matches)
                if success:
                    logger.info("Successfully saved match history to database")
                else:
                    logger.warning("Failed to save match history to database")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in fetch_and_save_match_history: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def fetch_and_save_upcoming_matches(self, force_refresh=False):
        """
        Fetch upcoming matches and save to both file and database.
        
        Args:
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            list: Upcoming matches data or None if failed
        """
        logger.info("Fetching and saving upcoming matches")
        
        try:
            # Fetch upcoming matches
            matches = self.fetch_upcoming_matches(force_refresh=force_refresh)
            if not matches:
                return None
            
            # Save to database if available
            if self.supabase_service.is_connected():
                success = self.supabase_service.save_upcoming_matches(matches)
                if success:
                    logger.info("Successfully saved upcoming matches to database")
                else:
                    logger.warning("Failed to save upcoming matches to database")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in fetch_and_save_upcoming_matches: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def calculate_and_save_player_stats(self, force_refresh=False):
        """
        Calculate player statistics and save to both file and database.
        
        Args:
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            list: Player statistics data or None if failed
        """
        logger.info("Calculating and saving player statistics")
        
        try:
            # Calculate player stats
            stats = self.calculate_player_stats(force_refresh=force_refresh)
            if not stats:
                return None
            
            # Save to database if available
            if self.supabase_service.is_connected():
                success = self.supabase_service.save_player_stats(stats)
                if success:
                    logger.info("Successfully saved player stats to database")
                else:
                    logger.warning("Failed to save player stats to database")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in calculate_and_save_player_stats: {str(e)}")
            return None

    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_data_from_database_or_file(self, data_type):
        """
        Get data from database if available, otherwise fallback to file.
        
        Args:
            data_type (str): Type of data ('matches', 'upcoming', 'stats')
            
        Returns:
            list: Data from database or file
        """
        logger.info(f"Getting {data_type} data from database or file")
        
        try:
            # Try to get from database first
            if self.supabase_service.is_connected():
                if data_type == 'matches':
                    data = self.supabase_service.get_match_history()
                elif data_type == 'upcoming':
                    data = self.supabase_service.get_upcoming_matches()
                elif data_type == 'stats':
                    data = self.supabase_service.get_player_stats()
                else:
                    data = []
                
                if data:
                    logger.info(f"Successfully retrieved {len(data)} {data_type} records from database")
                    return data
                else:
                    logger.info(f"No {data_type} data found in database, falling back to file")
            
            # Fallback to file-based data
            if data_type == 'matches':
                return self.get_match_history()
            elif data_type == 'upcoming':
                return self.get_upcoming_matches()
            elif data_type == 'stats':
                return self.get_player_statistics()
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting {data_type} data: {str(e)}")
            return []

    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_system_status(self):
        """
        Get system status including database connectivity.
        
        Returns:
            dict: System status information
        """
        try:
            status = {
                'timestamp': logger.name,
                'file_system': 'operational',
                'database_connected': self.supabase_service.is_connected(),
                'database_stats': {}
            }
            
            if self.supabase_service.is_connected():
                status['database_stats'] = self.supabase_service.get_database_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}

    def cleanup(self):
        """
        Cleanup resources.
        """
        logger.info("Cleaning up data service resources")

    def save_match_history_to_db(self, matches_data):
        """
        Save match history data to database.
        
        Args:
            matches_data (list): List of match data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.supabase_service.is_connected():
            logger.warning("Database not connected, cannot save match history")
            return False
            
        try:
            # Convert the match data format to match our database schema
            processed_matches = []
            for i, match in enumerate(matches_data):
                # Handle different possible data structures
                if isinstance(match, dict):
                    # Extract match ID from various possible fields
                    match_id = (match.get('id') or 
                              match.get('fixtureId') or 
                              match.get('match_id') or 
                              f"match_{i}")
                    
                    # Extract team names safely
                    home_team = 'Unknown'
                    away_team = 'Unknown'
                    
                    if 'homeTeam' in match and isinstance(match['homeTeam'], dict):
                        home_team = match['homeTeam'].get('name', 'Unknown')
                    elif 'home_team' in match:
                        home_team = match['home_team']
                    
                    if 'awayTeam' in match and isinstance(match['awayTeam'], dict):
                        away_team = match['awayTeam'].get('name', 'Unknown')
                    elif 'away_team' in match:
                        away_team = match['away_team']
                    
                    processed_match = {
                        'match_id': str(match_id),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': match.get('homeScore') or match.get('home_score'),
                        'away_score': match.get('awayScore') or match.get('away_score'),
                        'match_date': match.get('fixtureStart') or match.get('startDate') or match.get('match_date'),
                        'tournament_id': match.get('tournamentId', 1),
                        'status': match.get('status', 'completed'),
                        'raw_data': match
                    }
                    processed_matches.append(processed_match)
            
            logger.info(f"Attempting to save {len(processed_matches)} matches to database")
            return self.supabase_service.save_match_history(processed_matches)
            
        except Exception as e:
            logger.error(f"Error saving match history to database: {str(e)}")
            return False

    def save_player_stats_to_db(self, stats_data):
        """
        Save player statistics to database.
        
        Args:
            stats_data (list or dict): Player statistics data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.supabase_service.is_connected():
            logger.warning("Database not connected, cannot save player stats")
            return False
            
        try:
            # Handle different data formats
            processed_stats = []
            
            # If stats_data is a dict with a 'players' key, extract that
            if isinstance(stats_data, dict):
                if 'players' in stats_data:
                    stats_list = stats_data['players']
                else:
                    # Convert dict to list
                    stats_list = [stats_data]
            else:
                stats_list = stats_data
            
            for stat in stats_list:
                # Handle case where stat might be a string key with dict value
                if isinstance(stat, str) and isinstance(stats_data, dict):
                    # This is a player name key, get the actual data
                    stat_data = stats_data[stat]
                    player_name = stat
                elif isinstance(stat, dict):
                    stat_data = stat
                    player_name = stat_data.get('player_name', stat_data.get('name', 'Unknown'))
                else:
                    continue
                
                processed_stat = {
                    'player_name': player_name,
                    'team': stat_data.get('team', 'Unknown'),
                    'games_played': int(stat_data.get('games_played', 0)),
                    'wins': int(stat_data.get('wins', 0)),
                    'losses': int(stat_data.get('losses', 0)),
                    'win_rate': float(stat_data.get('win_rate', 0.0)),
                    'total_score': int(stat_data.get('total_score', 0)),
                    'avg_score': float(stat_data.get('avg_score', 0.0)),
                    'raw_data': stat_data
                }
                processed_stats.append(processed_stat)
            
            logger.info(f"Attempting to save {len(processed_stats)} player stats to database")
            return self.supabase_service.save_player_stats(processed_stats)
            
        except Exception as e:
            logger.error(f"Error saving player stats to database: {str(e)}")
            return False
