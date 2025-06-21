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
        Cleanup resources before shutting down.
        """
        logger.info("Cleaning up resources")
        
        try:
            # Perform any necessary cleanup actions
            pass
        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
