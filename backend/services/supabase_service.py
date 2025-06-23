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
                    'match_id': match.get('match_id', f"{match.get('home_team', '')}_{match.get('away_team', '')}_{datetime.now().strftime('%Y%m%d')}"),
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'home_player': match.get('home_player'),
                    'away_player': match.get('away_player'),
                    'scheduled_date': match.get('scheduled_date') or match.get('match_date'),
                    'tournament_id': match.get('tournament_id'),
                    'tournament_name': match.get('tournament_name'),
                    'status': match.get('status', 'scheduled'),
                    'raw_data': match,
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

    def save_model_registry(self, model_data: Dict[str, Any]) -> bool:
        """
        Save model metadata to the model registry.
        
        Args:
            model_data (Dict): Model metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False

        try:
            # First deactivate any existing active models
            if model_data.get('is_active', False):
                self.client.table('model_registry').update({
                    'is_active': False
                }).eq('model_name', model_data.get('model_name')).execute()

            # Insert new model record
            result = self.client.table('model_registry').upsert([model_data], on_conflict='model_name,model_version').execute()

            logger.info(f"Successfully saved model {model_data.get('model_name')} {model_data.get('model_version')} to registry")
            return True

        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
            return False

    def save_match_predictions(self, predictions: List[Dict[str, Any]]) -> bool:
        """
        Save match predictions to the database.
        
        Args:
            predictions (List[Dict]): List of match predictions
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False

        try:            # Prepare predictions for insertion
            prediction_records = []
            for pred in predictions:
                record = {
                    'match_id': pred.get('match_id'),
                    'model_version': pred.get('model_version'),
                    'home_player': pred.get('home_player'),
                    'away_player': pred.get('away_player'),
                    'predicted_home_score': pred.get('predicted_home_score'),
                    'predicted_away_score': pred.get('predicted_away_score'),
                    'predicted_total_score': pred.get('predicted_total_score'),
                    'predicted_winner': pred.get('predicted_winner'),
                    'home_win_probability': pred.get('home_win_probability'),
                    'confidence_score': pred.get('confidence_score'),
                    'prediction_date': datetime.now(timezone.utc).isoformat()
                }
                prediction_records.append(record)

            # Insert predictions
            result = self.client.table('match_predictions').upsert(
                prediction_records, 
                on_conflict='match_id,model_version'
            ).execute()

            logger.info(f"Successfully saved {len(prediction_records)} predictions to database")
            return True

        except Exception as e:
            logger.error(f"Error saving match predictions: {str(e)}")
            return False

    # --- Training Job Status Methods ---
    def save_training_job_status(self, job_id: str, status: dict) -> bool:
        """
        Save or update a training job status in the database.
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        try:
            record = {
                'job_id': job_id,
                'status': status.get('status'),
                'timestamp': status.get('timestamp'),
                'message': status.get('message'),
                'model_version': status.get('model_version'),
                'metrics': status.get('metrics'),
                'raw_data': status
            }
            self.client.table('training_jobs').upsert([record], on_conflict='job_id').execute()
            logger.info(f"Saved training job status for job_id={job_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving training job status: {str(e)}")
            return False

    def get_training_job_status(self, job_id: str) -> Optional[dict]:
        """
        Get the status of a training job by job_id.
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return None
        try:
            result = self.client.table('training_jobs').select('*').eq('job_id', job_id).limit(1).execute()
            if result.data:
                return result.data[0].get('raw_data', result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error fetching training job status: {str(e)}")
            return None

    # --- Upsert predictions (for /api/ml/predictions endpoint) ---
    def upsert_match_predictions(self, predictions: list) -> bool:
        """
        Upsert match predictions to the database (by match_id, model_version).
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        try:
            for pred in predictions:
                pred['prediction_date'] = datetime.now(timezone.utc).isoformat()
            self.client.table('match_predictions').upsert(predictions, on_conflict='match_id,model_version').execute()
            logger.info(f"Upserted {len(predictions)} match predictions to database")
            return True
        except Exception as e:
            logger.error(f"Error upserting match predictions: {str(e)}")
            return False

    # --- Model file upload/download stubs for persistent storage ---
    def upload_model_file(self, model_path: Any, version: str) -> bool:
        """
        Upload a trained model file to Supabase Storage (bucket: 'models').
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        try:
            bucket = 'models'
            dest_filename = f"model_{version}.pkl"
            with open(model_path, 'rb') as f:
                file_bytes = f.read()
            # Create bucket if not exists (idempotent)
            try:
                self.client.storage.create_bucket(bucket)
            except Exception:
                pass  # Already exists
            self.client.storage.from_(bucket).upload(dest_filename, file_bytes, upsert=True)
            logger.info(f"Uploaded model file to Supabase Storage: {bucket}/{dest_filename}")
            return True
        except Exception as e:
            logger.error(f"Error uploading model file to Supabase Storage: {str(e)}")
            return False

    def download_model_file(self, version: str, dest_path: Any) -> bool:
        """
        Download a model file from Supabase Storage (bucket: 'models') if not present locally.
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
        try:
            bucket = 'models'
            filename = f"model_{version}.pkl"
            # Download file from storage
            res = self.client.storage.from_(bucket).download(filename)
            if hasattr(dest_path, 'write_bytes'):
                dest_path.write_bytes(res)
            else:
                with open(dest_path, 'wb') as f:
                    f.write(res)
            logger.info(f"Downloaded model file from Supabase Storage: {bucket}/{filename} -> {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model file from Supabase Storage: {str(e)}")
            return False

    def get_database_stats(self) -> dict:
        """
        Return basic stats for key tables: row counts for matches, player_stats, upcoming_matches, model_registry, training_jobs.
        """
        if not self.client:
            logger.error("Supabase client not initialized")
            return {"error": "not connected"}
        stats = {}
        try:
            for table in [
                "matches",
                "player_stats",
                "upcoming_matches",
                "model_registry",
                "training_jobs"
            ]:
                try:
                    res = self.client.table(table).select('id').execute()
                    stats[table] = len(res.data) if res.data else 0
                except Exception as e:
                    stats[table] = f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            stats["error"] = str(e)
        return stats

    # Utility Methods
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self.client:
            return False
        # You can implement a simple query here if needed, or just return True if client exists
        return True
