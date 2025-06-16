"""Live scores service for fetching and processing live NBA data from H2H API."""

import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LiveScoresService:
    """Service for fetching live NBA scores from H2H API."""
    
    def __init__(self):
        self.api_url = "https://api-h2h.hudstats.com/v1/live/nba"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def fetch_live_scores(self) -> List[Dict]:
        """Fetch live scores from H2H API."""
        try:
            response = requests.get(self.api_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched {len(data)} live matches")
                # Debug: Log sample data to understand format
                if data and len(data) > 0:
                    logger.info(f"Sample live match data: {data[0]}")
                return data
            else:
                logger.error(f"Failed to fetch live scores: HTTP {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching live scores: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing live scores JSON: {str(e)}")
            return []
    
    def process_live_match(self, live_match: Dict) -> Dict:
        """Process a single live match into our format."""
        start_date = live_match.get('startDate')
        logger.info(f"Processing live match with startDate: {start_date}, status: {live_match.get('status')}")
        
        return {
            'external_id': live_match.get('externalId'),
            'stream_name': live_match.get('streamName'),
            'team_a_name': live_match.get('teamAName'),
            'team_b_name': live_match.get('teamBName'),
            'participant_a_name': live_match.get('participantAName'),
            'participant_b_name': live_match.get('participantBName'),
            'start_date': start_date,
            'status': live_match.get('status'),
            'team_a_score': live_match.get('teamAScore'),
            'team_b_score': live_match.get('teamBScore'),
            'live_updated_at': datetime.now().isoformat()
        }
    
    def match_with_predictions(self, live_scores: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Match live scores with existing predictions."""
        matched_data = []
        
        for live_match in live_scores:
            processed_live = self.process_live_match(live_match)
            
            # Try to find matching prediction based on participant names
            matching_prediction = None
            participant_a = processed_live.get('participant_a_name', '').lower()
            participant_b = processed_live.get('participant_b_name', '').lower()
            
            for prediction in predictions:
                home_player = prediction.get('homePlayer', {}).get('name', '').lower()
                away_player = prediction.get('awayPlayer', {}).get('name', '').lower()
                
                # Check if participants match (in either order)
                if ((participant_a in home_player or home_player in participant_a) and
                    (participant_b in away_player or away_player in participant_b)) or \
                   ((participant_a in away_player or away_player in participant_a) and
                    (participant_b in home_player or home_player in participant_b)):
                    matching_prediction = prediction
                    break
            
            if matching_prediction:
                # Merge live data with prediction
                merged_match = matching_prediction.copy()
                merged_match['live_scores'] = processed_live
                merged_match['has_live_scores'] = True
                matched_data.append(merged_match)
            else:
                # Create a basic match structure for unmatched live games
                unmatched_match = {
                    'id': processed_live['external_id'],
                    'fixtureId': processed_live['external_id'],
                    'homePlayer': {'name': processed_live['participant_a_name']},
                    'awayPlayer': {'name': processed_live['participant_b_name']},
                    'fixtureStart': processed_live['start_date'],
                    'live_scores': processed_live,
                    'has_live_scores': True,
                    'prediction_available': False
                }
                matched_data.append(unmatched_match)
        
        return matched_data
    
    def get_live_matches_with_predictions(self, predictions: List[Dict] = None) -> Dict:
        """Get live matches merged with predictions."""
        live_scores = self.fetch_live_scores()
        
        if not live_scores:
            return {
                'matches': [],
                'total_count': 0,
                'last_updated': datetime.now().isoformat(),
                'error': 'No live data available'
            }
        
        if predictions:
            matched_data = self.match_with_predictions(live_scores, predictions)
        else:
            # Process live scores without predictions
            matched_data = []
            for live_match in live_scores:
                processed_live = self.process_live_match(live_match)
                match_data = {
                    'id': processed_live['external_id'],
                    'fixtureId': processed_live['external_id'],
                    'homePlayer': {'name': processed_live['participant_a_name']},
                    'awayPlayer': {'name': processed_live['participant_b_name']},
                    'fixtureStart': processed_live['start_date'],
                    'live_scores': processed_live,
                    'has_live_scores': True,
                    'prediction_available': False
                }
                matched_data.append(match_data)
        
        return {
            'matches': matched_data,
            'total_count': len(matched_data),
            'last_updated': datetime.now().isoformat()
        }