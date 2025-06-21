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
    
    def get_live_matches(self) -> Dict:
        """Get live matches data."""
        live_scores = self.fetch_live_scores()
        
        if not live_scores:
            return {
                'matches': [],
                'total_count': 0,
                'last_updated': datetime.now().isoformat(),
                'error': 'No live data available'
            }
        
        # Process live scores
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
                'has_live_scores': True
            }
            matched_data.append(match_data)
        
        return {
            'matches': matched_data,
            'total_count': len(matched_data),
            'last_updated': datetime.now().isoformat()
        }