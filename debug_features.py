#!/usr/bin/env python3
"""
Debug script to test feature extraction with small datasets.
"""

import json
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent / 'backend'
sys.path.append(str(backend_dir))

from config.settings import PLAYER_STATS_FILE, MATCH_HISTORY_FILE
from core.models.feature_engineering import FeatureEngineer

def debug_feature_extraction(sample_size=50):
    """Debug feature extraction with a small sample."""
    
    # Load data
    print(f"Loading player stats from {PLAYER_STATS_FILE}")
    with open(PLAYER_STATS_FILE, 'r', encoding='utf-8') as f:
        player_stats = json.load(f)
    
    print(f"Loading match history from {MATCH_HISTORY_FILE}")
    with open(MATCH_HISTORY_FILE, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    print(f"Total matches: {len(matches)}")
    print(f"Total players: {len(player_stats)}")
    
    # Take a small sample
    sample_matches = matches[:sample_size]
    print(f"\nUsing sample of {len(sample_matches)} matches")
    
    # Check first few matches
    print("\nFirst 3 matches:")
    for i, match in enumerate(sample_matches[:3]):
        print(f"Match {i+1}:")
        print(f"  ID: {match.get('id')}")
        print(f"  Home Player: {match.get('homePlayer', {}).get('name')} (ID: {match.get('homePlayer', {}).get('id')})")
        print(f"  Away Player: {match.get('awayPlayer', {}).get('name')} (ID: {match.get('awayPlayer', {}).get('id')})")
        print(f"  Home Score: {match.get('homeScore')}")
        print(f"  Away Score: {match.get('awayScore')}")
        
        # Check if players exist in stats
        home_id = str(match.get('homePlayer', {}).get('id', ''))
        away_id = str(match.get('awayPlayer', {}).get('id', ''))
        print(f"  Home player in stats: {home_id in player_stats}")
        print(f"  Away player in stats: {away_id in player_stats}")
        print()
    
    # Filter to only players in the sample
    match_player_ids = set()
    valid_matches = 0
    for match in sample_matches:
        if 'homeScore' in match and 'awayScore' in match:
            home_id = str(match.get('homePlayer', {}).get('id', ''))
            away_id = str(match.get('awayPlayer', {}).get('id', ''))
            if home_id in player_stats and away_id in player_stats:
                match_player_ids.add(home_id)
                match_player_ids.add(away_id)
                valid_matches += 1
    
    print(f"Valid matches with scores and player stats: {valid_matches}")
    print(f"Unique players in sample: {len(match_player_ids)}")
    
    if valid_matches == 0:
        print("\nERROR: No valid matches found!")
        return
    
    # Filter player stats
    filtered_player_stats = {pid: stats for pid, stats in player_stats.items() 
                           if pid in match_player_ids}
    
    print(f"Filtered player stats: {len(filtered_player_stats)}")
    
    # Try feature extraction
    print("\nTesting feature extraction...")
    feature_engineer = FeatureEngineer()
    
    try:
        X, y = feature_engineer.extract_features(
            filtered_player_stats, sample_matches, for_score_prediction=False
        )
        print(f"SUCCESS: Extracted {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features")
        if len(X) > 0:
            print(f"Labels: {y[:5] if len(y) > 5 else y}")
    except Exception as e:
        print(f"ERROR in feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_extraction(200)