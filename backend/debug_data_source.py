#!/usr/bin/env python3
"""
Debug script to investigate data sources and dates.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add the backend directory to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from core.data.fetchers.match_history import MatchHistoryFetcher
from config.settings import MATCH_HISTORY_FILE, PREDICTION_HISTORY_FILE

def analyze_data_sources():
    """
    Analyze the data sources to understand if we're using real or mock data.
    """
    print("=== Data Source Analysis ===")
    
    # Check current date
    current_date = datetime.now()
    print(f"Current date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze match history file
    print("\n=== Match History File Analysis ===")
    if Path(MATCH_HISTORY_FILE).exists():
        with open(MATCH_HISTORY_FILE, 'r') as f:
            matches = json.load(f)
        
        print(f"Total matches in file: {len(matches)}")
        
        if matches:
            # Check date range
            dates = [match.get('fixtureStart', '') for match in matches if match.get('fixtureStart')]
            dates.sort()
            
            print(f"Earliest match: {dates[0] if dates else 'N/A'}")
            print(f"Latest match: {dates[-1] if dates else 'N/A'}")
            
            # Check if dates are in the future
            future_matches = 0
            for date_str in dates:
                try:
                    match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if match_date > current_date:
                        future_matches += 1
                except:
                    pass
            
            print(f"Matches with future dates: {future_matches}/{len(dates)}")
            
            # Show sample matches
            print("\nSample matches:")
            for i, match in enumerate(matches[:3]):
                print(f"  {i+1}. ID: {match.get('id')}, Date: {match.get('fixtureStart')}, Score: {match.get('homeScore')}-{match.get('awayScore')}")
    else:
        print("Match history file not found")
    
    # Analyze prediction history file
    print("\n=== Prediction History File Analysis ===")
    if Path(PREDICTION_HISTORY_FILE).exists():
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            predictions = json.load(f)
        
        print(f"Total predictions in file: {len(predictions)}")
        
        if predictions:
            # Check validation status
            validated_count = sum(1 for p in predictions if 'prediction_correct' in p)
            correct_count = sum(1 for p in predictions if p.get('prediction_correct') == True)
            
            print(f"Validated predictions: {validated_count}/{len(predictions)}")
            print(f"Correct predictions: {correct_count}/{validated_count if validated_count > 0 else 1}")
            
            # Show sample predictions
            print("\nSample predictions:")
            for i, pred in enumerate(predictions[:3]):
                print(f"  {i+1}. ID: {pred.get('fixtureId')}, Date: {pred.get('fixtureStart')}, Actual: {pred.get('homeScore')}-{pred.get('awayScore')}, Correct: {pred.get('prediction_correct')}")
    else:
        print("Prediction history file not found")
    
    # Try to fetch fresh data from API
    print("\n=== Fresh API Data Test ===")
    try:
        fetcher = MatchHistoryFetcher()
        print("Attempting to fetch fresh data from API...")
        fresh_matches = fetcher.fetch_match_history(save_to_file=False)
        
        print(f"Fresh matches fetched: {len(fresh_matches)}")
        
        if fresh_matches:
            # Check if fresh data has same date issues
            fresh_dates = [match.get('fixtureStart', '') for match in fresh_matches if match.get('fixtureStart')]
            fresh_dates.sort()
            
            print(f"Fresh data date range: {fresh_dates[0] if fresh_dates else 'N/A'} to {fresh_dates[-1] if fresh_dates else 'N/A'}")
            
            # Check if fresh dates are in the future
            future_fresh = 0
            for date_str in fresh_dates:
                try:
                    match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if match_date > current_date:
                        future_fresh += 1
                except:
                    pass
            
            print(f"Fresh matches with future dates: {future_fresh}/{len(fresh_dates)}")
            
            # Show sample fresh matches
            print("\nSample fresh matches:")
            for i, match in enumerate(fresh_matches[:3]):
                print(f"  {i+1}. ID: {match.get('id')}, Date: {match.get('fixtureStart')}, Score: {match.get('homeScore')}-{match.get('awayScore')}")
        
    except Exception as e:
        print(f"Error fetching fresh data: {str(e)}")

if __name__ == "__main__":
    analyze_data_sources()