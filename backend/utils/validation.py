"""
Validation utility functions for the 2K Flash application.
"""

import json
import os
from pathlib import Path


def validate_json_file(file_path):
    """
    Validate that a file exists and contains valid JSON.

    Args:
        file_path (str or Path): Path to the JSON file

    Returns:
        bool: True if the file exists and contains valid JSON, False otherwise
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def validate_match_data(match):
    """
    Validate that a match data dictionary contains all required fields.

    Args:
        match (dict): Match data dictionary

    Returns:
        bool: True if the match data is valid, False otherwise
    """
    # Import logger here to avoid circular imports
    from config.logging_config import get_data_fetcher_logger
    logger = get_data_fetcher_logger()

    # Check for required fields in the H2H GG League API format
    required_fields = [
        'fixtureId',  # or 'id'
        'homeParticipantId',
        'homeParticipantName',
        'awayParticipantId',
        'awayParticipantName',
        'homeTeamId',
        'homeTeamName',
        'awayTeamId',
        'awayTeamName',
        'fixtureStart'
    ]

    # Map API fields to our expected format with more alternatives
    field_mapping = {
        'fixtureId': ['fixtureId', 'id', 'matchId', 'gameId'],
        'homeParticipantId': ['homeParticipantId', 'homePlayerId', 'homeId'],
        'homeParticipantName': ['homeParticipantName', 'homePlayerName', 'homeName', 'homePlayer'],
        'awayParticipantId': ['awayParticipantId', 'awayPlayerId', 'awayId'],
        'awayParticipantName': ['awayParticipantName', 'awayPlayerName', 'awayName', 'awayPlayer'],
        'homeTeamId': ['homeTeamId', 'homeTeam.id', 'homeTeamID'],
        'homeTeamName': ['homeTeamName', 'homeTeam.name', 'homeTeam'],
        'awayTeamId': ['awayTeamId', 'awayTeam.id', 'awayTeamID'],
        'awayTeamName': ['awayTeamName', 'awayTeam.name', 'awayTeam'],
        'fixtureStart': ['fixtureStart', 'startTime', 'matchStart', 'gameStart', 'start']
    }

    # Check that all required fields are present using the mapping
    missing_fields = []
    for field, possible_names in field_mapping.items():
        field_present = False
        for name in possible_names:
            if name in match:
                field_present = True
                break
        if not field_present:
            missing_fields.append(field)

    # Only require the essential fields: fixtureId, fixtureStart
    essential_fields = ['fixtureId', 'fixtureStart']
    essential_missing = [f for f in essential_fields if f in missing_fields]

    # For upcoming matches, we only need the fixture ID and start time
    # We'll be more lenient with player names since they might be in different fields
    if essential_missing:
        match_id = match.get('fixtureId', match.get('id', 'unknown'))
        logger.warning(f"Match {match_id} is missing essential fields: {', '.join(essential_missing)}")
        if missing_fields:
            logger.info(f"Match {match_id} is missing non-essential fields: {', '.join(missing_fields)}")
        return False

    # Log non-essential missing fields but still accept the match
    if missing_fields:
        match_id = match.get('fixtureId', match.get('id', 'unknown'))
        logger.info(f"Match {match_id} is missing non-essential fields: {', '.join(missing_fields)}")

    # For completed matches, check that scores are present
    if 'result' in match or 'matchEnded' in match:
        if 'homeScore' not in match or 'awayScore' not in match:
            match_id = match.get('fixtureId', match.get('id', 'unknown'))
            logger.warning(f"Completed match {match_id} is missing score information")
            return False

    return True


def validate_player_stats(player_stats):
    """
    Validate that a player stats dictionary contains all required fields.

    Args:
        player_stats (dict): Player stats dictionary

    Returns:
        bool: True if the player stats are valid, False otherwise
    """
    required_fields = [
        'player_name',
        'total_matches',
        'wins',
        'losses',
        'total_score',
        'win_rate',
        'avg_score',
        'teams_used',
        'opponents_faced'
    ]

    # Check that all required fields are present
    for field in required_fields:
        if field not in player_stats:
            return False

    return True
