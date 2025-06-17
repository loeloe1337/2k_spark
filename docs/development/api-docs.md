# API Documentation

## Overview

The 2K Spark API provides RESTful endpoints for accessing predictions, statistics, and match data. All endpoints return JSON responses and support CORS for frontend integration.

**Base URL**: `http://localhost:5000` (development) or your deployed backend URL

## Authentication

Currently, the API does not require authentication for read operations. All endpoints are publicly accessible.

## Endpoints

### 🎯 Predictions

#### Get Winner Predictions
```http
GET /api/predictions
```

Returns predictions for upcoming matches with winner probabilities.

**Response Example**:
```json
{
  "predictions": [
    {
      "match_id": "12345",
      "home_team": "PLAYER1",
      "away_team": "PLAYER2",
      "predicted_winner": "PLAYER1",
      "home_win_probability": 0.75,
      "away_win_probability": 0.25,
      "confidence": "high",
      "match_date": "2025-06-18 20:00:00"
    }
  ],
  "total_count": 1,
  "last_updated": "2025-06-17T10:30:00Z"
}
```

#### Get Score Predictions
```http
GET /api/score-predictions
```

Returns predicted scores for upcoming matches.

**Response Example**:
```json
{
  "predictions": [
    {
      "match_id": "12345",
      "home_team": "PLAYER1",
      "away_team": "PLAYER2",
      "predicted_home_score": 21,
      "predicted_away_score": 18,
      "predicted_total_score": 39,
      "confidence_interval": {
        "home_lower": 18,
        "home_upper": 24,
        "away_lower": 15,
        "away_upper": 21
      },
      "match_date": "2025-06-18 20:00:00"
    }
  ],
  "total_count": 1,
  "last_updated": "2025-06-17T10:30:00Z"
}
```

### 📊 Statistics

#### Get Prediction Statistics
```http
GET /api/stats
```

Returns overall prediction accuracy and performance metrics.

**Response Example**:
```json
{
  "winner_prediction": {
    "accuracy": 0.74,
    "total_predictions": 150,
    "correct_predictions": 111,
    "model_version": "v1.2.3"
  },
  "score_prediction": {
    "mae": 3.2,
    "rmse": 4.1,
    "total_predictions": 145,
    "model_version": "v1.1.8"
  },
  "last_updated": "2025-06-17T10:30:00Z"
}
```

#### Get Player Statistics
```http
GET /api/player-stats
```

Returns comprehensive player performance statistics.

**Response Example**:
```json
[
  {
    "id": "PLAYER1",
    "name": "PLAYER1",
    "games_played": 45,
    "wins": 34,
    "losses": 11,
    "win_rate": 0.756,
    "avg_score": 20.4,
    "avg_opponent_score": 17.8,
    "recent_form": {
      "last_5_games": {
        "wins": 4,
        "losses": 1,
        "win_rate": 0.8
      }
    },
    "performance_metrics": {
      "consistency": 0.82,
      "momentum": 0.15,
      "variance": 2.3
    }
  }
]
```

### 📅 Match Data

#### Get Upcoming Matches
```http
GET /api/upcoming-matches
```

Returns upcoming match schedule.

**Response Example**:
```json
{
  "matches": [
    {
      "match_id": "12345",
      "home_team": "PLAYER1",
      "away_team": "PLAYER2",
      "match_date": "2025-06-18 20:00:00",
      "tournament": "H2H GG League",
      "round": "Quarterfinals"
    }
  ],
  "total_count": 5,
  "last_updated": "2025-06-17T10:30:00Z"
}
```

#### Get Prediction History
```http
GET /api/prediction-history
```

Returns historical predictions with results.

**Query Parameters**:
- `limit` (optional): Number of results to return (default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `date_from` (optional): Start date filter (YYYY-MM-DD)
- `date_to` (optional): End date filter (YYYY-MM-DD)

**Response Example**:
```json
{
  "predictions": [
    {
      "match_id": "12344",
      "home_team": "PLAYER1",
      "away_team": "PLAYER2",
      "predicted_winner": "PLAYER1",
      "actual_winner": "PLAYER1",
      "predicted_home_score": 21,
      "predicted_away_score": 18,
      "actual_home_score": 22,
      "actual_away_score": 17,
      "correct_winner": true,
      "score_error": 1.4,
      "match_date": "2025-06-16 20:00:00",
      "validated_at": "2025-06-17T08:00:00Z"
    }
  ],
  "total_count": 150,
  "accuracy": 0.74,
  "avg_score_error": 3.2
}
```

### 🔄 System Operations

#### Refresh Data
```http
POST /api/refresh
```

Triggers a complete data refresh and prediction update.

**Response Example**:
```json
{
  "status": "success",
  "message": "Refresh process started",
  "process_id": "12345"
}
```

#### Get Live Scores
```http
GET /api/live-scores
```

Returns live match scores and updates.

**Response Example**:
```json
{
  "matches": [
    {
      "match_id": "12346",
      "home_team": "PLAYER1",
      "away_team": "PLAYER2",
      "home_score": 15,
      "away_score": 12,
      "status": "in_progress",
      "time_remaining": "3:45",
      "prediction": {
        "predicted_winner": "PLAYER1",
        "confidence": 0.75
      }
    }
  ],
  "total_count": 3,
  "last_updated": "2025-06-17T15:45:30Z"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error message description",
  "code": "ERROR_CODE",
  "timestamp": "2025-06-17T10:30:00Z"
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Endpoint or resource not found |
| 500 | Internal Server Error - Server-side error |

### Common Error Scenarios

#### Data Not Available
```json
{
  "error": "No prediction data available",
  "code": "NO_DATA",
  "timestamp": "2025-06-17T10:30:00Z"
}
```

#### System Refresh In Progress
```json
{
  "error": "System refresh in progress, try again later",
  "code": "REFRESH_IN_PROGRESS",
  "timestamp": "2025-06-17T10:30:00Z"
}
```

## Rate Limiting

Currently, no rate limiting is implemented, but it's recommended to:
- Limit requests to 1 per second for prediction endpoints
- Use the `last_updated` timestamp to avoid unnecessary requests
- Cache responses when appropriate

## CORS Support

The API supports Cross-Origin Resource Sharing (CORS) for frontend integration. The following origins are allowed by default:
- `http://localhost:3000` (development)
- Your production frontend domain

## Example Usage

### JavaScript/TypeScript
```typescript
// Fetch predictions
const response = await fetch('http://localhost:5000/api/predictions');
const data = await response.json();

// Handle errors
if (!response.ok) {
  console.error('API Error:', data.error);
  return;
}

console.log('Predictions:', data.predictions);
```

### Python
```python
import requests

# Fetch player statistics
response = requests.get('http://localhost:5000/api/player-stats')

if response.status_code == 200:
    players = response.json()
    print(f"Found {len(players)} players")
else:
    print(f"Error: {response.status_code}")
```

### cURL
```bash
# Get predictions
curl -X GET "http://localhost:5000/api/predictions"

# Trigger refresh
curl -X POST "http://localhost:5000/api/refresh"

# Get stats with pretty formatting
curl -X GET "http://localhost:5000/api/stats" | jq
```

## WebSocket Support (Future)

Planned features for real-time updates:
- Live match score updates
- Real-time prediction changes
- System status notifications

---

**Last Updated**: June 17, 2025  
**API Version**: 1.0.0
