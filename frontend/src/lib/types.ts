/**
 * Type definitions for 2K Spark API responses
 */

export interface Player {
  id: string;
  name: string;
  games_played: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_score: number;
  avg_opponent_score: number;
  recent_form?: {
    last_5_games: {
      wins: number;
      losses: number;
      win_rate: number;
    };
  };
  performance_metrics?: {
    consistency: number;
    momentum: number;
    variance: number;
  };
}

export interface Match {
  match_id: string;
  home_team: string;
  away_team: string;
  match_date: string;
  tournament?: string;
  round?: string;
}

export interface Prediction {
  match_id: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  home_win_probability: number;
  away_win_probability: number;
  confidence?: string;
  match_date: string;
}

export interface ScorePrediction {
  match_id: string;
  home_team: string;
  away_team: string;
  predicted_home_score: number;
  predicted_away_score: number;
  predicted_total_score: number;
  confidence_interval?: {
    home_lower: number;
    home_upper: number;
    away_lower: number;
    away_upper: number;
  };
  match_date: string;
}

export interface PredictionHistory {
  match_id: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  actual_winner?: string;
  predicted_home_score: number;
  predicted_away_score: number;
  actual_home_score?: number;
  actual_away_score?: number;
  correct_winner?: boolean;
  score_error?: number;
  match_date: string;
  validated_at?: string;
}

export interface Stats {
  winner_prediction: {
    accuracy: number;
    total_predictions: number;
    correct_predictions: number;
    model_version: string;
  };
  score_prediction: {
    mae: number;
    rmse: number;
    total_predictions: number;
    model_version: string;
  };
  last_updated: string;
}

export interface LiveMatch {
  match_id: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  status: string;
  time_remaining?: string;
  prediction?: {
    predicted_winner: string;
    confidence: number;
  };
}

// API Response wrapper types
export interface ApiListResponse<T> {
  [key: string]: T[] | number | string;
  total_count: number;
  last_updated: string;
}

export interface PredictionsResponse extends ApiListResponse<Prediction> {
  predictions: Prediction[];
}

export interface ScorePredictionsResponse extends ApiListResponse<ScorePrediction> {
  predictions: ScorePrediction[];
}

export interface MatchesResponse extends ApiListResponse<Match> {
  matches: Match[];
}

export interface PredictionHistoryResponse {
  predictions: PredictionHistory[];
  total_count: number;
  last_updated: string;
  accuracy?: number;
  avg_score_error?: number;
}

export interface LiveMatchesResponse extends ApiListResponse<LiveMatch> {
  matches: LiveMatch[];
}

export interface RefreshResponse {
  status: string;
  message: string;
  process_id?: string;
}
