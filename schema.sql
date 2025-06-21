-- Complete Supabase Database Schema for 2K Flash Application
-- This script creates all necessary tables for NBA match data and ML model versioning

-- Enable Row Level Security (configure policies in Supabase dashboard as needed)

-- ============================================================================
-- CORE DATA TABLES
-- ============================================================================

-- Matches table for historical match data
CREATE TABLE IF NOT EXISTS matches (
    id BIGSERIAL PRIMARY KEY,
    match_id VARCHAR(100) UNIQUE NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_player VARCHAR(200),
    away_player VARCHAR(200),
    home_score INTEGER,
    away_score INTEGER,
    match_date TIMESTAMPTZ,
    tournament_id INTEGER,
    tournament_name VARCHAR(200),
    status VARCHAR(50) DEFAULT 'completed',
    result VARCHAR(20), -- 'home_win', 'away_win'
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Player statistics table
CREATE TABLE IF NOT EXISTS player_stats (
    id BIGSERIAL PRIMARY KEY,
    player_name VARCHAR(200) UNIQUE NOT NULL,
    player_external_id VARCHAR(100),
    team VARCHAR(100),
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0.00,
    total_score INTEGER DEFAULT 0,
    avg_score DECIMAL(8,2) DEFAULT 0.00,
    recent_form JSONB, -- Last 5-10 games performance
    head_to_head_stats JSONB, -- H2H records against other players
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Upcoming matches table
CREATE TABLE IF NOT EXISTS upcoming_matches (
    id BIGSERIAL PRIMARY KEY,
    match_id VARCHAR(100) UNIQUE NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_player VARCHAR(200),
    away_player VARCHAR(200),
    scheduled_date TIMESTAMPTZ,
    tournament_id INTEGER,
    tournament_name VARCHAR(200),
    status VARCHAR(50) DEFAULT 'scheduled',
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- MACHINE LEARNING MODEL VERSIONING TABLES
-- ============================================================================

-- Model registry table for managing multiple model versions
CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'unified_prediction',
    training_date TIMESTAMPTZ DEFAULT NOW(),
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    model_path TEXT,
    feature_count INTEGER,
    training_samples INTEGER,
    validation_accuracy DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);

-- Match predictions table to store and track prediction results
CREATE TABLE IF NOT EXISTS match_predictions (
    id BIGSERIAL PRIMARY KEY,
    match_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    home_player VARCHAR(200) NOT NULL,
    away_player VARCHAR(200) NOT NULL,
    predicted_home_score DECIMAL(6,2),
    predicted_away_score DECIMAL(6,2),
    predicted_total_score DECIMAL(6,2),
    predicted_winner VARCHAR(200),
    home_win_probability DECIMAL(5,4),
    confidence_score DECIMAL(5,4),
    prediction_date TIMESTAMPTZ DEFAULT NOW(),
    match_start_time TIMESTAMPTZ,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_winner VARCHAR(200),
    prediction_correct BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature importance table to track which features matter most
CREATE TABLE IF NOT EXISTS feature_importance (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    importance_score DECIMAL(8,6),
    feature_rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model performance tracking table
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMPTZ DEFAULT NOW(),
    test_period_start TIMESTAMPTZ,
    test_period_end TIMESTAMPTZ,
    winner_accuracy DECIMAL(5,4),
    home_score_mae DECIMAL(6,2),
    away_score_mae DECIMAL(6,2),
    total_score_mae DECIMAL(6,2),
    predictions_count INTEGER,
    correct_predictions INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Core data indexes
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date DESC);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_matches_players ON matches(home_player, away_player);
CREATE INDEX IF NOT EXISTS idx_player_stats_win_rate ON player_stats(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_upcoming_matches_date ON upcoming_matches(scheduled_date ASC);

-- ML model indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(is_active, model_name);
CREATE INDEX IF NOT EXISTS idx_match_predictions_model ON match_predictions(model_version, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_match_predictions_match ON match_predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_feature_importance_model ON feature_importance(model_version, importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_version ON model_performance(model_version, evaluation_date DESC);

-- ============================================================================
-- TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update the updated_at column
CREATE TRIGGER update_matches_updated_at
    BEFORE UPDATE ON matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_player_stats_updated_at
    BEFORE UPDATE ON player_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_upcoming_matches_updated_at
    BEFORE UPDATE ON upcoming_matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_match_predictions_updated_at
    BEFORE UPDATE ON match_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MODEL MANAGEMENT FUNCTIONS
-- ============================================================================

-- Function to set a model as active (deactivate others)
CREATE OR REPLACE FUNCTION set_active_model(model_name_param VARCHAR, version_param VARCHAR)
RETURNS VOID AS $$
BEGIN
    -- Deactivate all models with the same name
    UPDATE model_registry 
    SET is_active = FALSE 
    WHERE model_name = model_name_param;
    
    -- Activate the specified version
    UPDATE model_registry 
    SET is_active = TRUE 
    WHERE model_name = model_name_param AND model_version = version_param;
END;
$$ LANGUAGE plpgsql;

-- Function to get the active model version
CREATE OR REPLACE FUNCTION get_active_model_version(model_name_param VARCHAR)
RETURNS VARCHAR AS $$
DECLARE
    active_version VARCHAR;
BEGIN
    SELECT model_version INTO active_version
    FROM model_registry
    WHERE model_name = model_name_param AND is_active = TRUE
    LIMIT 1;
    
    RETURN active_version;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA FOR TESTING (Optional - remove in production)
-- ============================================================================

-- Sample model registry entry
INSERT INTO model_registry (model_name, model_version, performance_metrics, is_active, validation_accuracy, feature_count, training_samples)
VALUES 
    ('nba2k_match_predictor', 'v1.0.0', '{"val_winner_accuracy": 0.6547, "val_home_mae": 6.23, "val_away_mae": 6.31}', TRUE, 0.6547, 54, 5000)
ON CONFLICT (model_name, model_version) DO NOTHING;

-- Sample matches (can be removed after real data is populated)
INSERT INTO matches (match_id, home_team, away_team, home_player, away_player, home_score, away_score, match_date, tournament_id, tournament_name, result) 
VALUES 
    ('sample_1', 'Lakers', 'Warriors', 'PLAYER1', 'PLAYER2', 112, 108, '2025-01-15 20:00:00', 1, 'Ebasketball H2H GG League', 'home_win'),
    ('sample_2', 'Celtics', 'Heat', 'PLAYER3', 'PLAYER4', 95, 89, '2025-01-16 19:30:00', 1, 'Ebasketball H2H GG League', 'home_win')
ON CONFLICT (match_id) DO NOTHING;

-- Sample player stats
INSERT INTO player_stats (player_name, player_external_id, team, games_played, wins, losses, win_rate, total_score, avg_score)
VALUES 
    ('PLAYER1', 'ext_001', 'Lakers', 45, 30, 15, 66.67, 1350, 30.00),
    ('PLAYER2', 'ext_002', 'Warriors', 42, 28, 14, 66.67, 1260, 30.00)
ON CONFLICT (player_name) DO NOTHING;

-- Sample upcoming matches
INSERT INTO upcoming_matches (match_id, home_team, away_team, home_player, away_player, scheduled_date, tournament_id, tournament_name, status)
VALUES 
    ('upcoming_1', 'Nuggets', 'Suns', 'PLAYER5', 'PLAYER6', '2025-06-22 21:00:00', 1, 'Ebasketball H2H GG League', 'scheduled'),
    ('upcoming_2', 'Knicks', 'Nets', 'PLAYER7', 'PLAYER8', '2025-06-23 20:00:00', 1, 'Ebasketball H2H GG League', 'scheduled')
ON CONFLICT (match_id) DO NOTHING;
