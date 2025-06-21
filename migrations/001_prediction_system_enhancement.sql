-- Migration: Enhanced Prediction System Tables
-- Applied: 2025-06-21
-- Description: Add model versioning and prediction tracking tables

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

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(is_active, model_name);
CREATE INDEX IF NOT EXISTS idx_match_predictions_model ON match_predictions(model_version, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_match_predictions_match ON match_predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_feature_importance_model ON feature_importance(model_version, importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_version ON model_performance(model_version, evaluation_date DESC);

-- Add updated_at trigger for match_predictions
CREATE TRIGGER update_match_predictions_updated_at
    BEFORE UPDATE ON match_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

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
