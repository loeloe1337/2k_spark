-- Supabase Database Schema for 2K Flash Application
-- This script creates the necessary tables for storing NBA match data

-- Enable Row Level Security
-- Note: RLS policies will be configured in Supabase dashboard

-- Matches table for historical match data
CREATE TABLE IF NOT EXISTS matches (
    id BIGSERIAL PRIMARY KEY,
    match_id VARCHAR(100) UNIQUE NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    match_date TIMESTAMPTZ,
    tournament_id INTEGER,
    status VARCHAR(50) DEFAULT 'completed',
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Player statistics table
CREATE TABLE IF NOT EXISTS player_stats (
    id BIGSERIAL PRIMARY KEY,
    player_name VARCHAR(200) UNIQUE NOT NULL,
    team VARCHAR(100),
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0.00,
    total_score INTEGER DEFAULT 0,
    avg_score DECIMAL(8,2) DEFAULT 0.00,
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
    scheduled_date TIMESTAMPTZ,
    tournament_id INTEGER,
    status VARCHAR(50) DEFAULT 'scheduled',
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date DESC);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_player_stats_win_rate ON player_stats(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_upcoming_matches_date ON upcoming_matches(scheduled_date ASC);

-- Create function to automatically update the updated_at timestamp
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

-- Insert some sample data for testing (optional)
-- This can be removed after initial setup

INSERT INTO matches (match_id, home_team, away_team, home_score, away_score, match_date, tournament_id, status) 
VALUES 
    ('sample_1', 'Lakers', 'Warriors', 112, 108, '2025-01-15 20:00:00', 1, 'completed'),
    ('sample_2', 'Celtics', 'Heat', 95, 89, '2025-01-16 19:30:00', 1, 'completed')
ON CONFLICT (match_id) DO NOTHING;

INSERT INTO player_stats (player_name, team, games_played, wins, losses, win_rate, total_score, avg_score)
VALUES 
    ('LeBron James', 'Lakers', 45, 30, 15, 66.67, 1350, 30.00),
    ('Stephen Curry', 'Warriors', 42, 28, 14, 66.67, 1260, 30.00)
ON CONFLICT (player_name) DO NOTHING;

INSERT INTO upcoming_matches (match_id, home_team, away_team, scheduled_date, tournament_id, status)
VALUES 
    ('upcoming_1', 'Nuggets', 'Suns', '2025-06-22 21:00:00', 1, 'scheduled'),
    ('upcoming_2', 'Knicks', 'Nets', '2025-06-23 20:00:00', 1, 'scheduled')
ON CONFLICT (match_id) DO NOTHING;
