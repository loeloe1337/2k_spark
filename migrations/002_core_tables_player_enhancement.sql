-- Migration: Core Tables Enhancement for Players
-- Applied: 2025-06-21
-- Description: Add player columns to core tables for enhanced prediction system

-- Add player columns to matches table
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='matches' AND column_name='home_player') THEN
        ALTER TABLE matches ADD COLUMN home_player VARCHAR(200);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='matches' AND column_name='away_player') THEN
        ALTER TABLE matches ADD COLUMN away_player VARCHAR(200);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='matches' AND column_name='tournament_name') THEN
        ALTER TABLE matches ADD COLUMN tournament_name VARCHAR(200);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='matches' AND column_name='result') THEN
        ALTER TABLE matches ADD COLUMN result VARCHAR(20);
    END IF;
END $$;

-- Add player columns to upcoming_matches table
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='upcoming_matches' AND column_name='home_player') THEN
        ALTER TABLE upcoming_matches ADD COLUMN home_player VARCHAR(200);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='upcoming_matches' AND column_name='away_player') THEN
        ALTER TABLE upcoming_matches ADD COLUMN away_player VARCHAR(200);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='upcoming_matches' AND column_name='tournament_name') THEN
        ALTER TABLE upcoming_matches ADD COLUMN tournament_name VARCHAR(200);
    END IF;
END $$;

-- Add enhanced columns to player_stats table
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='player_stats' AND column_name='player_external_id') THEN
        ALTER TABLE player_stats ADD COLUMN player_external_id VARCHAR(100);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='player_stats' AND column_name='recent_form') THEN
        ALTER TABLE player_stats ADD COLUMN recent_form JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='player_stats' AND column_name='head_to_head_stats') THEN
        ALTER TABLE player_stats ADD COLUMN head_to_head_stats JSONB;
    END IF;
END $$;

-- Add additional indexes for performance with new columns
CREATE INDEX IF NOT EXISTS idx_matches_players ON matches(home_player, away_player);
CREATE INDEX IF NOT EXISTS idx_upcoming_matches_players ON upcoming_matches(home_player, away_player);
CREATE INDEX IF NOT EXISTS idx_player_stats_external_id ON player_stats(player_external_id);
