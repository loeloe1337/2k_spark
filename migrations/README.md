# Database Migrations

This directory contains the migration history for the 2K Flash database schema.

## Migration Files

- **`000_original_schema.sql`** - Original base schema (historical reference)
- **`001_prediction_system_enhancement.sql`** - Added ML model versioning tables (Applied: 2025-06-21)
- **`002_core_tables_player_enhancement.sql`** - Added player columns to core tables (Applied: 2025-06-21)

## Usage

The main schema file is in the root directory (`../schema.sql`). These migration files are for:

1. **Historical Reference** - Track what changes were made and when
2. **Development** - Apply incremental changes during development
3. **Documentation** - Understand the evolution of the database schema

## Current State

All migrations have been applied to the Supabase database and are reflected in the main `schema.sql` file.

## Adding New Migrations

When making schema changes:

1. Create a new migration file: `003_description_of_changes.sql`
2. Apply the migration to the database
3. Update the main `schema.sql` file with the complete current schema
4. Document the changes in this README
