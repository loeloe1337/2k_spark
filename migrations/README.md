# Database Migrations

This directory contains the migration history and data migration tools for the 2K Flash database schema.

## Files

### `migrate_data.py`
A comprehensive data migration script that transfers existing data from JSON files to the Supabase database. This script:
- Migrates match history from `output/match_history.json`
- Migrates player statistics from `output/player_stats.json`
- Migrates upcoming matches from `output/upcoming_matches.json`
- Handles data transformation and validation
- Provides detailed logging and error handling

**Usage:**
```bash
# From the project root directory
python migrations/migrate_data.py
```

**Requirements:**
- Supabase environment variables configured (`SUPABASE_URL`, `SUPABASE_KEY`)
- Existing JSON data files in the `output/` directory
- Database schema already applied (see `../schema.sql`)

## Current State

All migrations have been applied directly to the Supabase database and are reflected in the main `schema.sql` file in the root directory.

The migration files have been removed since:
1. All schema changes are now consolidated in the main `schema.sql` file
2. Database constraints have been applied directly using Supabase tools
3. The current schema is stable and production-ready

## Schema Management

- **Main Schema**: Use `../schema.sql` for the complete, current database schema
- **Database Constraints**: All necessary constraints have been applied to the production database
- **Future Changes**: New schema modifications should be:
  1. Applied directly to the database using Supabase SQL Editor or MCP tools
  2. Reflected in the main `schema.sql` file
  3. Documented in commit messages for version control

## Data Migration Workflow

1. **Setup Database Schema**: Ensure the database schema is applied using `../schema.sql`
2. **Configure Environment**: Set `SUPABASE_URL` and `SUPABASE_KEY` environment variables
3. **Prepare Data**: Ensure JSON data files exist in the `output/` directory
4. **Run Migration**: Execute `python migrations/migrate_data.py`
5. **Verify Results**: Check Supabase dashboard or use API endpoints to confirm data transfer

### Troubleshooting

- **Connection Issues**: Verify Supabase environment variables are correctly set
- **Missing Files**: Ensure JSON data files exist in the expected locations
- **Data Format**: The script handles various JSON formats and provides detailed error logs
- **Partial Failures**: Each data type (matches, stats, upcoming) is migrated independently

## Applied Changes History

The following migrations were previously applied and are now part of the main schema:

- **Base Schema** - Core tables for matches, player stats, and upcoming matches
- **ML Model Versioning** - Added model registry and prediction tracking tables
- **Player Enhancements** - Added player columns to core tables
- **Database Constraints** - Added unique constraints for proper data integrity

All tables now have proper primary keys and unique constraints for optimal performance and data integrity.
