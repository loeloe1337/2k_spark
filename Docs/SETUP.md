# 2K Flash - Docker and Supabase Setup Guide

This guide will help you set up the 2K Flash application with Docker and Supabase database integration.

## Prerequisites

1. **Docker Desktop** - Download and install from [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Supabase Account** - Sign up at [supabase.com](https://supabase.com)

## Phase 1: Docker Setup

### 1. Start Docker Desktop
Make sure Docker Desktop is running on your system.

### 2. Build and Run the Application
```bash
# Navigate to the project directory
cd /path/to/2k_spark

# Build and start the services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Verify the Application
- **API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health
- **API Documentation**: http://localhost:5000/docs

## Phase 2: Supabase Setup

### 1. Create a Supabase Project
1. Go to [supabase.com](https://supabase.com) and sign in
2. Click "New Project"
3. Choose your organization and enter project details
4. Wait for the project to be created (1-2 minutes)

### 2. Get Your Project Credentials
1. Go to your project dashboard
2. Click on "Settings" in the sidebar
3. Go to "API" section
4. Copy the following values:
   - **Project URL** (something like `https://xxxxx.supabase.co`)
   - **Anon public key** (starts with `eyJ...`)
   - **Service role key** (starts with `eyJ...`)

### 3. Set Up Database Schema
1. In your Supabase dashboard, go to "SQL Editor"
2. Create a new query
3. Copy and paste the contents of `schema.sql` from this project
4. Click "Run" to create the tables

### 4. Configure Environment Variables

#### Option A: Using .env file (Recommended for local development)
1. Copy the template: `cp .env.template .env`
2. Edit `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_KEY=your-anon-public-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ```

#### Option B: Update docker-compose.yml
1. Edit `docker-compose.yml`
2. Uncomment and update the Supabase environment variables:
   ```yaml
   environment:
     - SUPABASE_URL=https://your-project-ref.supabase.co
     - SUPABASE_KEY=your-anon-public-key
     - SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ```

### 5. Restart the Application
```bash
# Stop the current containers
docker-compose down

# Start with the new configuration
docker-compose up -d
```

## Phase 3: Data Migration

### 1. Migrate Existing Data (Optional)
If you have existing JSON data files, you can migrate them to the database:

```bash
# Run the migration script
python migrate_data.py
```

### 2. Verify Database Integration
1. Check the API endpoints:
   - http://localhost:5000/api/upcoming-matches
   - http://localhost:5000/api/player-stats
   - http://localhost:5000/api/live-scores

2. The API will now:
   - Try to fetch data from Supabase database first
   - Fall back to JSON files if database is unavailable
   - Save new data to both database and files

## Troubleshooting

### Docker Issues
```bash
# Check if Docker is running
docker --version

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose down
docker-compose up --build
```

### Supabase Connection Issues
1. Verify your credentials in the environment variables
2. Check that your Supabase project is active
3. Ensure the database schema was created correctly
4. Check the application logs: `docker-compose logs backend`

### Common Issues

#### 1. "Supabase client not initialized"
- Check that `SUPABASE_URL` and `SUPABASE_KEY` are set correctly
- Verify the credentials are valid in your Supabase dashboard

#### 2. "Database connection test failed"
- Ensure your Supabase project is not paused
- Check your internet connection
- Verify the project URL is correct

#### 3. "Docker build failed"
- Make sure Docker Desktop is running
- Check that all required files are present
- Try cleaning Docker cache: `docker system prune`

## Development Workflow

### 1. Local Development
- Code changes are automatically reflected (volume mounts)
- Database changes persist between container restarts
- Logs are available in the `logs/` directory

### 2. Adding New Features
1. Update the database schema in `schema.sql` if needed
2. Apply schema changes in Supabase SQL editor
3. Update the Python code
4. Test with `docker-compose restart backend`

### 3. Database Management
- Use Supabase dashboard for database administration
- Data is automatically backed up by Supabase
- You can export data from the dashboard if needed

## Next Steps

1. **Configure Row Level Security (RLS)** in Supabase for production
2. **Set up authentication** if you plan to add user accounts
3. **Configure real-time subscriptions** for live data updates
4. **Set up monitoring** using Supabase analytics

## Support

- **Supabase Documentation**: https://supabase.com/docs
- **Docker Documentation**: https://docs.docker.com
- **Project Issues**: Create an issue in the repository
