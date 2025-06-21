# Supabase Setup Guide for 2K Flash

This guide will walk you through setting up Supabase for your 2K Flash NBA analytics application.

## Prerequisites

- Docker and Docker Compose installed and running
- A Supabase account (sign up at https://supabase.com)

## Step 1: Create a New Supabase Project

1. Go to https://supabase.com and sign in to your account
2. Click "New Project"
3. Choose your organization
4. Fill in project details:
   - **Name**: `2k-flash-nba` (or your preferred name)
   - **Database Password**: Choose a strong password
   - **Region**: Select the region closest to you
5. Click "Create new project"

## Step 2: Get Your Project Credentials

Once your project is created, you'll need to collect the following information:

### From Project Settings > API:
- **Project URL**: Found under "Project URL" (format: `https://your-project-ref.supabase.co`)
- **Anon Key**: Found under "Project API keys" > "anon public" 
- **Service Role Key**: Found under "Project API keys" > "service_role" (keep this secret!)

### Your Project Reference ID:
From your project URL, the reference ID is the subdomain. For example:
- URL: `https://sjlqonaqpvnqonmsnabk.supabase.co`
- Reference ID: `sjlqonaqpvnqonmsnabk`

## Step 3: Set Up Database Schema

1. In your Supabase dashboard, go to **SQL Editor**
2. Click "New Query"
3. Copy and paste the contents of `schema.sql` from your project root
4. Click "Run" to execute the SQL commands

This will create the necessary tables:
- `matches` - For historical match data
- `player_stats` - For player statistics
- `upcoming_matches` - For scheduled matches

## Step 4: Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file and update the Supabase configuration:
   ```bash
   # Supabase Configuration
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_KEY=your-anon-public-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ```

3. Update your `docker-compose.yml` to use the environment file:
   ```yaml
   services:
     backend:
       # ... other configuration ...
       env_file:
         - .env
       # ... rest of configuration ...
   ```

## Step 5: Configure Row Level Security (RLS)

For production use, enable Row Level Security:

1. In Supabase dashboard, go to **Authentication** > **Policies**
2. For each table (`matches`, `player_stats`, `upcoming_matches`):
   - Click "New Policy" 
   - Choose "Enable read access for all users" for public read access
   - Or create custom policies based on your requirements

## Step 6: Test Database Connection

1. Restart your Docker containers:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

2. Check the logs to verify successful connection:
   ```bash
   docker-compose logs -f backend
   ```

   You should see: `"Supabase client initialized successfully"`

3. Test the API endpoints:
   ```bash
   # Test health check
   curl http://localhost:5000/api/health
   
   # Test database connectivity (should return empty arrays initially)
   curl http://localhost:5000/api/upcoming-matches
   curl http://localhost:5000/api/player-stats
   ```

## Step 7: Migrate Existing Data (Optional)

If you have existing JSON data files, you can migrate them to Supabase:

1. Ensure your `.env` file is configured correctly
2. Run the migration script:
   ```bash
   docker-compose exec backend python /app/migrate_data.py
   ```

## Step 8: Verify Setup

1. **Check Database Tables**: In Supabase dashboard > **Table Editor**, verify your tables exist and contain data
2. **Test API Endpoints**: Make requests to verify data is being served from the database
3. **Monitor Logs**: Check both application logs and Supabase logs for any issues

## Troubleshooting

### Common Issues:

1. **"Supabase URL or Key not configured"**
   - Verify your `.env` file exists and contains correct values
   - Restart Docker containers after changing environment variables

2. **"Failed to initialize Supabase client"**
   - Check that your SUPABASE_URL and SUPABASE_KEY are correct
   - Verify your project is active in Supabase dashboard

3. **Database connection errors**
   - Ensure your database is not paused (free tier projects auto-pause)
   - Check your network connection and firewall settings

4. **Import/permission errors**
   - Verify RLS policies allow the operations you're trying to perform
   - Check that you're using the correct API key (anon vs service_role)

### Useful Commands:

```bash
# View logs
docker-compose logs -f backend

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up --build -d

# Access container shell for debugging
docker-compose exec backend bash
```

## Next Steps

Once Supabase is configured:

1. **Data Pipeline**: Set up automated data fetching and processing
2. **Real-time Updates**: Configure webhooks for live score updates
3. **Analytics Dashboard**: Build a frontend to visualize the data
4. **Performance Monitoring**: Set up logging and monitoring for production use

## Security Considerations

- Never commit your `.env` file to version control
- Use the `service_role` key only for server-side operations
- Enable RLS policies for production deployments
- Regularly rotate your API keys
- Monitor usage in Supabase dashboard

For more detailed information, refer to the [Supabase Documentation](https://supabase.com/docs).
