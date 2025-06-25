# 2K Spark - Render Deployment Guide

## Pre-Deployment Checklist ✅

All major issues have been identified and fixed:

- ✅ Port configuration updated for Render compatibility
- ✅ Chrome/Selenium optimized for container environments
- ✅ Job system enhanced with database fallback
- ✅ Service initialization error handling added
- ✅ Container environment detection implemented
- ✅ Enhanced health checks with component status
- ✅ Requirements.txt updated with version constraints
- ✅ Dockerfile improved for better Chrome support
- ✅ API startup test passes completely

## Render Deployment Steps

### 1. Repository Setup
1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Ensure all files are committed, especially the fixes applied

### 2. Create Render Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name:** `2k-spark-api` (or your preferred name)
   - **Environment:** `Python 3`
   - **Region:** Choose closest to your users
   - **Branch:** `main` (or your default branch)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app.api:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables
Set these in the Render dashboard under "Environment":

**Required:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

**Recommended:**
```
SELENIUM_HEADLESS=true
SELENIUM_TIMEOUT=30
CORS_ORIGINS=*
UPCOMING_MATCHES_DAYS=30
```

### 4. Health Check Configuration
- **Health Check Path:** `/api/health`
- **Port:** Use default (Render auto-configures)

### 5. Deploy
1. Click "Create Web Service"
2. Wait for the build and deployment to complete
3. Monitor the logs for any issues

## Post-Deployment Testing

### 1. Basic Health Check
```bash
curl https://your-service-name.onrender.com/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-25T...",
  "service": "2K Flash API",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "data_service": "healthy",
    "prediction_service": "healthy",
    "database": "connected"
  }
}
```

### 2. System Status Check
```bash
curl https://your-service-name.onrender.com/api/system-status
```

### 3. Run Data Pipeline
```bash
curl -X POST https://your-service-name.onrender.com/api/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "train_new_model": false,
    "refresh_token": false,
    "history_days": 30,
    "return_predictions": true
  }'
```

Expected response:
```json
{
  "status": "started",
  "job_id": "uuid-string",
  "job_type": "data_pipeline",
  "message": "Pipeline job created and started...",
  "estimated_duration": "5-15 minutes",
  "monitoring": {
    "job_status_url": "/api/jobs/{job_id}",
    "job_list_url": "/api/jobs"
  }
}
```

### 4. Check Job Status
```bash
curl https://your-service-name.onrender.com/api/jobs/{job_id}
```

### 5. Get Predictions
```bash
curl https://your-service-name.onrender.com/api/ml/predictions
```

## Troubleshooting

### Common Issues and Solutions

**1. Service Won't Start**
- Check logs in Render dashboard
- Verify all environment variables are set
- Ensure build command completed successfully

**2. Database Connection Issues**
- Verify Supabase URL and keys are correct
- Check if Supabase project is active
- Review network access settings in Supabase

**3. Selenium/Chrome Issues**
- Check if Chrome installation completed in build logs
- Verify SELENIUM_HEADLESS=true is set
- Check for memory/timeout issues in logs

**4. Job System Issues**
- Jobs will use local storage fallback if database is unavailable
- Check job status endpoints for error details
- Monitor system health endpoint

### Log Monitoring
Monitor these log patterns in Render:

**Success Patterns:**
- ✅ "All services initialized successfully"
- ✅ "Supabase client initialized successfully"
- ✅ "Job handlers registered successfully"

**Warning Patterns (Non-Critical):**
- ⚠️ "Database unavailable - using local storage fallback"
- ⚠️ "Supabase service not available"
- ⚠️ "Failed to register some job handlers"

**Error Patterns (Critical):**
- ❌ "Error initializing services"
- ❌ "Import Error"
- ❌ "Failed to start server"

## API Endpoints Summary

**Core Endpoints:**
- `GET /api/health` - Service health check
- `GET /api/system-status` - Detailed system status
- `GET /` - API information and endpoint list

**Data Endpoints:**
- `GET /api/upcoming-matches` - Get upcoming matches
- `GET /api/player-stats` - Get player statistics
- `POST /api/run-pipeline` - Run complete data pipeline
- `GET /api/pipeline-results` - Get pipeline results

**Machine Learning Endpoints:**
- `GET /api/ml/predictions` - Get match predictions
- `POST /api/ml/train` - Train new model
- `GET /api/ml/models` - List model versions

**Job Management:**
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/jobs/{job_id}/cancel` - Cancel job

## Performance Considerations

**Render Free Tier Limitations:**
- 512MB RAM limit
- Shared CPU
- Service sleeps after 15 minutes of inactivity
- 500 build minutes per month

**Optimization Applied:**
- Reduced Selenium timeouts for containers
- Local storage fallback for database operations
- Background job system to prevent request timeouts
- Memory-optimized Chrome options

**Scaling Options:**
- Upgrade to paid Render plan for more resources
- Use horizontal scaling for high traffic
- Consider Redis for job queue in production

## Next Steps

1. **Deploy to Render** using the configuration above
2. **Test all endpoints** to ensure functionality
3. **Set up monitoring** using Render's built-in tools
4. **Configure alerts** for health check failures
5. **Plan for production** data backup and model persistence

## Support

If you encounter issues during deployment:

1. Check the error analysis document: `Docs/error_analysis_and_fixes.md`
2. Review Render logs for specific error messages
3. Test locally using the startup test: `python test_startup.py`
4. Verify all environment variables are configured correctly

The system is now ready for production deployment on Render! 🚀
