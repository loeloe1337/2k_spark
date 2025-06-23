# Tests

This directory contains test scripts for the 2K Spark NBA eSports Backend.

## Test Files

### `test_local_pipeline.py`
Tests the full pipeline functionality when running the backend locally on `localhost:5000`. This script:
- Triggers the `/api/run-pipeline` endpoint with reduced parameters for faster testing
- Waits for completion and checks results via `/api/pipeline-results`
- Validates that all pipeline steps complete successfully
- Checks for database storage errors
- Displays prediction summaries if available

### `test_deployed_pipeline.py`
Tests the full pipeline functionality on the deployed Render instance at `https://twok-spark.onrender.com`. This script:
- Triggers the `/api/run-pipeline` endpoint on the production deployment
- Waits longer for completion due to cold starts and network latency
- Validates end-to-end functionality in the production environment
- Displays detailed prediction results

## Running Tests

Before running tests, ensure you have the required dependencies:
```bash
pip install requests
```

### Local Testing
1. Start the backend server locally:
   ```bash
   python -m backend.app.api
   ```
2. Run the local test:
   ```bash
   python tests/test_local_pipeline.py
   ```

### Deployed Testing
Simply run:
```bash
python tests/test_deployed_pipeline.py
```

## Expected Output

Both tests should show:
- Successful pipeline initiation (200 status)
- Pipeline completion status
- List of completed/failed steps
- Any errors encountered
- Prediction summaries with match forecasts

The tests validate that:
- Token fetching works (including Selenium automation)
- Data fetching completes successfully
- Player statistics are processed
- Predictions are generated
- All data is properly stored in Supabase
