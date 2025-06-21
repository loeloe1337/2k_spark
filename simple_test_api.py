"""
Minimal API server for testing Render deployment.
"""

from datetime import datetime
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI(title="2K Flash API - Minimal", description="Minimal API for testing deployment", version="1.0.0")

@app.get('/')
def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        dict: API information
    """
    return {
        "message": "2K Flash API - Minimal Version",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/api/health",
            "test": "/api/test"
        }
    }

@app.get('/api/health')
def health_check():
    """
    Health check endpoint for monitoring and Docker health checks.
    
    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "2K Flash API - Minimal",
        "version": "1.0.0"
    }

@app.get('/api/test')
def test_endpoint():
    """
    Simple test endpoint.
    
    Returns:
        dict: Test response
    """
    return {
        "message": "Test endpoint working!",
        "timestamp": datetime.now().isoformat(),
        "deployment": "render",
        "status": "success"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
