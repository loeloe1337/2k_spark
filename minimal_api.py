"""
Minimal API for testing deployment issues.
"""

from fastapi import FastAPI
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="2K Flash API", description="Minimal API for testing", version="1.0.0")

@app.get('/')
def root():
    """Root endpoint"""
    return {
        "message": "2K Flash API - Minimal Version",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "online"
    }

@app.get('/api/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "2K Flash API - Minimal",
        "version": "1.0.0"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
