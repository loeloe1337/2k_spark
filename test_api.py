"""
Minimal test API to verify deployment infrastructure.
"""
from fastapi import FastAPI

app = FastAPI(title="Test API")

@app.get("/")
def root():
    return {"message": "Test API is working!", "status": "healthy"}

@app.get("/test")
def test():
    return {"test": "successful", "deployment": "working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
