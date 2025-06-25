"""
Simple API startup test to verify there are no critical import or initialization errors.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir / "backend"
sys.path.insert(0, str(backend_dir))

def test_api_startup():
    """Test that the API can be imported and initialized without errors."""
    try:
        print("Testing API imports...")
        
        # Test basic config imports
        from config.settings import API_HOST, API_PORT
        print(f"✅ Config loaded - API will run on {API_HOST}:{API_PORT}")
        
        # Test service imports
        from services.supabase_service import SupabaseService
        print("✅ Supabase service import successful")
        
        from services.job_service import JobService
        print("✅ Job service import successful")
        
        # Test core imports
        from core.data.fetchers import TokenFetcher
        print("✅ Token fetcher import successful")
        
        # Test API import (this is the critical test)
        print("Testing FastAPI app import...")
        from app.api import app
        print("✅ FastAPI app imported successfully")
        
        # Test health endpoint
        print("Testing health check endpoint...")
        from app.api import health_check
        health_status = health_check()
        print(f"✅ Health check works: {health_status.get('status', 'unknown')}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The API should start successfully on Render.")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("This indicates a missing dependency or incorrect import path.")
        return False
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        print("This indicates a runtime issue that needs to be fixed.")
        return False

if __name__ == "__main__":
    print("2K Spark API Startup Test")
    print("=" * 50)
    
    success = test_api_startup()
    
    if success:
        print("\n✅ Ready for Render deployment!")
        print("\nNext steps:")
        print("1. Push code to Git repository")
        print("2. Deploy to Render with the provided configuration")
        print("3. Set the required environment variables")
        print("4. Test the health endpoint: /api/health")
    else:
        print("\n❌ Fix the errors above before deploying to Render")
        sys.exit(1)
