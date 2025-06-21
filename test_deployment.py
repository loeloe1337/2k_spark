#!/usr/bin/env python3
"""
Test script to validate the deployed 2K Spark API on Render.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "https://twok-spark.onrender.com"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test a single endpoint and return the result."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "response": response.text[:500] if response.text else "No content",
            "headers": dict(response.headers)
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def main():
    """Run comprehensive tests on the deployed API."""
    print("🚀 Testing 2K Spark API Deployment on Render")
    print("=" * 60)
    
    # Test endpoints
    endpoints = [
        {"path": "/", "method": "GET", "description": "Root endpoint"},
        {"path": "/api/health", "method": "GET", "description": "Health check"},
        {"path": "/api/system/status", "method": "GET", "description": "System status"},
        {"path": "/api/upcoming-matches", "method": "GET", "description": "Upcoming matches"},
        {"path": "/api/data/player-stats", "method": "GET", "description": "Player stats"},
        {"path": "/api/run-pipeline", "method": "POST", "description": "Run pipeline", 
         "data": {"include_prediction": True, "save_results": True}}
    ]
    
    results = {}
    
    for endpoint in endpoints:
        print(f"\n📋 Testing: {endpoint['description']}")
        print(f"   Endpoint: {endpoint['method']} {endpoint['path']}")
        
        result = test_endpoint(
            endpoint['path'], 
            endpoint['method'], 
            endpoint.get('data')
        )
        
        results[endpoint['path']] = result
        
        if result['status'] == 'success':
            print(f"   ✅ Status: {result['status_code']}")
            if result['status_code'] == 200:
                print(f"   📄 Response: {result['response'][:100]}...")
            else:
                print(f"   ⚠️  Response: {result['response']}")
        else:
            print(f"   ❌ Error: {result['message']}")
        
        # Wait between requests to be respectful
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    successful = sum(1 for r in results.values() if r['status'] == 'success' and r.get('status_code') == 200)
    total = len(results)
    
    print(f"   ✅ Successful: {successful}/{total}")
    print(f"   ❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print("\n🎉 All tests passed! Deployment is successful!")
    else:
        print("\n⚠️  Some tests failed. Check the deployment logs.")
        
    # Detailed results
    print("\n📋 Detailed Results:")
    for path, result in results.items():
        status_icon = "✅" if result['status'] == 'success' and result.get('status_code') == 200 else "❌"
        print(f"   {status_icon} {path}: {result}")

if __name__ == "__main__":
    main()
