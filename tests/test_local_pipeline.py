import requests
import json
import time

# Test the local pipeline endpoint to verify fixes
url = "http://localhost:5000/api/run-pipeline"
payload = {
    "return_predictions": True,
    "train_new_model": False,
    "history_days": 7  # Reduce to minimize processing time
}

print("🧪 Testing local pipeline endpoint...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    # Start the pipeline
    response = requests.post(url, json=payload, timeout=30)
    print(f"\n📤 Pipeline Start Response (Status {response.status_code}):")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        print("\n⏳ Waiting 30 seconds for pipeline to complete...")
        time.sleep(30)
        
        # Check pipeline results
        results_url = "http://localhost:5000/api/pipeline-results"
        results_response = requests.get(results_url, timeout=30)
        print(f"\n📥 Pipeline Results Response (Status {results_response.status_code}):")
        
        if results_response.status_code == 200:
            results = results_response.json()
            print(f"Pipeline Status: {results.get('status')}")
            print(f"Steps Completed: {results.get('steps_completed', [])}")
            print(f"Steps Failed: {results.get('steps_failed', [])}")
            print(f"Errors: {results.get('errors', [])}")
            
            # Check if database saves worked
            if 'errors' in results and results['errors']:
                print("\n❌ Found errors:")
                for error in results['errors']:
                    print(f"  - {error}")
            else:
                print("\n✅ No errors found!")
                
            predictions = results.get('predictions')
            if predictions:
                print(f"\n🎯 Predictions Summary:")
                print(f"Total Matches: {predictions.get('total_matches')}")
                print(f"Average Confidence: {predictions.get('average_confidence'):.3f}")
        else:
            print(f"Error getting results: {results_response.text}")
    else:
        print(f"Error starting pipeline: {response.text}")
            
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
except json.JSONDecodeError as e:
    print(f"❌ JSON decode error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
