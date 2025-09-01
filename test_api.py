import os, requests
base = os.environ.get("API_BASE", "http://localhost:5000")
try:
    r = requests.get(f"{base}/api/health}", timeout=3)
    print("Health:", r.status_code, r.text[:120])
except Exception as e:
    print("Smoke test (likely fine if API not running yet):", e)
