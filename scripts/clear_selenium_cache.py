# tools/clear_selenium_cache.py
import os, shutil
from pathlib import Path

def possible_cache_dirs():
    dirs = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "selenium")
    else:
        # Linux & macOS typical
        dirs.append(Path.home() / ".cache" / "selenium")
        dirs.append(Path.home() / "Library" / "Caches" / "selenium")
    return [d for d in dirs if d.exists()]

if __name__ == "__main__":
    dirs = possible_cache_dirs()
    if not dirs:
        print("No Selenium cache directories found.")
    for d in dirs:
        print(f"Removing: {d}")
        shutil.rmtree(d, ignore_errors=True)
    print("Done. Re-run your script to let Selenium Manager re-resolve.")
