
# scripts/driver_info.py
import os, selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_caps_value(caps, *keys, default=None):
    cur = caps
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
    return cur if cur is not None else default

def main():
    opts = Options()
    # Headless/new is stable. Required when running as root or in CI.
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    # Optional: pin a non-standard Chrome binary
    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        opts.binary_location = chrome_bin

    # Fallback switch: USE_WDM=1 forces webdriver-manager path
    use_wdm = os.environ.get("USE_WDM") == "1"
    if use_wdm:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)  # Selenium Manager resolves driver

    caps = driver.capabilities or {}

    browser_version = get_caps_value(caps, "browserVersion") or get_caps_value(caps, "version")
    # ChromeDriver version can appear in different places across platforms
    cdv = (
        get_caps_value(caps, "chrome", "chromedriverVersion")
        or get_caps_value(caps, "goog:chromeOptions", "debuggerAddress")  # placeholder path; not version
        or get_caps_value(caps, "chromedriverVersion")
    )
    if isinstance(cdv, str) and " " in cdv:
        cdv = cdv.split(" ")[0]

    print(f"Selenium: {selenium.__version__}")
    print(f"Chrome (browser) version: {browser_version}")
    print(f"ChromeDriver version: {cdv}")
    driver.quit()

if __name__ == "__main__":
    main()
