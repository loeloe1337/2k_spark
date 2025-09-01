# scripts/launch_chrome.py
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def main():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        opts.binary_location = chrome_bin

    if os.environ.get("USE_WDM") == "1":
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    else:
        driver = webdriver.Chrome(options=opts)

    driver.get("https://example.org/")
    print("Title:", driver.title)
    driver.quit()
    print("OK: Selenium ran headless Chrome and fetched a page.")

if __name__ == "__main__":
    main()
