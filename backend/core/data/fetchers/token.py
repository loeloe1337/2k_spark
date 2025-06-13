"""Token fetcher for the H2H GG League API."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError:
    print("Error: selenium library not found. Install with: pip install selenium")
    print("Also ensure Chrome/Chromium browser is installed")
    exit(1)

from config.settings import H2H_WEBSITE_URL, H2H_TOKEN_LOCALSTORAGE_KEY, SELENIUM_HEADLESS, SELENIUM_TIMEOUT
from config.logging_config import get_data_fetcher_logger
from utils.logging import log_execution_time, log_exceptions

logger = get_data_fetcher_logger()


class TokenFetcher:
    """
    Fetches authentication token from H2H GG League website using Selenium.
    """
    
    def __init__(self, website_url=H2H_WEBSITE_URL, token_key=H2H_TOKEN_LOCALSTORAGE_KEY, 
                 headless=SELENIUM_HEADLESS, timeout=SELENIUM_TIMEOUT):
        """
        Initialize the TokenFetcher.
        
        Args:
            website_url (str): URL of the website to fetch token from
            token_key (str): LocalStorage key for the token
            headless (bool): Whether to run Chrome in headless mode
            timeout (int): Timeout in seconds for waiting for token
        """
        self.website_url = website_url
        self.token_key = token_key
        self.headless = headless
        self.timeout = timeout
        self.token = None
        self.token_timestamp = None
        self.token_cache_file = Path("auth_token.json")
    
    @log_execution_time(logger)
    @log_exceptions(logger)
    def get_token(self, force_refresh=False):
        """
        Get the authentication token.
        
        Args:
            force_refresh (bool): Whether to force a token refresh
            
        Returns:
            str: Authentication token
        """
        # Try to load cached token first
        if not force_refresh:
            cached_token = self._load_cached_token()
            if cached_token:
                logger.debug("Using cached token from file")
                self.token = cached_token
                return cached_token
            
            # Check if we already have a valid token in memory
            if self.token and self.token_timestamp:
                # Token is considered valid for 1 hour
                if time.time() - self.token_timestamp < 3600:
                    logger.debug("Using cached token from memory")
                    return self.token
        
        logger.info("Fetching new authentication token")
        token = self._fetch_token_from_website()
        
        if token:
            self.token = token
            self.token_timestamp = time.time()
            self._save_token_to_cache(token)
            logger.info("Successfully retrieved and cached authentication token")
            return token
        else:
            logger.error("Failed to retrieve authentication token")
            raise ValueError("Token not found")
    
    def _fetch_token_from_website(self) -> Optional[str]:
        """
        Fetch token from the H2HGGL website using Selenium.
        
        Returns:
            str: Authentication token or None if failed
        """
        driver = None
        try:
            # Setup Chrome options with enhanced configuration
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Core stability options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # GPU-related options to fix GPU errors
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-gpu-sandbox")
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument("--disable-background-timer-throttling")
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-renderer-backgrounding")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--disable-features=TranslateUI")
            chrome_options.add_argument("--disable-ipc-flooding-protection")
            chrome_options.add_argument("--use-gl=swiftshader")
            chrome_options.add_argument("--disable-vulkan")
            chrome_options.add_argument("--disable-d3d11")
            chrome_options.add_argument("--disable-accelerated-2d-canvas")
            chrome_options.add_argument("--disable-accelerated-jpeg-decoding")
            chrome_options.add_argument("--disable-accelerated-mjpeg-decode")
            chrome_options.add_argument("--disable-accelerated-video-decode")
            chrome_options.add_argument("--disable-accelerated-video-encode")
            
            # Anti-detection options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Additional stability options (but keep JavaScript enabled)
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-default-apps")
            chrome_options.add_argument("--disable-logging")
            chrome_options.add_argument("--silent")
            chrome_options.add_argument("--log-level=3")
            
            # Initialize the Chrome driver
            logger.debug("Starting Chrome browser...")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.timeout)
            
            # Add script to avoid detection
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.debug(f"Navigating to {self.website_url}")
            driver.get(self.website_url)
            
            logger.debug("Page loaded successfully. Waiting for token to be set...")
            
            # Wait a moment for the page to fully load and set the token
            time.sleep(3)
            
            # Extract token from local storage using JavaScript
            token = driver.execute_script(f"""
                return localStorage.getItem('{self.token_key}');
            """)
            
            if token:
                logger.debug(f"Successfully extracted token: {token[:50]}...")
                return token
            else:
                logger.warning(f"No token found in local storage with key '{self.token_key}'")
                return None
                
        except TimeoutException:
            logger.error(f"Timeout: Page failed to load within {self.timeout} seconds")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching token: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _load_cached_token(self) -> Optional[str]:
        """
        Load token from cache file if it exists and is still valid.
        
        Returns:
            str: Cached token or None if not valid
        """
        try:
            if not self.token_cache_file.exists():
                return None
            
            with open(self.token_cache_file, 'r', encoding='utf-8') as f:
                token_data = json.load(f)
            
            # Check if token is still valid (1 hour expiry)
            extracted_at = datetime.fromisoformat(token_data.get('extracted_at', ''))
            if (datetime.now() - extracted_at).total_seconds() < 3600:
                return token_data.get('token')
            else:
                logger.debug("Cached token has expired")
                return None
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Error loading cached token: {e}")
            return None
    
    def _save_token_to_cache(self, token: str) -> bool:
        """
        Save the extracted token to a cache file.
        
        Args:
            token (str): The authentication token to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            token_data = {
                "token": token,
                "extracted_at": datetime.now().isoformat(),
                "source": self.website_url,
                "key": self.token_key
            }
            
            with open(self.token_cache_file, 'w', encoding='utf-8') as f:
                json.dump(token_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Token saved to {self.token_cache_file}")
            return True
            
        except IOError as e:
            logger.warning(f"Error saving token to cache: {e}")
            return False
    
    @log_exceptions(logger)
    def get_auth_headers(self):
        """
        Get the authentication headers for API requests.
        
        Returns:
            dict: Headers dictionary with authentication token
        """
        token = self.get_token()
        return {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'authorization': f'Bearer {token}',
            'origin': 'https://www.h2hggl.com',
            'referer': 'https://www.h2hggl.com/'
        }
