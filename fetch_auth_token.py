#!/usr/bin/env python3
"""
H2H GG League - Authentication Token Fetcher

This script navigates to the H2HGGL website and extracts the authentication token
from local storage to save it for use with the API.

Usage:
    python fetch_auth_token.py
    python fetch_auth_token.py --headless
    python fetch_auth_token.py --output custom_token.json
    python fetch_auth_token.py --url https://custom-url.com

Requires:
    - selenium library for browser automation
    - Chrome/Chromium browser
"""

import argparse
import json
import sys
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
    sys.exit(1)


class H2HTokenFetcher:
    """Fetches authentication token from H2HGGL website using Selenium."""
    
    def __init__(self, target_url: str = None, token_key: str = None, headless: bool = True, timeout: int = 30):
        """
        Initialize the H2HTokenFetcher.
        
        Args:
            target_url (str): URL to navigate to for token extraction
            token_key (str): LocalStorage key for the token
            headless (bool): Whether to run browser in headless mode
            timeout (int): Timeout in seconds for page operations
        """
        self.headless = headless
        self.timeout = timeout
        self.target_url = target_url or "https://www.h2hggl.com/en/ebasketball/players/"
        self.token_key = token_key or "sis-hudstats-token"
    
    def fetch_token(self) -> Optional[str]:
        """Navigate to H2HGGL website and extract auth token from local storage."""
        driver = None
        try:
            # Setup Chrome options with enhanced configuration
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize the Chrome driver
            print("Starting Chrome browser...")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.timeout)
            
            # Add script to avoid detection
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print(f"Navigating to {self.target_url}...")
            
            # Navigate to the target URL
            driver.get(self.target_url)
            
            print("Page loaded successfully. Waiting for token to be set...")
            
            # Wait a moment for the page to fully load and set the token
            time.sleep(3)
            
            # Extract token from local storage using JavaScript
            token = driver.execute_script(f"""
                return localStorage.getItem('{self.token_key}');
            """)
            
            if token:
                print(f"Successfully extracted token: {token[:50]}...")
                return token
            else:
                print(f"No token found in local storage with key '{self.token_key}'")
                # Try to wait a bit longer and check again
                print("Waiting additional 5 seconds for token to be set...")
                time.sleep(5)
                token = driver.execute_script(f"""
                    return localStorage.getItem('{self.token_key}');
                """)
                
                if token:
                    print(f"Successfully extracted token after additional wait: {token[:50]}...")
                    return token
                else:
                    print("Still no token found. The page might not have loaded properly.")
                    return None
                
        except TimeoutException:
            print(f"Timeout: Page failed to load within {self.timeout} seconds")
            return None
        except WebDriverException as e:
            print(f"WebDriver error: {e}")
            print("Make sure Chrome/Chromium browser is installed and accessible.")
            return None
        except Exception as e:
            print(f"Error fetching token: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def save_token_to_file(self, token: str, output_file: str = "auth_token.json") -> bool:
        """Save the extracted token to a JSON file."""
        
        try:
            token_data = {
                "token": token,
                "extracted_at": self._get_current_timestamp(),
                "source": self.target_url,
                "key": self.token_key,
                "expires_at": self._get_expiry_timestamp()
            }
            
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(token_data, f, indent=2, ensure_ascii=False)
            
            print(f"Token saved to {output_path.absolute()}")
            return True
            
        except IOError as e:
            print(f"Error saving token to file: {e}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def _get_expiry_timestamp(self) -> str:
        """Get expiry timestamp (1 hour from now) in ISO format."""
        from datetime import timedelta
        expiry_time = datetime.now() + timedelta(hours=1)
        return expiry_time.isoformat()


def main():
    """Main function to handle command line arguments and fetch token."""
    parser = argparse.ArgumentParser(
        description="Fetch authentication token from H2HGGL website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_auth_token.py
  python fetch_auth_token.py --headless
  python fetch_auth_token.py --output my_token.json
  python fetch_auth_token.py --url https://www.h2hggl.com/en/match/NB122120625
  python fetch_auth_token.py --token-key custom-token-key
        """
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: False for better debugging)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for page operations (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auth_token.json",
        help="Output file for the token (default: auth_token.json)"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to navigate to (default: H2HGGL players page)"
    )
    parser.add_argument(
        "--token-key",
        type=str,
        help="Custom localStorage key for the token (default: sis-hudstats-token)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Headless: {args.headless}")
        print(f"  Timeout: {args.timeout}s")
        print(f"  Output: {args.output}")
        print(f"  URL: {args.url or 'default'}")
        print(f"  Token Key: {args.token_key or 'default'}")
        print()
    
    # Create fetcher and get token
    fetcher = H2HTokenFetcher(
        target_url=args.url,
        token_key=args.token_key,
        headless=args.headless, 
        timeout=args.timeout
    )
    
    print("Starting token extraction...")
    token = fetcher.fetch_token()
    
    if token:
        if fetcher.save_token_to_file(token, args.output):
            print(f"\n✅ Token successfully saved to {args.output}")
            print(f"Token preview: {token[:50]}...")
            print("\nYou can now use this token for API requests.")
            return 0
        else:
            print("\n❌ Failed to save token to file")
            return 1
    else:
        print("\n❌ Failed to fetch authentication token")
        print("\nTroubleshooting tips:")
        print("1. Make sure Chrome/Chromium browser is installed")
        print("2. Try running without --headless flag to see what's happening")
        print("3. Check if the website URL is accessible")
        print("4. Verify the localStorage key is correct")
        return 1


if __name__ == '__main__':
    sys.exit(main())