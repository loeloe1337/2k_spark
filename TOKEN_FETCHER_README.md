# H2H Token Fetcher - Enhanced Implementation

This document describes the enhanced token fetching functionality for the H2H GG League API, based on the sample implementation and improved for production use.

## Overview

The token fetcher has been significantly improved with the following enhancements:

- **File-based caching**: Tokens are cached to disk for persistence across application restarts
- **Enhanced Chrome options**: Better compatibility and anti-detection measures
- **Robust error handling**: Comprehensive error handling with detailed logging
- **Standalone script**: Independent script for manual token fetching
- **Flexible configuration**: Configurable URLs, token keys, and timeouts

## Files Modified/Created

### Enhanced Files
- `backend/core/data/fetchers/token.py` - Enhanced with caching and better error handling
- `requirements.txt` - Added webdriver-manager for easier Chrome driver management

### New Files
- `fetch_auth_token.py` - Standalone token fetching script
- `TOKEN_FETCHER_README.md` - This documentation file

## Usage

### 1. Standalone Token Fetcher

The standalone script can be used independently to fetch tokens:

```bash
# Basic usage (non-headless for debugging)
python fetch_auth_token.py

# Headless mode (for automation)
python fetch_auth_token.py --headless

# Custom output file
python fetch_auth_token.py --output my_token.json

# Custom URL and token key
python fetch_auth_token.py --url https://custom-url.com --token-key custom-key

# Verbose output
python fetch_auth_token.py --verbose
```

#### Command Line Options

- `--headless`: Run browser in headless mode (default: False)
- `--timeout SECONDS`: Timeout for page operations (default: 30)
- `--output FILE`: Output file for the token (default: auth_token.json)
- `--url URL`: Custom URL to navigate to
- `--token-key KEY`: Custom localStorage key for the token
- `--verbose`: Enable verbose output

### 2. Backend Service Integration

The enhanced backend service automatically uses file caching:

```python
from backend.core.data.fetchers.token import TokenFetcher

# Initialize the token fetcher
fetcher = TokenFetcher()

# Get token (uses cache if available and valid)
token = fetcher.get_token()

# Force refresh (ignores cache)
token = fetcher.get_token(force_refresh=True)

# Get auth headers for API requests
headers = fetcher.get_auth_headers()
```

### 3. Testing

You can test the token fetcher functionality using the standalone script:

```bash
# Test with verbose output
python fetch_auth_token.py --verbose

# Test in headless mode
python fetch_auth_token.py --headless --verbose

# Check if cached token exists
ls -la auth_token.json
```

## Token Caching

### Cache File Format

Tokens are cached in JSON format with the following structure:

```json
{
  "token": "eyJhbGciOiJIUzI1NiJ9...",
  "extracted_at": "2024-01-15T10:30:00.123456",
  "source": "https://h2hggl.com/en/match/NB122120625",
  "key": "sis-hudstats-token",
  "expires_at": "2024-01-15T11:30:00.123456"
}
```

### Cache Behavior

- Tokens are cached for 1 hour by default
- Cache is checked before making new requests
- Cache file is automatically created in the working directory
- Invalid or expired cache is automatically ignored

## Configuration

### Backend Configuration

The backend service uses settings from `backend/config/settings.py`:

```python
# H2H GG League API settings
H2H_BASE_URL = "https://api-sis-stats.hudstats.com/v1"
H2H_WEBSITE_URL = "https://h2hggl.com/en/match/NB122120625"
H2H_TOKEN_LOCALSTORAGE_KEY = "sis-hudstats-token"

# Selenium settings
SELENIUM_HEADLESS = True
SELENIUM_TIMEOUT = 10  # seconds
```

### Chrome Options

The enhanced implementation includes comprehensive Chrome options for stability and anti-detection:

- `--no-sandbox`: Bypass OS security model
- `--disable-dev-shm-usage`: Overcome limited resource problems
- `--disable-gpu`: Disable GPU hardware acceleration
- `--disable-gpu-sandbox`: Additional GPU security bypass
- `--disable-software-rasterizer`: Disable software rendering
- `--use-gl=swiftshader`: Use software GL implementation
- `--disable-vulkan`: Disable Vulkan API
- `--window-size=1920,1080`: Set window size
- `--disable-blink-features=AutomationControlled`: Avoid detection
- `--user-agent`: Custom user agent to avoid detection
- `--headless`: Run without GUI (when enabled)
- Various other stability and performance options

## Troubleshooting

### Common Issues

1. **Chrome driver not found**
   - Install webdriver-manager: `pip install webdriver-manager`
   - Ensure Chrome/Chromium is installed

2. **Token not found in localStorage**
   - Try running without `--headless` to see what's happening
   - Verify the website URL is correct
   - Check if the localStorage key is correct

3. **Timeout errors**
   - Increase timeout with `--timeout` option
   - Check internet connection
   - Verify website is accessible

4. **Permission errors**
   - Ensure write permissions in the working directory
   - Check if antivirus is blocking Chrome

### Debug Mode

Run the standalone script without `--headless` to see the browser in action:

```bash
python fetch_auth_token.py --verbose
```

This will show you exactly what the browser is doing and help identify issues.

### Logging

The backend service includes comprehensive logging:

- Debug: Detailed operation information
- Info: General operation status
- Warning: Non-critical issues (e.g., cache errors)
- Error: Critical failures

Logs are written to the configured log files in the `logs/` directory.

## Security Considerations

- Tokens are stored in plain text in cache files
- Cache files should be excluded from version control
- Consider implementing token encryption for production use
- Regularly rotate tokens for security

## Dependencies

Required packages (automatically installed with `pip install -r requirements.txt`):

- `selenium>=4.0.0`: Browser automation
- `webdriver-manager>=3.8.0`: Automatic Chrome driver management

## Integration with Existing Code

The enhanced token fetcher is fully backward compatible with existing code. No changes are required to existing implementations that use the `TokenFetcher` class.

## Performance Improvements

- **Reduced API calls**: File caching reduces the need for frequent token fetching
- **Faster startup**: Cached tokens eliminate browser startup time for subsequent requests
- **Better reliability**: Enhanced error handling and retry logic
- **Resource efficiency**: Optimized Chrome options reduce memory usage

## Future Enhancements

Potential improvements for future versions:

- Token encryption in cache files
- Automatic token refresh before expiry
- Multiple token source support
- Distributed caching for multi-instance deployments
- Integration with secret management systems