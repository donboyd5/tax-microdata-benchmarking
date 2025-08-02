#!/usr/bin/env python3
"""
Final CPS URL test with SSL handling.

This version handles SSL certificate issues that may occur
with some Census Bureau subdomains.
"""

import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Disable SSL warnings if we need to skip verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URLs from cps.py
CPS_URLS = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip", 
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
}

def create_robust_session():
    """Create a requests session with retry logic and SSL handling."""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set a proper User-Agent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    return session

def test_url_comprehensive(year, url):
    """Test a single URL comprehensively."""
    print(f"\n{'='*60}")
    print(f"Testing {year}: {url}")
    print(f"{'='*60}")
    
    session = create_robust_session()
    result = {'year': year, 'url': url}
    
    # Test 1: Try with SSL verification
    try:
        print("Step 1: Testing with SSL verification...")
        response = session.head(url, timeout=30, verify=True)
        result['ssl_verification'] = True
        result['accessible'] = True
        print(f"✓ Accessible with SSL verification (Status: {response.status_code})")
        
    except requests.exceptions.SSLError:
        print("⚠ SSL verification failed, trying without verification...")
        try:
            response = session.head(url, timeout=30, verify=False)
            result['ssl_verification'] = False  
            result['accessible'] = True
            print(f"✓ Accessible without SSL verification (Status: {response.status_code})")
        except Exception as e:
            print(f"✗ Still failed: {e}")
            result['accessible'] = False
            return result
            
    except Exception as e:
        print(f"✗ Not accessible: {e}")
        result['accessible'] = False
        return result
    
    # Test 2: Get file information
    print("Step 2: Getting file information...")
    content_length = response.headers.get('Content-Length')
    accept_ranges = response.headers.get('Accept-Ranges')
    content_type = response.headers.get('Content-Type')
    
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        result['size_mb'] = size_mb
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print("  File size: Unknown")
        result['size_mb'] = None
    
    print(f"  Content-Type: {content_type}")
    print(f"  Accept-Ranges: {accept_ranges}")
    result['supports_ranges'] = accept_ranges == 'bytes'
    
    # Test 3: Test streaming capability
    print("Step 3: Testing streaming capability...")
    try:
        headers = {'Range': 'bytes=0-1023'}  # First 1KB
        stream_response = session.get(
            url, 
            headers=headers, 
            timeout=30, 
            verify=result['ssl_verification']
        )
        
        if stream_response.status_code == 206:
            print("✓ Supports partial content requests (streaming possible)")
            result['streaming'] = True
            
            # Verify it's actually ZIP data
            content = stream_response.content
            if content[:4] == b'PK\x03\x04':
                print("✓ Content verified as ZIP format")
                result['valid_zip'] = True
            else:
                print("⚠ Content may not be ZIP format")
                result['valid_zip'] = False
                
        elif stream_response.status_code == 200:
            print("⚠ Server ignores range requests (full download required)")
            print(f"  Downloaded {len(stream_response.content)} bytes instead of 1KB")
            result['streaming'] = False
        else:
            print(f"✗ Range request failed: {stream_response.status_code}")
            result['streaming'] = False
            
    except Exception as e:
        print(f"✗ Streaming test failed: {e}")
        result['streaming'] = False
    
    return result

def main():
    """Test all CPS URLs and provide recommendations."""
    print("CPS URL Comprehensive Test")
    print("Testing all URLs from tmd/datasets/cps.py")
    print("This may take a few minutes...")
    
    results = []
    
    # Test each URL
    for year, url in CPS_URLS.items():
        result = test_url_comprehensive(year, url)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    accessible = [r for r in results if r.get('accessible')]
    streaming = [r for r in results if r.get('streaming')]
    ssl_issues = [r for r in results if r.get('accessible') and not r.get('ssl_verification', True)]
    
    print(f"Total URLs tested: {len(results)}")
    print(f"Accessible: {len(accessible)}/{len(results)}")
    print(f"Support streaming: {len(streaming)}/{len(results)}")
    print(f"SSL verification issues: {len(ssl_issues)}")
    
    print("\nDetailed results:")
    for result in results:
        year = result['year']
        if result.get('accessible'):
            ssl_info = "SSL ✓" if result.get('ssl_verification', True) else "SSL ✗"
            stream_info = "Stream ✓" if result.get('streaming') else "Stream ✗"
            size_info = f"({result['size_mb']:.1f}MB)" if result.get('size_mb') else "(size unknown)"
            print(f"  {year}: ✓ {ssl_info}, {stream_info} {size_info}")
        else:
            print(f"  {year}: ✗ Not accessible")
    
    # Recommendations for the TMD project
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR TMD PROJECT")
    print(f"{'='*60}")
    
    if len(accessible) == 0:
        print("❌ CRITICAL: No URLs are accessible")
        print("   - Check network connectivity and firewall settings")
        print("   - The TMD project cannot download CPS data")
        
    elif len(ssl_issues) > 0:
        print("⚠️  SSL verification issues detected")
        print("   - Some URLs require verify=False in requests")
        print("   - Consider updating cps.py to handle SSL issues")
        print("   - This is common with government data sites")
        
    if len(streaming) == 0 and len(accessible) > 0:
        print("⚠️  No streaming support detected")
        print("   - CPS files must be downloaded completely before processing")
        print("   - Ensure adequate disk space and bandwidth")
        print("   - Consider adding download progress indicators")
        
    elif len(streaming) > 0:
        print("✅ Streaming support available")
        print("   - Memory-efficient processing possible")
        print("   - Can process large files without full download")
    
    # Specific code suggestions
    if len(ssl_issues) > 0:
        print(f"\n{'='*60}")
        print("CODE SUGGESTIONS")
        print(f"{'='*60}")
        print("Consider modifying requests calls in cps.py to handle SSL:")
        print("  response = requests.get(url, verify=False)")
        print("Or add SSL context handling for production use.")

if __name__ == "__main__":
    main()