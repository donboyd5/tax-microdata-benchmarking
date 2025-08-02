#!/usr/bin/env python3
"""
Test program to check CPS URL accessibility and streaming capability.

This program tests whether the CPS URLs from the TMD project can be:
1. Downloaded successfully
2. Used in streaming fashion (partial content support)
3. Accessed without authentication issues

Run with: python test_cps_urls.py
"""

import requests
from io import BytesIO
from zipfile import ZipFile
import time
from typing import Dict, Tuple, Optional

# CPS URLs from tmd/datasets/cps.py
CPS_URL_BY_YEAR = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip",
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
}

def test_url_accessibility(url: str, timeout: int = 30) -> Tuple[bool, str, Optional[Dict]]:
    """
    Test if URL is accessible via HEAD request.
    
    Returns:
        (success, message, headers_dict)
    """
    try:
        print(f"  Testing accessibility...")
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            headers = dict(response.headers)
            return True, f"✓ Accessible (Status: {response.status_code})", headers
        else:
            return False, f"✗ HTTP Error {response.status_code}: {response.reason}", None
            
    except requests.exceptions.Timeout:
        return False, f"✗ Timeout after {timeout} seconds", None
    except requests.exceptions.ConnectionError:
        return False, "✗ Connection error", None
    except requests.exceptions.RequestException as e:
        return False, f"✗ Request error: {str(e)}", None

def test_streaming_support(url: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Test if URL supports range requests for streaming.
    
    Returns:
        (supports_streaming, message)
    """
    try:
        print(f"  Testing streaming support...")
        
        # Test with a small range request
        headers = {'Range': 'bytes=0-1023'}  # First 1KB
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 206:  # Partial Content
            return True, "✓ Supports range requests (streaming possible)"
        elif response.status_code == 200:
            # Server doesn't support range requests but file is accessible
            content_length = response.headers.get('Content-Length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                return False, f"✗ No range support, full download required ({size_mb:.1f} MB)"
            else:
                return False, "✗ No range support, full download required (unknown size)"
        else:
            return False, f"✗ Range request failed: {response.status_code}"
            
    except Exception as e:
        return False, f"✗ Streaming test error: {str(e)}"

def test_partial_download(url: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Test downloading a small portion to verify content.
    
    Returns:
        (success, message)
    """
    try:
        print(f"  Testing partial download...")
        
        # Download first 64KB to test if it's a valid ZIP
        headers = {'Range': 'bytes=0-65535'}
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code in [200, 206]:
            content = response.content
            
            # Check if it starts with ZIP signature
            if content[:4] == b'PK\x03\x04':
                return True, f"✓ Valid ZIP content ({len(content)} bytes downloaded)"
            else:
                return False, f"✗ Content doesn't appear to be ZIP format"
        else:
            return False, f"✗ Partial download failed: {response.status_code}"
            
    except Exception as e:
        return False, f"✗ Partial download error: {str(e)}"

def test_full_download_feasibility(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Test if full download would be feasible (check size, don't actually download).
    
    Returns:
        (feasible, message)
    """
    try:
        print(f"  Testing download feasibility...")
        
        response = requests.head(url, timeout=timeout)
        content_length = response.headers.get('Content-Length')
        
        if content_length:
            size_bytes = int(content_length)
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb < 100:  # Less than 100MB is reasonable
                return True, f"✓ Download feasible ({size_mb:.1f} MB)"
            elif size_mb < 500:  # 100-500MB is manageable
                return True, f"⚠ Large download ({size_mb:.1f} MB)"
            else:
                return False, f"✗ Very large download ({size_mb:.1f} MB)"
        else:
            return False, "? Unknown file size"
            
    except Exception as e:
        return False, f"✗ Size check error: {str(e)}"

def test_cps_url(year: int, url: str) -> Dict:
    """Test a single CPS URL comprehensively."""
    print(f"\n{'='*60}")
    print(f"Testing CPS {year}: {url}")
    print(f"{'='*60}")
    
    results = {
        'year': year,
        'url': url,
        'accessible': False,
        'supports_streaming': False,
        'partial_download': False,
        'download_feasible': False,
        'headers': None,
        'messages': []
    }
    
    # Test 1: Basic accessibility
    accessible, msg, headers = test_url_accessibility(url)
    results['accessible'] = accessible
    results['headers'] = headers
    results['messages'].append(f"Accessibility: {msg}")
    print(f"  {msg}")
    
    if not accessible:
        print("  Skipping further tests due to accessibility issues.")
        return results
    
    # Show relevant headers
    if headers:
        print("  Headers of interest:")
        for header in ['Content-Length', 'Accept-Ranges', 'Content-Type']:
            if header in headers:
                print(f"    {header}: {headers[header]}")
    
    # Test 2: Streaming support
    streaming, msg = test_streaming_support(url)
    results['supports_streaming'] = streaming
    results['messages'].append(f"Streaming: {msg}")
    print(f"  {msg}")
    
    # Test 3: Partial download
    partial, msg = test_partial_download(url)
    results['partial_download'] = partial
    results['messages'].append(f"Partial download: {msg}")
    print(f"  {msg}")
    
    # Test 4: Download feasibility
    feasible, msg = test_full_download_feasibility(url)
    results['download_feasible'] = feasible
    results['messages'].append(f"Download feasibility: {msg}")
    print(f"  {msg}")
    
    return results

def print_summary(all_results):
    """Print a summary of all test results."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    accessible_count = sum(1 for r in all_results if r['accessible'])
    streaming_count = sum(1 for r in all_results if r['supports_streaming'])
    
    print(f"URLs tested: {len(all_results)}")
    print(f"Accessible: {accessible_count}/{len(all_results)}")
    print(f"Support streaming: {streaming_count}/{len(all_results)}")
    
    print("\nDetailed Results:")
    for result in all_results:
        status = "✓" if result['accessible'] else "✗"
        stream = "Stream: ✓" if result['supports_streaming'] else "Stream: ✗"
        print(f"  {status} {result['year']}: {stream}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if accessible_count == len(all_results):
        print("✓ All URLs are accessible")
    else:
        print("⚠ Some URLs are not accessible - check network connectivity")
    
    if streaming_count > 0:
        print("✓ Some URLs support streaming - efficient processing possible")
    else:
        print("⚠ No URLs support streaming - full downloads required")
        print("  Consider implementing download progress indicators")
    
    if streaming_count == len(all_results):
        print("✓ All URLs support streaming - optimal for memory-efficient processing")
    

def main():
    """Main test function."""
    print("CPS URL Accessibility and Streaming Test")
    print("Testing URLs from tmd/datasets/cps.py")
    
    all_results = []
    
    for year, url in CPS_URL_BY_YEAR.items():
        result = test_cps_url(year, url)
        all_results.append(result)
        
        # Small delay between tests to be respectful
        time.sleep(1)
    
    print_summary(all_results)
    
    # Instructions for user
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Review the accessibility results above")
    print("2. If URLs are not accessible, check:")
    print("   - Internet connectivity")
    print("   - Corporate firewall/proxy settings")
    print("   - VPN configuration")
    print("3. If streaming is not supported:")
    print("   - The code will need to download complete files")
    print("   - Ensure sufficient disk space and bandwidth")
    print("4. Test actual usage in cps.py with a small dataset")

if __name__ == "__main__":
    main()