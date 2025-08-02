#!/usr/bin/env python3
"""
Simple CPS URL test - step through manually to diagnose issues.

Run this interactively or step through with a debugger to understand:
1. Whether URLs are accessible
2. Whether they support streaming/range requests
3. What the actual file sizes are
"""

import requests
import sys

# Test URLs from cps.py
TEST_URLS = {
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
}

def step1_basic_connectivity():
    """Step 1: Test basic HTTP connectivity"""
    print("=" * 50)
    print("STEP 1: Testing basic connectivity")
    print("=" * 50)
    
    test_url = "https://www.census.gov"
    print(f"Testing basic connectivity to: {test_url}")
    
    try:
        response = requests.get(test_url, timeout=10)
        print(f"✓ Basic connectivity works: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ Basic connectivity failed: {e}")
        print("  Check internet connection, proxy settings, or firewall")
        return False

def step2_test_cps_access():
    """Step 2: Test CPS URL access"""
    print("\n" + "=" * 50)
    print("STEP 2: Testing CPS URL access")
    print("=" * 50)
    
    results = {}
    
    for year, url in TEST_URLS.items():
        print(f"\nTesting {year}: {url}")
        
        try:
            # Try HEAD request first (doesn't download content)
            print("  Trying HEAD request...")
            response = requests.head(url, timeout=30, allow_redirects=True)
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                # Get file info
                content_length = response.headers.get('Content-Length')
                accept_ranges = response.headers.get('Accept-Ranges')
                content_type = response.headers.get('Content-Type')
                
                print(f"  Content-Type: {content_type}")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    print(f"  File size: {size_mb:.1f} MB")
                else:
                    print("  File size: Unknown")
                
                print(f"  Accept-Ranges: {accept_ranges}")
                
                results[year] = {
                    'accessible': True,
                    'size_mb': size_mb if content_length else None,
                    'supports_ranges': accept_ranges == 'bytes'
                }
            else:
                print(f"  ✗ HTTP {response.status_code}: {response.reason}")
                results[year] = {'accessible': False}
                
        except requests.exceptions.Timeout:
            print("  ✗ Timeout - server may be slow or URL may be wrong")
            results[year] = {'accessible': False, 'error': 'timeout'}
        except requests.exceptions.ConnectionError as e:
            print(f"  ✗ Connection error: {e}")
            results[year] = {'accessible': False, 'error': 'connection'}
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[year] = {'accessible': False, 'error': str(e)}
    
    return results

def step3_test_streaming(results):
    """Step 3: Test streaming capability"""
    print("\n" + "=" * 50)
    print("STEP 3: Testing streaming capability")
    print("=" * 50)
    
    for year, result in results.items():
        if not result.get('accessible'):
            print(f"\nSkipping {year} - not accessible")
            continue
            
        url = TEST_URLS[year]
        print(f"\nTesting streaming for {year}")
        
        try:
            # Try to download just the first 1KB
            print("  Requesting first 1KB...")
            headers = {'Range': 'bytes=0-1023'}
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 206:
                print("  ✓ Supports partial content (streaming possible)")
                # Check if it's really a ZIP file
                content = response.content
                if content[:4] == b'PK\x03\x04':
                    print("  ✓ Content is valid ZIP format")
                else:
                    print("  ⚠ Content may not be ZIP format")
                result['streaming'] = True
                
            elif response.status_code == 200:
                print("  ✗ Server ignores range requests (full download required)")
                print(f"    Downloaded {len(response.content)} bytes")
                result['streaming'] = False
            else:
                print(f"  ✗ Range request failed: {response.status_code}")
                result['streaming'] = False
                
        except Exception as e:
            print(f"  ✗ Streaming test failed: {e}")
            result['streaming'] = False

def step4_summary(results):
    """Step 4: Print summary and recommendations"""
    print("\n" + "=" * 50)
    print("STEP 4: Summary and Recommendations")
    print("=" * 50)
    
    accessible_count = sum(1 for r in results.values() if r.get('accessible'))
    streaming_count = sum(1 for r in results.values() if r.get('streaming'))
    
    print(f"URLs tested: {len(results)}")
    print(f"Accessible: {accessible_count}/{len(results)}")
    print(f"Support streaming: {streaming_count}/{len(results)}")
    
    print("\nResults by year:")
    for year, result in results.items():
        if result.get('accessible'):
            size_info = f"({result.get('size_mb', '?'):.1f} MB)" if result.get('size_mb') else ""
            stream_info = "streaming ✓" if result.get('streaming') else "streaming ✗"
            print(f"  {year}: ✓ accessible {size_info}, {stream_info}")
        else:
            error = result.get('error', 'unknown error')
            print(f"  {year}: ✗ not accessible ({error})")
    
    print("\nRecommendations:")
    if accessible_count == 0:
        print("  ⚠ No URLs accessible - check network configuration")
        print("    - Verify internet connectivity")
        print("    - Check if behind corporate firewall/proxy")
        print("    - Try running from different network")
    elif accessible_count < len(results):
        print("  ⚠ Some URLs not accessible - may be temporary server issues")
    else:
        print("  ✓ All URLs accessible")
    
    if streaming_count == 0 and accessible_count > 0:
        print("  ⚠ No streaming support - cps.py will need to download full files")
        print("    - Ensure sufficient disk space")
        print("    - Expect longer processing times")
        print("    - Consider adding progress indicators")
    elif streaming_count > 0:
        print("  ✓ Streaming supported - efficient processing possible")

def main():
    """Run all test steps"""
    print("CPS URL Test - Step by Step Analysis")
    print("This will test whether CPS URLs work for the TMD project")
    
    # Step 1: Basic connectivity
    if not step1_basic_connectivity():
        print("\nStopping - fix connectivity issues first")
        return
    
    # Step 2: Test CPS URLs
    results = step2_test_cps_access()
    
    # Step 3: Test streaming
    step3_test_streaming(results)
    
    # Step 4: Summary
    step4_summary(results)
    
    print(f"\n{'='*50}")
    print("Test complete!")
    print("You can now step through this code in a debugger")
    print("or run individual functions to diagnose specific issues.")

if __name__ == "__main__":
    main()