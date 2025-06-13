#!/usr/bin/env python3
"""
Chinese Character Stroke Order Data Extractor - FIXED VERSION
Python equivalent of the JavaScript cnchar stroke order extraction
Uses the correct Hanzi Writer Data API endpoints with actual Chinese characters
"""

import json
import requests
from typing import Dict, List, Any
import time
import urllib.parse

class ChineseStrokeExtractor:
    def __init__(self):
        self.stroke_data = {}
        
    def get_stroke_order_hanzi_writer_data(self, char: str) -> Dict[str, Any]:
        """
        Get stroke order data from the official hanzi-writer-data repository
        FIXED: Uses actual Chinese characters in URL, not Unicode hex codes
        """
        try:
            # URL encode the Chinese character for safe transmission
            encoded_char = urllib.parse.quote(char, safe='')
            
            # The CORRECT working URLs based on hanzi-writer documentation
            # Uses actual Chinese characters, not hex codes!
            urls = [
                # Primary: JSDelivr CDN (recommended by official docs)
                f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{encoded_char}.json",
                # Backup: Direct GitHub raw content with character
                f"https://raw.githubusercontent.com/chanind/hanzi-writer-data/master/{encoded_char}.json",
                # Alternative: unpkg CDN
                f"https://unpkg.com/hanzi-writer-data/{encoded_char}.json"
            ]
            
            for i, url in enumerate(urls, 1):
                print(f"  [{i}/{len(urls)}] Trying: {url}")
                
                try:
                    response = requests.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  ✓ SUCCESS: Found stroke data!")
                        return {
                            'strokes': data.get('strokes', []),
                            'medians': data.get('medians', []),
                            'character': data.get('character', char),
                            'url_used': url,
                            'raw_data': data  # Include full data for analysis
                        }
                    else:
                        print(f"    Status: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"    Network error: {str(e)[:100]}...")
                    continue
                    
            return {}
                
        except Exception as e:
            print(f"    Unexpected error: {e}")
            return {}
    
    def get_stroke_order_alternative_apis(self, char: str) -> Dict[str, Any]:
        """
        Try alternative stroke order APIs as fallback
        """
        try:
            # Alternative API endpoints (if main one fails)
            alt_urls = [
                # Try the data explorer endpoint
                f"https://chanind.github.io/hanzi-writer-data/{urllib.parse.quote(char, safe='')}.json",
            ]
            
            for url in alt_urls:
                print(f"  [ALT] Trying: {url}")
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  ✓ SUCCESS: Alternative API worked!")
                        return {
                            'strokes': data.get('strokes', []),
                            'medians': data.get('medians', []),
                            'character': data.get('character', char),
                            'url_used': url,
                            'raw_data': data
                        }
                except:
                    continue
                    
            return {}
                
        except Exception as e:
            print(f"    Alternative API error: {e}")
            return {}
    
    def test_network_connectivity(self) -> bool:
        """
        Test if we can reach the internet and the specific APIs
        """
        test_urls = [
            "https://github.com",
            "https://cdn.jsdelivr.net",
            "https://unpkg.com",
            # Test with a known working character
            "https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/%E6%88%91.json"  # '我' character
        ]
        
        print("Testing network connectivity and API endpoints...")
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✓ {url} - OK")
                    if 'hanzi-writer-data' in url:
                        print(f"    └─ Hanzi Writer Data API is accessible!")
                else:
                    print(f"  ✗ {url} - Status {response.status_code}")
            except Exception as e:
                print(f"  ✗ {url} - {str(e)[:50]}...")
                continue
        
        # Test with a simple character to verify API format
        print("\nTesting API format with test character '我'...")
        try:
            test_url = "https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/%E6%88%91.json"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                stroke_count = len(data.get('strokes', []))
                print(f"  ✓ Test successful! Character '我' has {stroke_count} strokes")
                return True
            else:
                print(f"  ✗ Test failed with status {response.status_code}")
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
        
        return False
    
    def process_characters(self, characters: List[str]) -> Dict[str, Any]:
        """Process a list of characters and extract their stroke information"""
        
        # Test connectivity first
        if not self.test_network_connectivity():
            print("\n⚠️  WARNING: Network issues detected. Results may be limited.")
            print("Please check your internet connection and try again.\n")
        
        result = {}
        
        for i, char in enumerate(characters, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(characters)}] Processing: {char} (U+{ord(char):04X})")
            print(f"{'='*60}")
            
            # Try primary API
            stroke_info = self.get_stroke_order_hanzi_writer_data(char)
            
            # If primary fails, try alternatives
            if not stroke_info:
                print("  Primary API failed, trying alternatives...")
                stroke_info = self.get_stroke_order_alternative_apis(char)
            
            # Build comprehensive character data
            char_data = {
                'character': char,
                'unicode': ord(char),
                'hex_code': f"{ord(char):04x}",
                'unicode_name': f"U+{ord(char):04X}",
                'url_encoded': urllib.parse.quote(char, safe=''),
                'stroke_count': 0,
                'has_stroke_data': bool(stroke_info),
                'data_source': 'hanzi-writer-data' if stroke_info else 'none'
            }
            
            if stroke_info:
                char_data['strokes'] = stroke_info['strokes']
                char_data['stroke_count'] = len(stroke_info['strokes'])
                char_data['medians'] = stroke_info.get('medians', [])
                char_data['api_url'] = stroke_info.get('url_used', '')
                
                # Additional analysis
                if 'raw_data' in stroke_info:
                    raw = stroke_info['raw_data']
                    char_data['radical'] = raw.get('radical', '')
                    char_data['definition'] = raw.get('definition', '')
                    char_data['pinyin'] = raw.get('pinyin', [])
                    char_data['decomposition'] = raw.get('decomposition', '')
                
                print(f"  ✓ Stroke count: {char_data['stroke_count']}")
                print(f"  ✓ Data source: {char_data['api_url']}")
                if stroke_info['strokes']:
                    first_stroke = stroke_info['strokes'][0]
                    print(f"  ✓ First stroke: {first_stroke[:80]}{'...' if len(first_stroke) > 80 else ''}")
                if char_data.get('pinyin'):
                    print(f"  ✓ Pinyin: {', '.join(char_data['pinyin'])}")
                    
            else:
                print(f"  ✗ No stroke data found")
                print(f"  ℹ️  URL encoded as: {char_data['url_encoded']}")
            
            result[char] = char_data
            
            # Be respectful to APIs
            print(f"  ⏳ Waiting 1 second before next request...")
            time.sleep(1)
            
        return result
    
    def save_to_json(self, data: Dict[str, Any], filename: str = 'stroke_order_data.json'):
        """Save stroke data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Show file stats
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            print(f"\n✓ Data saved to {filename}")
            print(f"✓ File size: {len(content):,} characters")
            print(f"✓ Characters processed: {len(data)}")
                
        except Exception as e:
            print(f"✗ Error saving file: {e}")
    
    def print_detailed_summary(self, data: Dict[str, Any]):
        """Print a detailed summary of results"""
        print(f"\n{'='*80}")
        print("DETAILED STROKE DATA EXTRACTION SUMMARY")
        print(f"{'='*80}")
        
        total_chars = len(data)
        successful = sum(1 for d in data.values() if d['has_stroke_data'])
        total_strokes = sum(d['stroke_count'] for d in data.values())
        
        print(f"📊 STATISTICS:")
        print(f"   Total characters processed: {total_chars}")
        print(f"   Successfully found stroke data: {successful}")
        print(f"   Success rate: {successful/total_chars*100:.1f}%")
        print(f"   Total strokes extracted: {total_strokes}")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 90)
        
        for char, info in data.items():
            status = "✅ SUCCESS" if info['has_stroke_data'] else "❌ FAILED"
            stroke_info = f"{info['stroke_count']:2d} strokes" if info['stroke_count'] > 0 else "No data"
            url_encoded = info.get('url_encoded', 'N/A')
            pinyin = ', '.join(info.get('pinyin', [])) if info.get('pinyin') else 'N/A'
            
            print(f"{char:2s} {info['unicode_name']:>8} {status:>10} {stroke_info:>12} {pinyin:>15}")
            print(f"     URL encoded: {url_encoded}")
            if info['has_stroke_data']:
                source = info.get('api_url', 'N/A')
                print(f"     Source: {source}")
        
        if successful > 0:
            print(f"\n🎉 SUCCESS! Found stroke data for {successful} characters!")
            print("💾 Check the JSON file for complete stroke path data.")
            print("\n💡 TIPS:")
            print("   - The 'strokes' array contains SVG path data for each stroke")
            print("   - The 'medians' array contains simplified stroke paths")
            print("   - Coordinates are on a 1024x1024 coordinate system")
            print("   - Upper-left corner is at (0, 900), lower-right at (1024, -124)")
        else:
            print(f"\n⚠️  No stroke data was retrieved. Possible issues:")
            print("   - Network connectivity problems")
            print("   - Characters might not be in the hanzi-writer-data database")
            print("   - API endpoints might be temporarily unavailable")
            print("   - Try running the connectivity test again")

    def demonstrate_api_usage(self):
        """Demonstrate the correct API usage with examples"""
        print("\n" + "="*80)
        print("API ENDPOINT DEMONSTRATION")
        print("="*80)
        
        examples = ['我', '你', '好']
        
        for char in examples:
            encoded = urllib.parse.quote(char, safe='')
            print(f"\nCharacter: {char}")
            print(f"Unicode: U+{ord(char):04X}")
            print(f"URL Encoded: {encoded}")
            print(f"✓ Correct API URL: https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{encoded}.json")
            print(f"✗ Wrong API URL: https://raw.githubusercontent.com/chanind/hanzi-writer-data/master/data/{ord(char):05x}.json")

def main():
    print("🖌️  Chinese Character Stroke Order Extractor - FIXED VERSION")
    print("Using Official Hanzi Writer Data API with correct character-based URLs")
    print("=" * 80)
    
    # Initialize extractor
    extractor = ChineseStrokeExtractor()
    
    # Show API usage examples
    extractor.demonstrate_api_usage()
    
    # Characters to process
    characters = ['你', '我', '好', '一', '爱', '中', '国']
    
    print(f"\n🎯 Target characters: {' '.join(characters)}")
    print(f"📡 API Source: Hanzi Writer Data (uses actual Chinese characters in URLs)")
    
    # Process characters
    stroke_data = extractor.process_characters(characters)
    
    # Save results
    extractor.save_to_json(stroke_data)
    
    # Print detailed summary
    extractor.print_detailed_summary(stroke_data)

if __name__ == "__main__":
    try:
        import requests
        import urllib.parse
    except ImportError as e:
        print(f"❌ Error: Required library missing: {e}")
        print("📦 Install with: pip install requests")
        exit(1)
    
    main()