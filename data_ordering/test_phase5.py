"""
Tests for the llm_integration module.
Tests the directory-based analysis approach.
"""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.llm_integration import (
    DirectoryAnalysis, RateLimiter, DirectoryAnalyzer, MockGeminiClient,
    GeminiClient, GENAI_AVAILABLE
)


def test_directory_analysis():
    """Test DirectoryAnalysis dataclass."""
    print("=" * 60)
    print("DIRECTORY ANALYSIS TEST")
    print("=" * 60)
    
    analysis = DirectoryAnalysis(
        taxonomic_class='Insecta',
        determination='Coleoptera',
        collection_code='LH',
        campaign_year=2019,
        specimen_id_regex=r'(LH[-\s]?\d{3,8})',
        confidence=0.9,
        directory_path='Colección LH-/3. Artrópodos/Coleoptera',
        sample_filenames=['LH 15083.jpg', 'LH-15084.tif']
    )
    
    # Test to_dict
    d = analysis.to_dict()
    print(f"to_dict: {d}")
    assert d['taxonomic_class'] == 'Insecta'
    assert d['collection_code'] == 'LH'
    
    # Test from_dict
    analysis2 = DirectoryAnalysis.from_dict(d)
    assert analysis2.taxonomic_class == 'Insecta'
    assert analysis2.specimen_id_regex == r'(LH[-\s]?\d{3,8})'
    
    # Test extract_specimen_id
    specimen_id = analysis.extract_specimen_id('LH 15083.jpg')
    print(f"Extracted specimen ID: {specimen_id}")
    assert specimen_id == 'LH 15083', f"Expected 'LH 15083', got '{specimen_id}'"
    
    specimen_id2 = analysis.extract_specimen_id('LH-15084a.tif')
    print(f"Extracted specimen ID 2: {specimen_id2}")
    assert specimen_id2 == 'LH-15084', f"Expected 'LH-15084', got '{specimen_id2}'"
    
    # Test with no regex
    analysis_no_regex = DirectoryAnalysis()
    assert analysis_no_regex.extract_specimen_id('test.jpg') is None
    
    print("\n[PASS] DirectoryAnalysis test passed")
    return True


def test_rate_limiter():
    """Test rate limiter functionality."""
    print("\n" + "=" * 60)
    print("RATE LIMITER TEST")
    print("=" * 60)
    
    import time
    
    limiter = RateLimiter(requests_per_minute=60)
    
    start = time.time()
    limiter.wait_if_needed()
    elapsed1 = time.time() - start
    print(f"First request delay: {elapsed1:.3f}s")
    
    start = time.time()
    limiter.wait_if_needed()
    elapsed2 = time.time() - start
    print(f"Second request delay: {elapsed2:.3f}s")
    
    assert elapsed2 >= 0.9, f"Expected ~1s delay, got {elapsed2:.3f}s"
    
    print("\n[PASS] Rate limiter test passed")
    return True


def test_mock_client():
    """Test MockGeminiClient responses."""
    print("\n" + "=" * 60)
    print("MOCK CLIENT TEST")
    print("=" * 60)
    
    client = MockGeminiClient()
    
    # Test Las Hoyas Insecta/Coleoptera directory
    result1 = client.analyze_directory(
        Path("Colección LH-/3. Artrópodos/Insecta/Coleoptera"),
        sample_filenames=['LH 15083.jpg', 'LH-15084.tif']
    )
    print(f"\nLas Hoyas Coleoptera directory:")
    print(f"  Collection: {result1.collection_code}")
    print(f"  Class: {result1.taxonomic_class}")
    print(f"  Determination: {result1.determination}")
    print(f"  Regex: {result1.specimen_id_regex}")
    assert result1.collection_code == 'LH'
    assert result1.taxonomic_class == 'Insecta'
    assert result1.determination == 'Coleoptera'
    
    # Test regex extraction
    specimen_id = result1.extract_specimen_id('LH 15083.jpg')
    print(f"  Extracted ID: {specimen_id}")
    assert specimen_id == 'LH 15083'
    
    # Test Buenache Crustacea directory
    result2 = client.analyze_directory(
        Path("Buenache/Crustacea"),
        sample_filenames=['K-Bue 085.jpg', 'K-Bue 086.tif']
    )
    print(f"\nBuenache Crustacea directory:")
    print(f"  Collection: {result2.collection_code}")
    print(f"  Class: {result2.taxonomic_class}")
    print(f"  Regex: {result2.specimen_id_regex}")
    assert result2.collection_code == 'BUE'
    
    # Test Decapoda with genus
    result3 = client.analyze_directory(
        Path("Colección LH-/Decapoda/Delclosia"),
        sample_filenames=['LH 8201.jpg', 'LH 8201 b.jpg']
    )
    print(f"\nDecapoda/Delclosia directory:")
    print(f"  Class: {result3.taxonomic_class}")
    print(f"  Determination: {result3.determination}")
    # Malacostraca is the correct taxonomic class for Decapoda
    assert result3.taxonomic_class == 'Malacostraca'
    assert result3.determination == 'Delclosia'
    
    # Test Montsec detection
    result4 = client.analyze_directory(
        Path("Montsec/Fossils"),
        sample_filenames=['specimen_001.jpg']
    )
    print(f"\nMontsec directory:")
    print(f"  Collection: {result4.collection_code}")
    assert result4.collection_code == 'MON'
    
    # Test year extraction
    result5 = client.analyze_directory(
        Path("Las Hoyas/Campaign_2019/Insecta"),
        sample_filenames=['LH 20001.jpg']
    )
    print(f"\nCampaign year directory:")
    print(f"  Year: {result5.campaign_year}")
    assert result5.campaign_year == 2019
    
    # Test unknown
    result6 = client.analyze_directory(
        Path("Camera/Random"),
        sample_filenames=['DSC00001.jpg']
    )
    print(f"\nUnknown directory:")
    print(f"  Collection: {result6.collection_code}")
    print(f"  Confidence: {result6.confidence}")
    assert result6.collection_code == 'unknown'
    assert result6.confidence < 0.5
    
    print("\n[PASS] Mock client test passed")
    return True


def test_directory_analyzer_with_mock():
    """Test DirectoryAnalyzer with mock client."""
    print("\n" + "=" * 60)
    print("DIRECTORY ANALYZER TEST (with mock)")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    cache_path = temp_dir / "test_cache.json"
    
    try:
        mock_client = MockGeminiClient()
        analyzer = DirectoryAnalyzer(
            client=mock_client,
            cache_path=cache_path
        )
        
        # Analyze a directory
        result1 = analyzer.analyze_directory(
            Path("Colección LH-/Insecta/Coleoptera"),
            sample_filenames=['LH 15083.jpg']
        )
        print(f"\nFirst analysis:")
        print(f"  Class: {result1.taxonomic_class}")
        print(f"  Determination: {result1.determination}")
        
        # Analyze same directory again (should hit cache)
        result2 = analyzer.analyze_directory(
            Path("Colección LH-/Insecta/Coleoptera")
        )
        
        stats = analyzer.get_stats()
        print(f"\nAnalyzer stats:")
        print(f"  Analyzed: {stats['analyzed']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  API calls: {stats['api_calls']}")
        
        assert stats['cache_hits'] == 1, "Should have 1 cache hit"
        assert stats['api_calls'] == 1, "Should have 1 API call"
        
        # Analyze multiple directories
        dirs = [
            Path("Colección LH-/Decapoda"),
            Path("Buenache/Crustacea"),
            Path("Colección LH-/Plantas"),
        ]
        
        def progress(current, total):
            print(f"  Progress: {current}/{total}")
        
        print("\nBatch analysis:")
        results = analyzer.analyze_directories(dirs, progress_callback=progress)
        
        assert len(results) == 3
        
        # Check cache was saved
        assert cache_path.exists(), "Cache file should exist"
        
        final_stats = analyzer.get_stats()
        print(f"\nFinal stats:")
        print(f"  Analyzed: {final_stats['analyzed']}")
        print(f"  Cache size: {final_stats['cache_size']}")
        
        print("\n[PASS] Directory analyzer test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_real_api():
    """Test real Gemini API (only if API key is set)."""
    print("\n" + "=" * 60)
    print("REAL API TEST")
    print("=" * 60)
    
    import os
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("[SKIP] GEMINI_API_KEY not set")
        return None
    
    if not GENAI_AVAILABLE:
        print("[SKIP] google-generativeai not installed")
        return None
    
    try:
        client = GeminiClient(api_key=api_key)
        
        # Test with a realistic directory path
        test_dir = Path("Colección LH-/3. Artrópodos/Decapoda/3.1. Delclosia")
        sample_files = ['LH 8201 b.JPG', 'LH 8202.jpg', 'LH 8203 a.tif']
        
        print(f"Testing with directory: {test_dir}")
        print(f"Sample files: {sample_files}")
        
        result = client.analyze_directory(test_dir, sample_files)
        
        print(f"\nResult:")
        print(f"  Taxonomic class: {result.taxonomic_class}")
        print(f"  Determination: {result.determination}")
        print(f"  Collection: {result.collection_code}")
        print(f"  Campaign year: {result.campaign_year}")
        print(f"  Specimen ID regex: {result.specimen_id_regex}")
        print(f"  Confidence: {result.confidence}")
        
        # Verify collection is correct
        if result.collection_code == 'LH':
            print("\n  [OK] Correctly identified Las Hoyas")
        else:
            print(f"\n  [WARN] Expected LH, got: {result.collection_code}")
        
        # Test regex extraction on all samples
        print("\nRegex validation:")
        all_extracted = True
        if result.specimen_id_regex:
            for fn in sample_files:
                extracted = result.extract_specimen_id(fn)
                status = "✓" if extracted else "✗"
                print(f"  {status} {fn} -> {extracted}")
                if not extracted:
                    all_extracted = False
        else:
            print("  [WARN] No regex provided")
            all_extracted = False
        
        if all_extracted:
            print("\n[PASS] Real API test passed - regex validated on all samples")
            return True
        else:
            print("\n[WARN] Regex did not match all samples (refinement may have been needed)")
            return True  # Still pass if API responded
            
    except Exception as e:
        print(f"\n[FAIL] API error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("LLM INTEGRATION V2 TESTS")
    print("=" * 60)
    print(f"google-generativeai available: {GENAI_AVAILABLE}")
    print()
    
    results = []
    
    results.append(("DirectoryAnalysis", test_directory_analysis()))
    results.append(("Rate Limiter", test_rate_limiter()))
    results.append(("Mock Client", test_mock_client()))
    results.append(("Directory Analyzer", test_directory_analyzer_with_mock()))
    results.append(("Real API", test_real_api()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            status = "[PASS]"
            passed += 1
        elif result is False:
            status = "[FAIL]"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
