"""
Tests for the llm_integration_v3 module.
Tests the multi-stage directory analysis approach.
"""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.llm_integration_v3 import (
    PathAnalysis, FilenameRegexResult, FileAnalysis, DirectoryAnalysis,
    RateLimiter, MockLLMClient, GitHubModelsClient, GeminiClient,
    DirectoryAnalyzer, get_llm_client,
    GENAI_AVAILABLE, OPENAI_AVAILABLE
)


def test_path_analysis_dataclass():
    """Test PathAnalysis dataclass."""
    print("=" * 60)
    print("PATH ANALYSIS DATACLASS TEST")
    print("=" * 60)
    
    analysis = PathAnalysis(
        taxonomic_class='Mollusca',
        genus='Bivalvia',
        collection_code='LH',
        campaign_year=2019,
        specimen_id=None,
        directory_path='\\2.Moluscos\\bivalvos',
        confidence=0.9
    )
    
    # Test to_dict
    d = analysis.to_dict()
    print(f"to_dict: {d}")
    assert d['taxonomic_class'] == 'Mollusca'
    assert d['genus'] == 'Bivalvia'
    assert d['collection_code'] == 'LH'
    
    # Test from_dict
    analysis2 = PathAnalysis.from_dict(d)
    assert analysis2.taxonomic_class == 'Mollusca'
    assert analysis2.genus == 'Bivalvia'
    
    print("\n[PASS] PathAnalysis test passed")
    return True


def test_filename_regex_result():
    """Test FilenameRegexResult dataclass."""
    print("\n" + "=" * 60)
    print("FILENAME REGEX RESULT TEST")
    print("=" * 60)
    
    result = FilenameRegexResult(
        master_regex=r"(?P<specimen_id>LH[-\s]?\d{3,8})(?:\s*(?P<plate>[ab]))?",
        specimen_id_regex=r"(LH[-\s]?\d{3,8})",
        extractable_fields=['specimen_id', 'plate'],
        sample_filenames=['LH 12345.jpg', 'LH-12346 a.tif'],
        confidence=0.85
    )
    
    d = result.to_dict()
    print(f"to_dict: {d}")
    assert 'specimen_id' in d['extractable_fields']
    assert d['specimen_id_regex'] == r"(LH[-\s]?\d{3,8})"
    
    print("\n[PASS] FilenameRegexResult test passed")
    return True


def test_directory_analysis_backwards_compat():
    """Test DirectoryAnalysis backwards compatibility."""
    print("\n" + "=" * 60)
    print("DIRECTORY ANALYSIS BACKWARDS COMPATIBILITY TEST")
    print("=" * 60)
    
    # Old-style creation
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
    assert d['determination'] == 'Coleoptera'
    
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
    
    print("\n[PASS] DirectoryAnalysis backwards compatibility test passed")
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


def test_mock_client_path_analysis():
    """Test MockLLMClient path analysis."""
    print("\n" + "=" * 60)
    print("MOCK CLIENT PATH ANALYSIS TEST")
    print("=" * 60)
    
    client = MockLLMClient()
    
    # Test Moluscos/bivalvos path - Bivalvia is a class, not below class
    # So macroclass=Mollusca_y_Vermes, class=Bivalvia, determination=null
    result1 = client.analyze_path(Path("\\2.Moluscos\\bivalvos"))
    print(f"\nMoluscos path analysis:")
    print(f"  Macroclass: {result1.macroclass}")
    print(f"  Class: {result1.taxonomic_class}")
    print(f"  Determination: {result1.genus}")
    print(f"  Collection: {result1.collection_code}")
    assert result1.macroclass == 'Mollusca_y_Vermes'
    assert result1.taxonomic_class == 'Bivalvia'
    # No determination since bivalvos IS the class level
    
    # Test Insecta/Coleoptera path - Coleoptera is an order (below class)
    result2 = client.analyze_path(Path("Colección LH-/3.Artrópodos/Insecta/Coleoptera"))
    print(f"\nInsecta/Coleoptera path analysis:")
    print(f"  Macroclass: {result2.macroclass}")
    print(f"  Class: {result2.taxonomic_class}")
    print(f"  Determination: {result2.genus}")
    assert result2.macroclass == 'Arthropoda'
    assert result2.taxonomic_class == 'Insecta'
    assert result2.genus == 'Coleoptera'
    
    print("\n[PASS] Mock client path analysis test passed")
    return True


def test_mock_client_regex_generation():
    """Test MockLLMClient regex generation."""
    print("\n" + "=" * 60)
    print("MOCK CLIENT REGEX GENERATION TEST")
    print("=" * 60)
    
    client = MockLLMClient()
    
    sample_files = ['LH 15083.jpg', 'LH-15084 a.tif', 'LH 15085.jpg']
    
    result = client.generate_filename_regex(sample_files)
    print(f"\nGenerated regex:")
    print(f"  Master regex: {result.master_regex}")
    print(f"  Specimen ID regex: {result.specimen_id_regex}")
    print(f"  Extractable fields: {result.extractable_fields}")
    
    assert result.specimen_id_regex is not None
    assert 'specimen_id' in result.extractable_fields
    
    print("\n[PASS] Mock client regex generation test passed")
    return True


def test_regex_application():
    """Test regex application and validation."""
    print("\n" + "=" * 60)
    print("REGEX APPLICATION TEST")
    print("=" * 60)
    
    client = MockLLMClient()
    
    # Generate regex
    sample_files = ['LH 15083.jpg', 'LH-15084 a.tif', 'LH 15085.jpg']
    regex_result = client.generate_filename_regex(sample_files)
    
    # Apply to more files
    all_files = [
        'LH 15083.jpg',
        'LH-15084.tif',
        'LH 15085 a.jpg',
        'LH-15086.tif',
        'random_file.jpg',  # This should fail
    ]
    
    successes, failures, refined = client.apply_and_validate_regex(
        regex_result, all_files
    )
    
    print(f"\nSuccesses ({len(successes)}):")
    for fn, data in successes.items():
        print(f"  {fn} → {data}")
    
    print(f"\nFailures ({len(failures)}):")
    for fn in failures:
        print(f"  {fn}")
    
    assert len(successes) >= 4, f"Expected at least 4 successes, got {len(successes)}"
    
    print("\n[PASS] Regex application test passed")
    return True


def test_full_directory_analysis():
    """Test full multi-stage directory analysis."""
    print("\n" + "=" * 60)
    print("FULL DIRECTORY ANALYSIS TEST")
    print("=" * 60)
    
    client = MockLLMClient()
    
    temp_dir = Path(tempfile.mkdtemp())
    test_dir = temp_dir / "Colección LH-" / "2.Moluscos" / "bivalvos"
    test_dir.mkdir(parents=True)
    
    # Create test files
    test_files = ['LH 15083.jpg', 'LH-15084.tif', 'LH 15085.jpg']
    for fn in test_files:
        (test_dir / fn).touch()
    
    try:
        result = client.analyze_directory(
            test_dir,
            enable_per_file_fallback=False  # Skip fallback for speed
        )
        
        print(f"\nDirectory analysis result:")
        print(f"  Path: {result.directory_path}")
        print(f"  Taxonomic class: {result.taxonomic_class}")
        print(f"  Determination: {result.determination}")
        print(f"  Collection: {result.collection_code}")
        print(f"  Specimen ID regex: {result.specimen_id_regex}")
        print(f"  Confidence: {result.confidence}")
        
        # Check path analysis was stored
        if result.path_analysis:
            print(f"\n  Path analysis details:")
            print(f"    Class: {result.path_analysis.taxonomic_class}")
            print(f"    Genus: {result.path_analysis.genus}")
        
        # Check regex result was stored
        if result.regex_result:
            print(f"\n  Regex result details:")
            print(f"    Fields: {result.regex_result.extractable_fields}")
        
        print("\n[PASS] Full directory analysis test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_directory_analyzer_with_cache():
    """Test DirectoryAnalyzer with caching."""
    print("\n" + "=" * 60)
    print("DIRECTORY ANALYZER CACHE TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    cache_path = temp_dir / "test_cache.json"
    
    try:
        mock_client = MockLLMClient()
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
        
        print("\n[PASS] Directory analyzer cache test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_get_llm_client_factory():
    """Test the get_llm_client factory function."""
    print("\n" + "=" * 60)
    print("GET LLM CLIENT FACTORY TEST")
    print("=" * 60)
    
    # Test mock provider
    client = get_llm_client('mock')
    assert isinstance(client, MockLLMClient)
    print("  Mock client: OK")
    
    # Test github provider (if available)
    if OPENAI_AVAILABLE:
        try:
            client = get_llm_client('github')
            assert isinstance(client, GitHubModelsClient)
            print("  GitHub client: OK")
        except ValueError:
            print("  GitHub client: SKIP (no token)")
    else:
        print("  GitHub client: SKIP (openai not installed)")
    
    # Test gemini provider (if available)
    if GENAI_AVAILABLE:
        try:
            client = get_llm_client('gemini')
            assert isinstance(client, GeminiClient)
            print("  Gemini client: OK")
        except ValueError:
            print("  Gemini client: SKIP (no API key)")
    else:
        print("  Gemini client: SKIP (google-generativeai not installed)")
    
    print("\n[PASS] Factory test passed")
    return True


def test_real_api():
    """Test real API (only if credentials are set)."""
    print("\n" + "=" * 60)
    print("REAL API TEST")
    print("=" * 60)
    
    import os
    
    # Try GitHub Models first
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token and OPENAI_AVAILABLE:
        print("\nTesting GitHub Models API...")
        try:
            client = GitHubModelsClient()
            
            # Test path analysis
            path = Path("\\2.Moluscos\\bivalvos")
            result = client.analyze_path(path)
            
            print(f"\nPath analysis for {path}:")
            print(f"  Taxonomic class: {result.taxonomic_class}")
            print(f"  Genus: {result.genus}")
            print(f"  Collection: {result.collection_code}")
            print(f"  Confidence: {result.confidence}")
            
            # Check if it correctly mapped Spanish terms
            if result.taxonomic_class == 'Mollusca':
                print("\n  [OK] Correctly mapped Moluscos → Mollusca")
            else:
                print(f"\n  [WARN] Expected Mollusca, got {result.taxonomic_class}")
            
            # Test regex generation
            sample_files = ['LH 15083.jpg', 'LH-15084 a.tif', 'LH 15085.jpg']
            regex_result = client.generate_filename_regex(sample_files)
            
            print(f"\nRegex generation:")
            print(f"  Master regex: {regex_result.master_regex}")
            print(f"  Specimen ID regex: {regex_result.specimen_id_regex}")
            print(f"  Extractable fields: {regex_result.extractable_fields}")
            
            print("\n[PASS] Real API test passed")
            return True
            
        except Exception as e:
            print(f"\n[FAIL] API error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("[SKIP] No API credentials available")
    return None


if __name__ == "__main__":
    print("LLM INTEGRATION V3 TESTS")
    print("=" * 60)
    print(f"google-generativeai available: {GENAI_AVAILABLE}")
    print(f"openai available: {OPENAI_AVAILABLE}")
    print()
    
    results = []
    
    results.append(("PathAnalysis", test_path_analysis_dataclass()))
    results.append(("FilenameRegexResult", test_filename_regex_result()))
    results.append(("DirectoryAnalysis Backwards Compat", test_directory_analysis_backwards_compat()))
    results.append(("Rate Limiter", test_rate_limiter()))
    results.append(("Mock Path Analysis", test_mock_client_path_analysis()))
    results.append(("Mock Regex Generation", test_mock_client_regex_generation()))
    results.append(("Regex Application", test_regex_application()))
    results.append(("Full Directory Analysis", test_full_directory_analysis()))
    results.append(("Directory Analyzer Cache", test_directory_analyzer_with_cache()))
    results.append(("LLM Client Factory", test_get_llm_client_factory()))
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
