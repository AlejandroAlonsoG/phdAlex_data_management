"""
Phase 2: Pattern Extraction Tests
Tests for the pattern_extractor module.
"""
import sys
from pathlib import Path

# Add parent to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_specimen_id_extractor():
    """Test specimen ID extraction."""
    from data_ordering.pattern_extractor import SpecimenIdExtractor
    
    print("\nTesting SpecimenIdExtractor...")
    extractor = SpecimenIdExtractor(['LH', 'MUPA', 'YCLH', 'MCCM'])
    
    # Test cases: (path_string, expected_prefix, expected_numeric, expected_plate)
    test_cases = [
        # LH prefix (2 chars - most common)
        ("LH-12345678-a.jpg", "LH", "12345678", "a"),
        ("LH-87654321.tif", "LH", "87654321", None),
        # MCCM prefix
        ("MCCM-12345678-b.jpg", "MCCM", "12345678", "b"),
        # Standard MUPA/YCLH format
        ("MUPA-12345678-a.jpg", "MUPA", "12345678", "a"),
        ("YCLH-87654321-b.tif", "YCLH", "87654321", "b"),
        # Without plate
        ("MUPA-12345678.jpg", "MUPA", "12345678", None),
        # With numeric plate
        ("LH-12345678-01.jpg", "LH", "12345678", "01"),
        # Underscores instead of dashes
        ("LH_12345678_a.jpg", "LH", "12345678", "a"),
        # Mixed case
        ("lh-12345678-A.jpg", "LH", "12345678", "a"),
        # In path
        ("C:/data/LH/campaign2023/12345678-a.jpg", "LH", "12345678", "a"),
        # With spaces (shouldn't break)
        ("LH - 12345678 - a.jpg", "LH", "12345678", "a"),
    ]
    
    passed = 0
    failed = 0
    
    for path_str, exp_prefix, exp_numeric, exp_plate in test_cases:
        path = Path(path_str)
        result = extractor.extract(path)
        
        if result is None:
            print(f"  ✗ {path_str}: No match found (expected {exp_prefix}-{exp_numeric})")
            failed += 1
            continue
        
        if (result.prefix == exp_prefix and 
            result.numeric_part == exp_numeric and 
            result.plate == exp_plate):
            print(f"  ✓ {path_str}: {result.specimen_id} (confidence: {result.confidence:.2f})")
            passed += 1
        else:
            print(f"  ✗ {path_str}: Got {result.prefix}-{result.numeric_part}-{result.plate}, "
                  f"expected {exp_prefix}-{exp_numeric}-{exp_plate}")
            failed += 1
    
    # Test case that should NOT match
    no_match_cases = [
        "random_file.jpg",
        "DSC_1234.jpg",
        "IMG_00001234.jpg",
    ]
    
    for path_str in no_match_cases:
        path = Path(path_str)
        result = extractor.extract(path)
        if result is None:
            print(f"  ✓ {path_str}: Correctly returned no match")
            passed += 1
        else:
            print(f"  ✗ {path_str}: Incorrectly matched as {result.specimen_id}")
            failed += 1
    
    print(f"\nSpecimen ID tests: {passed} passed, {failed} failed")
    return failed == 0


def test_numeric_id_extractor():
    """Test numeric ID extraction."""
    from data_ordering.pattern_extractor import NumericIdExtractor
    
    print("\nTesting NumericIdExtractor...")
    extractor = NumericIdExtractor()
    
    # Test cases: (filename, expected_camera_numbers, expected_other_numbers)
    test_cases = [
        ("DSC_1234.jpg", ["1234"], []),
        ("IMG_0001.jpg", ["0001"], []),
        ("specimen_12345678.jpg", [], ["12345678"]),
        ("DSC_1234_specimen_87654321.jpg", ["1234"], ["87654321"]),
        ("MUPA-12345678-a.jpg", [], ["12345678"]),
        ("photo_20231015_143022.jpg", [], ["20231015", "143022"]),
    ]
    
    passed = 0
    failed = 0
    
    for filename, exp_camera, exp_other in test_cases:
        results = extractor.extract_all(Path(filename).stem)
        
        camera_nums = [r.numeric_id for r in results if r.is_likely_camera_number]
        other_nums = [r.numeric_id for r in results if not r.is_likely_camera_number]
        
        if set(camera_nums) == set(exp_camera) and set(other_nums) == set(exp_other):
            print(f"  ✓ {filename}: camera={camera_nums}, other={other_nums}")
            passed += 1
        else:
            print(f"  ✗ {filename}: Got camera={camera_nums}, other={other_nums}, "
                  f"expected camera={exp_camera}, other={exp_other}")
            failed += 1
    
    print(f"\nNumeric ID tests: {passed} passed, {failed} failed")
    return failed == 0


def test_date_extractor():
    """Test date extraction."""
    from data_ordering.pattern_extractor import DateExtractor
    
    print("\nTesting DateExtractor...")
    extractor = DateExtractor()
    
    # Test cases: (path_string, expected_year, expected_month, expected_day)
    test_cases = [
        # YYYYMMDD format
        ("photo_20231015.jpg", 2023, 10, 15),
        # YYYY-MM-DD format
        ("photo_2023-10-15.jpg", 2023, 10, 15),
        # Just year in path
        ("C:/data/2023/photo.jpg", 2023, None, None),
        # Campaign year
        ("C:/data/campaign_2022/photo.jpg", 2022, None, None),
    ]
    
    passed = 0
    failed = 0
    
    for path_str, exp_year, exp_month, exp_day in test_cases:
        path = Path(path_str)
        results = extractor.extract_all(path)
        
        # Find a matching date
        found = False
        for r in results:
            if r.year == exp_year and r.month == exp_month and r.day == exp_day:
                print(f"  ✓ {path_str}: {r.to_yyyymmdd()}")
                found = True
                passed += 1
                break
        
        if not found:
            dates_found = [(r.year, r.month, r.day) for r in results]
            print(f"  ✗ {path_str}: Expected ({exp_year}, {exp_month}, {exp_day}), got {dates_found}")
            failed += 1
    
    print(f"\nDate extraction tests: {passed} passed, {failed} failed")
    return failed == 0


def test_path_analyzer():
    """Test path analysis for taxonomy and campaign."""
    from data_ordering.pattern_extractor import PathAnalyzer
    
    print("\nTesting PathAnalyzer...")
    analyzer = PathAnalyzer()
    
    # Test cases: (path_string, expected_taxonomy_levels, expected_campaign_year)
    test_cases = [
        ("C:/data/Arthropoda/Insecta/Coleoptera/specimen.jpg", 
         ['phylum', 'class', 'order'], None),
        ("C:/data/Mollusca/Gastropoda/specimen.jpg",
         ['phylum', 'class'], None),
        ("C:/data/2023/specimen.jpg",
         [], 2023),
        ("C:/data/campaign_2022/Insecta/specimen.jpg",
         ['class'], 2022),
        ("C:/data/Scarabaeidae/specimen.jpg",
         ['family'], None),  # -idae suffix
    ]
    
    passed = 0
    failed = 0
    
    for path_str, exp_levels, exp_campaign in test_cases:
        path = Path(path_str)
        taxonomy, campaign = analyzer.analyze(path)
        
        levels = [t.level for t in taxonomy]
        campaign_year = campaign.year if campaign else None
        
        if set(levels) == set(exp_levels) and campaign_year == exp_campaign:
            print(f"  ✓ {path_str}")
            print(f"      Taxonomy: {[f'{t.level}={t.value}' for t in taxonomy]}")
            if campaign:
                print(f"      Campaign: {campaign.year}")
            passed += 1
        else:
            print(f"  ✗ {path_str}")
            print(f"      Expected levels={exp_levels}, campaign={exp_campaign}")
            print(f"      Got levels={levels}, campaign={campaign_year}")
            failed += 1
    
    print(f"\nPath analyzer tests: {passed} passed, {failed} failed")
    return failed == 0


def test_pattern_extractor():
    """Test the combined pattern extractor."""
    from data_ordering.pattern_extractor import PatternExtractor
    
    print("\nTesting PatternExtractor (combined)...")
    extractor = PatternExtractor(['LH', 'MUPA', 'YCLH', 'MCCM'])
    
    # Complex test paths
    test_cases = [
        "C:/data/LH/2023/Arthropoda/Insecta/LH-12345678-a.jpg",
        "C:/data/MCCM/campaign_2022/Mollusca/MCCM-87654321.tif",
        "C:/data/unknown/DSC_1234.jpg",
    ]
    
    for path_str in test_cases:
        print(f"\n  Path: {path_str}")
        result = extractor.extract(Path(path_str))
        
        if result.specimen_id:
            print(f"    Specimen ID: {result.specimen_id.specimen_id} "
                  f"(confidence: {result.specimen_id.confidence:.2f})")
        else:
            print(f"    Specimen ID: None")
        
        if result.numeric_ids:
            nums = [(n.numeric_id, 'camera' if n.is_likely_camera_number else 'other') 
                    for n in result.numeric_ids]
            print(f"    Numeric IDs: {nums}")
        
        if result.dates:
            print(f"    Dates: {[d.to_yyyymmdd() for d in result.dates]}")
        
        if result.taxonomy_hints:
            print(f"    Taxonomy: {[f'{t.level}={t.value}' for t in result.taxonomy_hints]}")
        
        if result.campaign:
            print(f"    Campaign: {result.campaign.year}")
    
    print("\n✓ Pattern extractor integration test complete")
    return True


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 50)
    print("Phase 2: Pattern Extraction Tests")
    print("=" * 50)
    
    results = []
    
    # Run each test
    results.append(("Specimen ID Extractor", test_specimen_id_extractor()))
    results.append(("Numeric ID Extractor", test_numeric_id_extractor()))
    results.append(("Date Extractor", test_date_extractor()))
    results.append(("Path Analyzer", test_path_analyzer()))
    results.append(("Pattern Extractor (Combined)", test_pattern_extractor()))
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
