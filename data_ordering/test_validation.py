"""
Validation test against real directory structure from user's data.
This tests the pattern extractor against actual filenames and paths.
"""
from pathlib import Path
import sys

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.pattern_extractor import (
    PatternExtractor, SpecimenIdExtractor, DateExtractor, 
    PathAnalyzer, NumericIdExtractor
)


def test_specimen_id_real_examples():
    """Test specimen ID extraction against real filenames from user's data."""
    
    # Real examples from the directory tree - should extract specimen IDs
    positive_examples = [
        # LH prefix variants
        ("LH-15083_a_.jpg", "LH", "15083", "a"),
        ("LH-6120.jpg", "LH", "6120", None),
        ("LH-7335_a_.jpg", "LH", "7335", "a"),
        ("LH-7335_b_.jpg", "LH", "7335", "b"),
        ("LH 30202 AB.JPG", "LH", "30202", "ab"),  # AB is valid plate
        ("LH-16059b.JPG", "LH", "16059", "b"),
        ("LH 16059 b.JPG", "LH", "16059", "b"),
        ("LH-07239a.JPG", "LH", "07239", "a"),
        ("LH 2442 (1).JPG", "LH", "2442", None),  # (1) is sequence, not plate
        ("LH-09645b Hispanamia newbreyi .JPG", "LH", "09645", "b"),
        ("LH_9300_20130704 (1).JPG", "LH", "9300", None),
        ("LH32560AB.JPG", "LH", "32560", "ab"),  # No separator, AB plate
        ("LH-09300.JPG", "LH", "09300", None),
        ("LH-11388 Gracilibatrachus.jpg", "LH", "11388", None),
        ("LH-928061.JPG", "LH", "928061", None),  # 6 digits
        ("LH-20265 (23).JPG", "LH", "20265", None),
        
        # MCCM-LH compound prefix
        ("MCCM-LH 26452A.JPG", "MCCM-LH", "26452", "a"),  # Compound prefix
        ("MCCM-LH 26071B.JPG", "MCCM-LH", "26071", "b"),
        ("MCCM-LH 20200A.JPG", "MCCM-LH", "20200", "a"),
        ("MCCM-LH 7008B.JPG", "MCCM-LH", "7008", "b"),
        ("MCCM-LH 928045A.JPG", "MCCM-LH", "928045", "a"),
        
        # PB prefix (new discovery!)
        ("PB 7005b.JPG", "PB", "7005", "b"),
        ("PB 22184b.JPG", "PB", "22184", "b"),
        ("PB 26134b.JPG", "PB", "26134", "b"),
        ("PB 13282a.JPG", "PB", "13282", "a"),
        ("PB 2580.JPG", "PB", "2580", None),
        
        # K-Bue and Cer-Bue prefixes (Buenache collection)
        ("K-Bue 085 (1).JPG", "K-Bue", "085", None),
        ("Cer-Bue 125.JPG", "Cer-Bue", "125", None),
        ("K-Bue 170 (1).JPG", "K-Bue", "170", None),
        ("Cer-Bue 292 AB.JPG", "Cer-Bue", "292", "ab"),
        
        # Numeric-only with context (these need path context)
        ("22270A.JPG", None, "22270", "a"),  # No prefix in filename
        ("26652B.JPG", None, "26652", "b"),
    ]
    
    # Camera-generated filenames - should NOT extract as specimen IDs
    negative_examples = [
        "DSC00016.jpg",
        "DSC_0495.JPG",
        "DSC_3163.JPG",
        "IMG_1171.JPG",
        "IMG_0823.JPG",
        "IMG_2277.JPG",
        "PA210066.JPG",  # Olympus camera
        "PB182509.JPG",  # Olympus camera (looks like PB prefix but isn't)
        "PC200057.JPG",  # Olympus camera
        "P1103168.JPG",  # Camera
        "_DSC1935.JPG",
        "DSCN1394.JPG",
    ]
    
    extractor = SpecimenIdExtractor(known_prefixes=['LH', 'MUPA', 'YCLH', 'MCCM', 'PB', 'K-Bue', 'Cer-Bue', 'MCCM-LH'])
    
    print("=" * 60)
    print("SPECIMEN ID EXTRACTION - POSITIVE EXAMPLES")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for example in positive_examples:
        if len(example) == 4:
            filename, expected_prefix, expected_num, expected_plate = example
        else:
            filename, expected_prefix, expected_num = example
            expected_plate = None
        
        path = Path(f"/test/{filename}")
        result = extractor.extract(path)
        
        if result:
            # Check if extraction is reasonable
            if expected_prefix is None or expected_prefix.upper() in result.prefix.upper():
                print(f"✓ {filename}")
                print(f"  → ID: {result.specimen_id}, prefix: {result.prefix}, num: {result.numeric_part}, plate: {result.plate}")
                passed += 1
            else:
                print(f"✗ {filename}")
                print(f"  Expected prefix: {expected_prefix}, got: {result.prefix}")
                failed += 1
        else:
            if expected_prefix is None:
                # Numeric-only without path context - might not match
                print(f"? {filename} - no match (may need path context)")
                passed += 1
            else:
                print(f"✗ {filename} - NO MATCH (expected {expected_prefix}-{expected_num})")
                failed += 1
    
    print(f"\nPositive examples: {passed} passed, {failed} failed")
    
    print("\n" + "=" * 60)
    print("SPECIMEN ID EXTRACTION - NEGATIVE EXAMPLES (should NOT match)")
    print("=" * 60)
    
    camera_passed = 0
    camera_failed = 0
    
    for filename in negative_examples:
        path = Path(f"/test/{filename}")
        result = extractor.extract(path)
        
        if result is None:
            print(f"✓ {filename} - correctly rejected")
            camera_passed += 1
        else:
            print(f"✗ {filename} - INCORRECTLY MATCHED: {result.specimen_id}")
            camera_failed += 1
    
    print(f"\nNegative examples: {camera_passed} passed, {camera_failed} failed")
    
    return passed, failed, camera_passed, camera_failed


def test_path_taxonomy():
    """Test taxonomy extraction from real path examples."""
    
    paths = [
        "Colección LH-/1. Plantas/Angiospermas/22270A.JPG",
        "Colección LH-/3. Artrópodos/Decapoda/3.1. Delclosia/LH 8201 b.JPG",
        "Colección LH-/4. Peces/Amiiformes/LH-09645b Hispanamia newbreyi .JPG",
        "Colección LH-/5. Tetrápodos/Anfibios/Anura/LH-11388 Gracilibatrachus.jpg",
        "Colección LH-/5. Tetrápodos/Aves/LH-00022 Iberomesornis/Iberomesornis (7).JPG",
        "Colección LH-/5. Tetrápodos/Reptiles/Meyasaurus/LH-445.JPG",
        "Fotos/FILICALES/Weichselia reticulata 02/PB 2812a.JPG",
        "Fotos/PERACARIDA/Spelaeogriphaceae indet.01/PB 15247a.JPG",
        "Fotos/Fotos/TELEOSTEI 126/PB 28422a.JPG",
        "4fotografia_grupos/Archosauria_LH_fotos/AvesLH/Concornis/Cocnrnis1.tif",
        "4fotografia_grupos/Catálogo fotografico insectos Las Hoyas/Iberonepa romerali/LH 1253/LH 1253a/P1040699.TIF",
        "Colección Buenache/Buenache 2017/Cantera/Crustacea/K-Bue 084.JPG",
        "Fotos Las Hoyas/FotosLH_grupos/Lissamfibia/Salentia_2016_CatalogoMCCM/3200.jpg",
    ]
    
    analyzer = PathAnalyzer()
    
    print("\n" + "=" * 60)
    print("TAXONOMY HINTS FROM PATHS")
    print("=" * 60)
    
    for path_str in paths:
        path = Path(path_str)
        hints = analyzer.find_taxonomy_hints(path)
        print(f"\n{path_str}")
        if hints:
            for hint in hints:
                print(f"  → {hint.level}: {hint.value}")
        else:
            print("  → No taxonomy hints found")


def test_date_extraction():
    """Test date extraction from real examples."""
    
    examples = [
        ("LH_9300_20130704 (1).JPG", 2013, 7, 4),
        ("LH_22385_20130828 (1).JPG", 2013, 8, 28),
        ("pez sin sigla 20131106 (1).JPG", 2013, 11, 6),
        ("Spinolestes  20120120", 2012, 1, 20),
    ]
    
    path_examples = [
        ("18Nov/IMG_1_18Nov_/LH-15083_a_.jpg", None, 11, 18),  # 18Nov folder
        ("Abel_21_1_2008/PA210089.JPG", 2008, 1, 21),
        ("Laboratorio_MCCM_03_05_07_/IMG_2224.jpg", 2007, 5, 3),
        ("Dino_Prep_/Img_Dino_Prep_06_01_11_/PC143066.JPG", 2011, 1, None),  # Ambiguous
        ("2010-10-18  01 Primer DíaTiffs/Imag 02.tif", 2010, 10, 18),
        ("Fotos 2009/FILICALES/Weichselia reticulata 02/PB 6216.JPG", 2009, None, None),
    ]
    
    date_extractor = DateExtractor()
    
    print("\n" + "=" * 60)
    print("DATE EXTRACTION FROM FILENAMES")
    print("=" * 60)
    
    for filename, exp_year, exp_month, exp_day in examples:
        # Use extract_all with a fake path to extract dates
        dates = date_extractor.extract_all(Path(filename))
        print(f"\n{filename}")
        if dates:
            for d in dates:
                print(f"  → {d.year}-{d.month or '?'}-{d.day or '?'} (raw: {d.raw_match})")
        else:
            print("  → No dates found")
    
    print("\n" + "=" * 60)
    print("DATE/CAMPAIGN YEAR FROM PATHS")
    print("=" * 60)
    
    analyzer = PathAnalyzer()
    
    for path_str, exp_year, exp_month, exp_day in path_examples:
        path = Path(path_str)
        _, campaign = analyzer.analyze(path)  # Use analyze() instead of find_campaign_year()
        print(f"\n{path_str}")
        if campaign:
            print(f"  → Campaign year: {campaign.year} (from: {campaign.raw_match})")
        else:
            print("  → No campaign year found")


def test_numeric_id():
    """Test numeric ID extraction distinguishing camera vs specimen numbers."""
    
    extractor = NumericIdExtractor()
    
    examples = [
        # (filename, should_be_camera)
        ("DSC00016.jpg", True),
        ("IMG_1171.JPG", True),
        ("PA210066.JPG", True),
        ("PB182509.JPG", True),  # Camera code, not PB prefix!
        ("LH-15083_a_.jpg", False),  # Specimen
        ("22270A.JPG", False),  # Specimen number
        ("3200.jpg", False),  # Catalog number
    ]
    
    print("\n" + "=" * 60)
    print("NUMERIC ID EXTRACTION (Camera vs Specimen)")
    print("=" * 60)
    
    for filename, expected_camera in examples:
        results = extractor.extract_all(filename)
        print(f"\n{filename}")
        for r in results:
            camera_str = "CAMERA" if r.is_likely_camera_number else "SPECIMEN"
            status = "✓" if r.is_likely_camera_number == expected_camera else "✗"
            print(f"  {status} {r.numeric_id} → {camera_str} (raw: {r.raw_match})")


def test_full_extraction():
    """Test complete extraction pipeline with real paths."""
    
    extractor = PatternExtractor(
        known_prefixes=['LH', 'MUPA', 'YCLH', 'MCCM', 'PB', 'K-Bue', 'Cer-Bue', 'MCCM-LH']
    )
    
    paths = [
        "Colección LH-/4. Peces/Amiiformes/LH-09645b Hispanamia newbreyi .JPG",
        "Colección LH-/5. Tetrápodos/Anfibios/Anura/LH-11388 Gracilibatrachus.jpg",
        "18Nov/IMG_1_18Nov_/LH-15083_a_.jpg",
        "Colección LH-/7. Coprolitos/MCCM-LH 26452A.JPG",
        "Fotos/PERACARIDA/Spelaeogriphaceae indet.01/PB 15247a.JPG",
        "Colección Buenache/Buenache 2017/Cantera/Crustacea/K-Bue 084.JPG",
        "Fotos 2009/FILICALES/Weichselia reticulata 02/PB 6216.JPG",
    ]
    
    print("\n" + "=" * 60)
    print("FULL EXTRACTION PIPELINE")
    print("=" * 60)
    
    for path_str in paths:
        path = Path(path_str)
        result = extractor.extract(path)
        
        print(f"\n{path_str}")
        print(f"  Specimen ID: {result.specimen_id.specimen_id if result.specimen_id else 'None'}")
        print(f"  Campaign: {result.campaign.year if result.campaign else 'None'}")
        print(f"  Taxonomy: {[h.value for h in result.taxonomy_hints] if result.taxonomy_hints else 'None'}")


if __name__ == "__main__":
    print("VALIDATION TEST AGAINST REAL DATA")
    print("=" * 70)
    
    pos_pass, pos_fail, neg_pass, neg_fail = test_specimen_id_real_examples()
    test_path_taxonomy()
    test_date_extraction()
    test_numeric_id()
    test_full_extraction()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Specimen ID Positive: {pos_pass} passed, {pos_fail} failed")
    print(f"Specimen ID Negative: {neg_pass} passed, {neg_fail} failed")
    print(f"Total specimen tests: {pos_pass + pos_fail + neg_pass + neg_fail}")
