"""
Tests for the file_scanner module.
"""
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.file_scanner import FileScanner, ScannedFile, ScanProgress, ScanStatus, scan_directory
from data_ordering.file_utils import FileType
from data_ordering.image_hasher import IMAGEHASH_AVAILABLE


def create_test_directory():
    """Create a test directory structure with sample files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create subdirectories mimicking real structure
    (temp_dir / "Las Hoyas" / "Insecta" / "Coleoptera").mkdir(parents=True)
    (temp_dir / "Buenache" / "Crustacea").mkdir(parents=True)
    (temp_dir / "Camera_dumps").mkdir(parents=True)
    
    # Create test files
    files_created = []
    
    # 1. Las Hoyas specimen images
    lh_files = [
        ("Las Hoyas/Insecta/Coleoptera/LH-15083_a_.jpg", b"fake image 1"),
        ("Las Hoyas/Insecta/Coleoptera/LH-15083_b_.jpg", b"fake image 2"),
        ("Las Hoyas/LH-6120.jpg", b"fake image 3"),
        ("Las Hoyas/MCCM-LH 26452A.jpg", b"fake image 4"),
    ]
    
    # 2. Buenache specimens
    bue_files = [
        ("Buenache/Crustacea/K-Bue 085.jpg", b"fake image 5"),
        ("Buenache/PB 7005b.jpg", b"fake image 6"),
    ]
    
    # 3. Camera files (no specimen ID)
    camera_files = [
        ("Camera_dumps/DSC00016.jpg", b"camera image 1"),
        ("Camera_dumps/IMG_1234.jpg", b"camera image 2"),
    ]
    
    # 4. Exact duplicate
    duplicate_files = [
        ("Las Hoyas/duplicate1.jpg", b"duplicate content"),
        ("Camera_dumps/duplicate2.jpg", b"duplicate content"),  # Same content
    ]
    
    # 5. Text files
    text_files = [
        ("Las Hoyas/notes.txt", b"Some notes about fossils"),
        ("readme.pdf", b"PDF content here"),
    ]
    
    all_files = lh_files + bue_files + camera_files + duplicate_files + text_files
    
    for rel_path, content in all_files:
        file_path = temp_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        files_created.append(file_path)
    
    return temp_dir, files_created


def test_basic_scan():
    """Test basic file scanning."""
    print("=" * 60)
    print("BASIC SCAN TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    
    try:
        # Create scanner without hashing (faster for test)
        scanner = FileScanner(hash_images=False)
        
        # Scan
        scanned_files = list(scanner.scan_directories([temp_dir]))
        
        print(f"Total files scanned: {len(scanned_files)}")
        print(f"Expected: {len(files)}")
        
        # Check counts
        assert len(scanned_files) == len(files), f"Expected {len(files)}, got {len(scanned_files)}"
        
        # Check file types
        print(f"\nFile type breakdown:")
        print(f"  Images: {scanner.progress.images}")
        print(f"  Text: {scanner.progress.text_files}")
        print(f"  Other: {scanner.progress.other_files}")
        
        # Images: 10 (4 LH + 2 BUE + 2 camera + 2 duplicates)
        # Text: 2 (notes.txt, readme.pdf)
        assert scanner.progress.images == 10, f"Expected 10 images, got {scanner.progress.images}"
        assert scanner.progress.text_files == 2, f"Expected 2 text, got {scanner.progress.text_files}"
        
        print("\n[PASS] Basic scan test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_pattern_extraction():
    """Test that patterns are extracted during scan."""
    print("\n" + "=" * 60)
    print("PATTERN EXTRACTION TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    
    try:
        scanner = FileScanner(hash_images=False, extract_patterns=True)
        scanned_files = list(scanner.scan_directories([temp_dir]))
        
        # Check specimen IDs found
        with_specimen = [f for f in scanned_files if f.specimen_id]
        without_specimen = [f for f in scanned_files if not f.specimen_id and f.file_type == FileType.IMAGE]
        
        print(f"Files with specimen ID: {len(with_specimen)}")
        for f in with_specimen:
            print(f"  {f.path.name} -> {f.specimen_id}")
        
        print(f"\nImages without specimen ID: {len(without_specimen)}")
        for f in without_specimen:
            print(f"  {f.path.name}")
        
        # Should find: LH-15083, LH-6120, MCCM-LH-26452, K-BUE-085, PB-7005
        assert len(with_specimen) >= 5, f"Expected at least 5 specimen IDs, got {len(with_specimen)}"
        
        print("\n[PASS] Pattern extraction test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_collection_detection():
    """Test that collections are detected."""
    print("\n" + "=" * 60)
    print("COLLECTION DETECTION TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    
    try:
        scanner = FileScanner(hash_images=False)
        scanned_files = list(scanner.scan_directories([temp_dir]))
        
        # Check collections
        print(f"Collections found: {scanner.progress.by_collection}")
        
        lh_files = scanner.get_files_by_collection("LH")
        bue_files = scanner.get_files_by_collection("BUE")
        
        print(f"\nLas Hoyas (LH) files: {len(lh_files)}")
        for f in lh_files:
            print(f"  {f.path.name}")
        
        print(f"\nBuenache (BUE) files: {len(bue_files)}")
        for f in bue_files:
            print(f"  {f.path.name}")
        
        # LH should have files with LH prefix + MCCM-LH + files in "Las Hoyas" folder
        # BUE should have K-Bue and PB files
        assert len(lh_files) >= 4, f"Expected at least 4 LH files, got {len(lh_files)}"
        assert len(bue_files) >= 2, f"Expected at least 2 BUE files, got {len(bue_files)}"
        
        print("\n[PASS] Collection detection test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_duplicate_detection():
    """Test duplicate detection with hashing."""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION TEST")
    print("=" * 60)
    
    if not IMAGEHASH_AVAILABLE:
        print("[SKIP] imagehash not available")
        return None
    
    temp_dir, files = create_test_directory()
    
    try:
        scanner = FileScanner(hash_images=True)
        scanned_files = list(scanner.scan_directories([temp_dir]))
        
        duplicates = scanner.get_duplicates()
        
        print(f"Exact duplicates found: {scanner.progress.exact_duplicates}")
        print(f"Perceptual duplicates found: {scanner.progress.perceptual_duplicates}")
        
        for dup in duplicates:
            print(f"  {dup.path.name} is duplicate of {dup.duplicate_of.name if dup.duplicate_of else 'unknown'}")
        
        # We created 2 files with identical content
        assert scanner.progress.exact_duplicates >= 1, "Should find at least 1 exact duplicate"
        
        print("\n[PASS] Duplicate detection test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_llm_review_flagging():
    """Test that files needing LLM review are flagged."""
    print("\n" + "=" * 60)
    print("LLM REVIEW FLAGGING TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    
    try:
        scanner = FileScanner(hash_images=False)
        scanned_files = list(scanner.scan_directories([temp_dir]))
        
        needs_review = scanner.get_files_needing_review()
        
        print(f"Files needing LLM review: {len(needs_review)}")
        for f in needs_review:
            print(f"  {f.path.name} (specimen_id: {f.specimen_id}, collection: {f.collection})")
        
        # Should have at least 2 files needing review (those without specimen IDs)
        assert len(needs_review) >= 2, "At least 2 files should need LLM review"
        
        print("\n[PASS] LLM review flagging test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_state_save_load():
    """Test saving and loading scan state."""
    print("\n" + "=" * 60)
    print("STATE SAVE/LOAD TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    state_file = temp_dir / "scan_state.json"
    
    try:
        # Scan and save
        scanner1 = FileScanner(hash_images=False)
        list(scanner1.scan_directories([temp_dir]))
        scanner1.save_state(state_file)
        
        print(f"Saved state with {len(scanner1.files)} files")
        
        # Load into new scanner
        scanner2 = FileScanner(hash_images=False)
        loaded = scanner2.load_state(state_file)
        
        assert loaded, "Failed to load state"
        print(f"Loaded state with {len(scanner2.files)} files")
        
        # Compare
        assert len(scanner1.files) == len(scanner2.files), "File count mismatch"
        assert scanner1.progress.images == scanner2.progress.images, "Image count mismatch"
        
        print("\n[PASS] State save/load test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_summary():
    """Test scan summary generation."""
    print("\n" + "=" * 60)
    print("SUMMARY TEST")
    print("=" * 60)
    
    temp_dir, files = create_test_directory()
    
    try:
        scanner = FileScanner(hash_images=False)
        list(scanner.scan_directories([temp_dir]))
        
        summary = scanner.get_summary()
        
        print("Scan Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        assert summary['total_files'] == len(files), "Total mismatch"
        assert 'by_collection' in summary, "Missing collection breakdown"
        
        print("\n[PASS] Summary test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("FILE SCANNER TESTS")
    print("=" * 60)
    print(f"imagehash available: {IMAGEHASH_AVAILABLE}")
    print()
    
    results = []
    
    results.append(("Basic Scan", test_basic_scan()))
    results.append(("Pattern Extraction", test_pattern_extraction()))
    results.append(("Collection Detection", test_collection_detection()))
    results.append(("Duplicate Detection", test_duplicate_detection()))
    results.append(("LLM Review Flagging", test_llm_review_flagging()))
    results.append(("State Save/Load", test_state_save_load()))
    results.append(("Summary", test_summary()))
    
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
