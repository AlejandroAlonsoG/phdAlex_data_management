"""
Tests for the image_hasher module.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.image_hasher import (
    ImageHasher, HashRegistry, ImageHash, DuplicateType,
    hash_image, IMAGEHASH_AVAILABLE
)


def test_md5_hashing():
    """Test MD5 hash computation."""
    print("=" * 60)
    print("MD5 HASHING TEST")
    print("=" * 60)
    
    hasher = ImageHasher()
    
    # Create a temp file to hash
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
        f.write(b"Hello, World!")
        temp_path = Path(f.name)
    
    try:
        md5 = hasher.compute_md5(temp_path)
        print(f"Test file MD5: {md5}")
        
        # Known MD5 of "Hello, World!"
        expected = "65a8e27d8879283831b664bd8b7f0ad4"
        if md5 == expected:
            print("✓ MD5 hash matches expected value")
            return True
        else:
            print(f"✗ MD5 mismatch. Expected: {expected}")
            return False
    finally:
        temp_path.unlink()


def test_perceptual_hashing():
    """Test perceptual hash computation (requires imagehash library)."""
    print("\n" + "=" * 60)
    print("PERCEPTUAL HASHING TEST")
    print("=" * 60)
    
    if not IMAGEHASH_AVAILABLE:
        print("⚠ imagehash library not available - skipping perceptual tests")
        print("  Install with: pip install imagehash Pillow")
        return None
    
    hasher = ImageHasher()
    
    # Create a simple test image
    from PIL import Image
    import tempfile
    
    # Create a solid color image
    img = Image.new('RGB', (100, 100), color='red')
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.png') as f:
        temp_path = Path(f.name)
        img.save(temp_path)
    
    try:
        phash = hasher.compute_phash(temp_path)
        ahash = hasher.compute_ahash(temp_path)
        
        print(f"Test image pHash: {phash}")
        print(f"Test image aHash: {ahash}")
        
        if phash and ahash:
            print("✓ Perceptual hashes computed successfully")
            return True
        else:
            print("✗ Failed to compute perceptual hashes")
            return False
    finally:
        temp_path.unlink()


def test_duplicate_detection():
    """Test duplicate detection with identical and similar images."""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION TEST")
    print("=" * 60)
    
    if not IMAGEHASH_AVAILABLE:
        print("⚠ imagehash library not available - skipping duplicate tests")
        return None
    
    from PIL import Image
    import tempfile
    import os
    
    # Create temp directory for test images
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create original image
        original = Image.new('RGB', (200, 200), color='blue')
        original_path = temp_dir / "original.png"
        original.save(original_path)
        
        # Create exact duplicate (same content)
        exact_dup_path = temp_dir / "exact_duplicate.png"
        original.save(exact_dup_path)
        
        # Create perceptual duplicate (slightly different)
        perceptual = original.copy()
        perceptual.putpixel((0, 0), (0, 0, 254))  # Tiny change
        perceptual_path = temp_dir / "perceptual_duplicate.png"
        perceptual.save(perceptual_path)
        
        # Create different image
        different = Image.new('RGB', (200, 200), color='green')
        different_path = temp_dir / "different.png"
        different.save(different_path)
        
        # Build hash registry
        registry = HashRegistry()
        registry.add(original_path)
        registry.add(exact_dup_path)
        registry.add(perceptual_path)
        registry.add(different_path)
        
        print(f"Total files in registry: {registry.total_files}")
        
        # Check exact duplicates
        exact_dups = registry.find_exact_duplicates(original_path)
        print(f"\nExact duplicates of original: {len(exact_dups)}")
        for dup in exact_dups:
            print(f"  - {dup.name}")
        
        # Check perceptual duplicates
        perceptual_dups = registry.find_perceptual_duplicates(original_path)
        print(f"\nPerceptual duplicates of original: {len(perceptual_dups)}")
        for dup in perceptual_dups:
            print(f"  - {dup.name}")
        
        # Check all duplicate groups
        groups = registry.get_all_duplicate_groups()
        print(f"\nExact duplicate groups: {len(groups[DuplicateType.EXACT])}")
        print(f"Perceptual duplicate groups: {len(groups[DuplicateType.PERCEPTUAL])}")
        
        # Verify results
        success = True
        if len(exact_dups) != 1:
            print("✗ Expected 1 exact duplicate")
            success = False
        else:
            print("✓ Exact duplicate detection working")
        
        if len(perceptual_dups) >= 1:
            print("✓ Perceptual duplicate detection working")
        else:
            print("⚠ Perceptual duplicate not detected (may be expected)")
        
        return success
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_hamming_distance():
    """Test Hamming distance calculation."""
    print("\n" + "=" * 60)
    print("HAMMING DISTANCE TEST")
    print("=" * 60)
    
    hasher = ImageHasher()
    
    # Test with known values
    # 0x00 and 0xFF have all bits different (8 bits for 2 hex chars)
    dist = hasher.hamming_distance("00", "ff")
    print(f"Distance between 00 and ff: {dist} (expected 8)")
    
    # Same hashes
    dist_same = hasher.hamming_distance("abcd", "abcd")
    print(f"Distance between identical hashes: {dist_same} (expected 0)")
    
    # One bit different
    dist_one = hasher.hamming_distance("00", "01")
    print(f"Distance with 1 bit different: {dist_one} (expected 1)")
    
    if dist == 8 and dist_same == 0 and dist_one == 1:
        print("✓ Hamming distance calculation correct")
        return True
    else:
        print("✗ Hamming distance calculation incorrect")
        return False


def test_registry_serialization():
    """Test registry export/import."""
    print("\n" + "=" * 60)
    print("REGISTRY SERIALIZATION TEST")
    print("=" * 60)
    
    # Create a registry with some data
    registry = HashRegistry()
    
    # Add a fake hash entry
    fake_path = Path("C:/test/image.jpg")
    fake_hash = ImageHash(
        file_path=fake_path,
        md5_hash="abc123",
        phash="def456",
        ahash="789ghi",
        file_size=12345
    )
    registry._hashes[fake_path] = fake_hash
    registry._md5_index["abc123"] = {fake_path}
    registry._phash_index["def456"] = {fake_path}
    
    # Export to dict
    exported = registry.to_dict()
    print(f"Exported {len(exported)} entries")
    
    # Create new registry and import
    registry2 = HashRegistry()
    registry2.from_dict(exported)
    
    # Verify
    if registry2.total_files == 1:
        imported_hash = registry2.get(fake_path)
        if imported_hash and imported_hash.md5_hash == "abc123":
            print("✓ Registry serialization working")
            return True
    
    print("✗ Registry serialization failed")
    return False


if __name__ == "__main__":
    print("IMAGE HASHER TESTS")
    print("=" * 60)
    print(f"imagehash library available: {IMAGEHASH_AVAILABLE}")
    print()
    
    results = []
    
    results.append(("MD5 Hashing", test_md5_hashing()))
    results.append(("Perceptual Hashing", test_perceptual_hashing()))
    results.append(("Hamming Distance", test_hamming_distance()))
    results.append(("Registry Serialization", test_registry_serialization()))
    results.append(("Duplicate Detection", test_duplicate_detection()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{name}: {status}")
