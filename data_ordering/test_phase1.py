"""
Quick tests for Phase 1: Core Infrastructure
Run this to verify everything is working.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from data_ordering.config import Config, config
        from data_ordering.logger_module import DataOrderingLogger, LogAction
        from data_ordering.excel_manager import ExcelManager, ImageRecord
        from data_ordering.file_utils import FileClassifier, DirectoryWalker, FileOperations
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    from data_ordering.config import Config
    
    cfg = Config()
    print(f"  Output base: {cfg.output_base_dir}")
    print(f"  LLM RPM: {cfg.llm_requests_per_minute}")
    print(f"  Image extensions: {len(cfg.image_extensions)} defined")
    print("✓ Configuration works")
    return True

def test_logger():
    """Test logger module."""
    print("\nTesting logger...")
    from data_ordering.logger_module import test_logger
    test_logger()
    return True

def test_excel_manager():
    """Test Excel manager."""
    print("\nTesting Excel manager...")
    from data_ordering.excel_manager import test_excel_manager
    test_excel_manager()
    return True

def test_file_utils():
    """Test file utilities."""
    print("\nTesting file utilities...")
    from data_ordering.file_utils import test_file_utils
    test_file_utils()
    return True

def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 50)
    print("Phase 1: Core Infrastructure Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_logger,
        test_excel_manager,
        test_file_utils,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
