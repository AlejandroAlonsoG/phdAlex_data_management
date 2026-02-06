"""
Tests for the main_orchestrator module.
Tests the full pipeline with mock data.
"""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ordering.main_orchestrator import (
    PipelineOrchestrator, PipelineState, PipelineStage,
    ProcessedFile, run_pipeline
)
from data_ordering.file_utils import FileType


def create_test_structure(base_dir: Path) -> dict:
    """Create a test directory structure with sample files."""
    
    # Las Hoyas - Insecta
    insecta_dir = base_dir / "Colección LH-" / "3. Artrópodos" / "Insecta" / "Coleoptera"
    insecta_dir.mkdir(parents=True)
    
    # Create sample "images" (just text files for testing)
    (insecta_dir / "LH 15083.jpg").write_text("fake image 1")
    (insecta_dir / "LH 15084.jpg").write_text("fake image 2")
    (insecta_dir / "LH 15085 a.jpg").write_text("fake image 3")
    
    # Las Hoyas - Decapoda
    decapoda_dir = base_dir / "Colección LH-" / "3. Artrópodos" / "Decapoda" / "Delclosia"
    decapoda_dir.mkdir(parents=True)
    
    (decapoda_dir / "LH 8201.jpg").write_text("fake image 4")
    (decapoda_dir / "LH 8201 b.jpg").write_text("fake image 4")  # Duplicate content!
    
    # Buenache
    buenache_dir = base_dir / "Buenache" / "Crustacea"
    buenache_dir.mkdir(parents=True)
    
    (buenache_dir / "K-Bue 085.jpg").write_text("fake image 5")
    (buenache_dir / "K-Bue 086.jpg").write_text("fake image 6")
    
    # Unknown/Camera files
    camera_dir = base_dir / "Camera"
    camera_dir.mkdir(parents=True)
    
    (camera_dir / "DSC00001.jpg").write_text("camera image 1")
    (camera_dir / "DSC00002.jpg").write_text("camera image 2")
    
    return {
        'insecta_dir': insecta_dir,
        'decapoda_dir': decapoda_dir,
        'buenache_dir': buenache_dir,
        'camera_dir': camera_dir,
        'total_files': 9,
    }


def test_processed_file():
    """Test ProcessedFile dataclass."""
    print("=" * 60)
    print("PROCESSED FILE TEST")
    print("=" * 60)
    
    pf = ProcessedFile(
        original_path=Path("/test/LH 15083.jpg"),
        filename="LH 15083.jpg",
        file_size=1024,
        file_type=FileType.IMAGE,
        specimen_id="LH 15083",
        collection_code="LH",
        taxonomic_class="Insecta",
    )
    
    # Test to_dict
    d = pf.to_dict()
    print(f"to_dict keys: {list(d.keys())}")
    assert d['specimen_id'] == "LH 15083"
    assert d['collection_code'] == "LH"
    
    # Test from_dict
    pf2 = ProcessedFile.from_dict(d)
    assert pf2.specimen_id == "LH 15083"
    assert pf2.collection_code == "LH"
    
    print("\n[PASS] ProcessedFile test passed")
    return True


def test_pipeline_state():
    """Test PipelineState save/load."""
    print("\n" + "=" * 60)
    print("PIPELINE STATE TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        state = PipelineState()
        state.stage = PipelineStage.HASHING
        state.total_files = 100
        state.files_scanned = 50
        state.processed_files = {
            '/test/file1.jpg': {'filename': 'file1.jpg', 'specimen_id': 'LH-001'},
            '/test/file2.jpg': {'filename': 'file2.jpg', 'specimen_id': 'LH-002'},
        }
        
        # Save
        state_path = temp_dir / "test_state.json"
        state.save(state_path)
        print(f"Saved state to: {state_path}")
        assert state_path.exists()
        
        # Load
        loaded = PipelineState.load(state_path)
        print(f"Loaded stage: {loaded.stage}")
        print(f"Loaded total_files: {loaded.total_files}")
        print(f"Loaded processed_files count: {len(loaded.processed_files)}")
        
        assert loaded.stage == PipelineStage.HASHING
        assert loaded.total_files == 100
        assert len(loaded.processed_files) == 2
        
        print("\n[PASS] PipelineState test passed")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_pipeline_dry_run():
    """Test pipeline in dry-run mode with mock data."""
    print("\n" + "=" * 60)
    print("PIPELINE DRY RUN TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    
    try:
        # Create test structure
        source_dir = temp_dir / "source"
        output_dir = temp_dir / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        
        test_info = create_test_structure(source_dir)
        print(f"Created test structure with {test_info['total_files']} files")
        
        # Track progress
        progress_log = []
        def progress_callback(stage, current, total):
            if current == total or current == 1:
                progress_log.append(f"{stage}: {current}/{total}")
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,  # Use mock client
            dry_run=True,   # Don't actually move files
            progress_callback=progress_callback
        )
        
        state = orchestrator.run()
        
        print(f"\nPipeline completed!")
        print(f"Stage: {state.stage.value}")
        print(f"Total files: {state.total_files}")
        print(f"Files scanned: {state.files_scanned}")
        print(f"Directories analyzed: {state.directories_analyzed}")
        
        # Get summary
        summary = orchestrator.get_summary()
        print(f"\nSummary:")
        print(f"  By collection: {summary['by_collection']}")
        print(f"  By class: {summary['by_taxonomic_class']}")
        print(f"  Duplicates: {summary['duplicates']}")
        
        print(f"\nProgress log:")
        for entry in progress_log:
            print(f"  {entry}")
        
        # Verify results
        assert state.stage == PipelineStage.COMPLETED
        assert state.total_files == test_info['total_files']
        assert state.files_scanned == test_info['total_files']
        
        # Check that no files were actually moved (dry run)
        for pf_data in state.processed_files.values():
            assert not pf_data.get('moved', False), "Files should not be moved in dry run"
        
        print("\n[PASS] Pipeline dry run test passed")
        return True
        
    finally:
        # Close logger to release file handles
        if orchestrator and orchestrator.action_logger:
            orchestrator.action_logger.close()
        
        # Give Windows time to release handles
        import time
        time.sleep(0.1)
        
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print("  [WARN] Could not clean up temp dir (file in use)")


def test_pipeline_with_files():
    """Test pipeline that actually moves files."""
    print("\n" + "=" * 60)
    print("PIPELINE WITH FILE MOVE TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    
    try:
        # Create test structure
        source_dir = temp_dir / "source"
        output_dir = temp_dir / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        
        test_info = create_test_structure(source_dir)
        
        # Run pipeline (not dry run)
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=False,
        )
        state = orchestrator.run()
        
        print(f"Pipeline stage: {state.stage.value}")
        
        # Check output directory structure
        print(f"\nOutput structure:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(output_dir)
                print(f"  {rel_path}")
        
        # Verify files were moved
        moved_count = sum(1 for pf in state.processed_files.values() if pf.get('moved'))
        print(f"\nFiles moved: {moved_count}")
        
        # Check registries were created
        registries_dir = output_dir / "registries"
        if registries_dir.exists():
            print(f"Registries created in: {registries_dir}")
        
        assert state.stage == PipelineStage.COMPLETED
        assert moved_count > 0
        
        print("\n[PASS] Pipeline with file move test passed")
        return True
        
    finally:
        if orchestrator and orchestrator.action_logger:
            orchestrator.action_logger.close()
        
        import time
        time.sleep(0.1)
        
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print("  [WARN] Could not clean up temp dir")


def test_pipeline_resume():
    """Test pipeline resume capability."""
    print("\n" + "=" * 60)
    print("PIPELINE RESUME TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    orchestrator2 = None
    
    try:
        # Create test structure
        source_dir = temp_dir / "source"
        output_dir = temp_dir / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        
        create_test_structure(source_dir)
        
        # Run pipeline partially
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=True,
        )
        
        # Run just scanning stage
        orchestrator._run_scanning()
        orchestrator._save_state()
        
        print(f"After scanning - Stage: {orchestrator.state.stage.value}")
        print(f"Files scanned: {orchestrator.state.files_scanned}")
        
        # Close first orchestrator's logger
        if orchestrator.action_logger:
            orchestrator.action_logger.close()
        
        # Create new orchestrator and resume
        orchestrator2 = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=True,
        )
        
        state = orchestrator2.run(resume=True)
        
        print(f"\nAfter resume - Stage: {state.stage.value}")
        
        assert state.stage == PipelineStage.COMPLETED
        
        print("\n[PASS] Pipeline resume test passed")
        return True
        
    finally:
        if orchestrator and orchestrator.action_logger:
            orchestrator.action_logger.close()
        if orchestrator2 and orchestrator2.action_logger:
            orchestrator2.action_logger.close()
        
        import time
        time.sleep(0.1)
        
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print("  [WARN] Could not clean up temp dir")


if __name__ == "__main__":
    print("MAIN ORCHESTRATOR TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("ProcessedFile", test_processed_file()))
    results.append(("PipelineState", test_pipeline_state()))
    results.append(("Pipeline Dry Run", test_pipeline_dry_run()))
    results.append(("Pipeline With Files", test_pipeline_with_files()))
    results.append(("Pipeline Resume", test_pipeline_resume()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        if result is True:
            status = "[PASS]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
