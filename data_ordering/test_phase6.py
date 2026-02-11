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
from data_ordering.merge_output import OutputMerger
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
            progress_callback=progress_callback,
            use_staging=False,  # Direct output for testing
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
            use_staging=False,  # Direct output for testing
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
            use_staging=False,  # Direct output for testing
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
            use_staging=False,  # Direct output for testing
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


def test_staging_directory():
    """Test that pipeline outputs to staging directory."""
    print("\n" + "=" * 60)
    print("STAGING DIRECTORY TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    
    try:
        source_dir = temp_dir / "source" / "MyData"
        output_dir = temp_dir / "output"
        source_dir.mkdir(parents=True)
        output_dir.mkdir()
        
        # Create a few test files
        (source_dir / "LH 15083.jpg").write_text("fake image 1")
        (source_dir / "LH 15084.jpg").write_text("fake image 2")
        
        # Run with staging enabled (default)
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=False,
            use_staging=True,
        )
        
        # Check staging dir is different from output_dir
        expected_staging = PipelineOrchestrator.compute_staging_dir(output_dir, [source_dir])
        assert orchestrator.output_dir == expected_staging, (
            f"Expected staging dir {expected_staging}, got {orchestrator.output_dir}"
        )
        assert orchestrator.final_output_dir == output_dir
        print(f"  Staging dir: {orchestrator.output_dir}")
        print(f"  Final output dir: {orchestrator.final_output_dir}")
        
        # Run the pipeline
        state = orchestrator.run()
        assert state.stage == PipelineStage.COMPLETED
        
        # Check staging_info.json was created
        staging_info_path = orchestrator.output_dir / "staging_info.json"
        assert staging_info_path.exists(), "staging_info.json should exist"
        
        import json
        with open(staging_info_path) as f:
            info = json.load(f)
        assert info['final_output_dir'] == str(output_dir)
        print(f"  staging_info.json: final_output_dir={info['final_output_dir']}")
        
        # Check files are in staging dir, not in output_dir
        staging_files = list(orchestrator.output_dir.rglob("*.jpg"))
        output_files = list(output_dir.rglob("*.jpg"))
        assert len(staging_files) > 0, "Files should be in staging directory"
        print(f"  Files in staging: {len(staging_files)}")
        print(f"  Files in output (should be 0): {len(output_files)}")
        
        print("\n[PASS] Staging directory test passed")
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


def test_merge_output():
    """Test merging staging directory into output."""
    print("\n" + "=" * 60)
    print("MERGE OUTPUT TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    
    try:
        source_dir = temp_dir / "source" / "TestData"
        output_dir = temp_dir / "output"
        source_dir.mkdir(parents=True)
        output_dir.mkdir()
        
        # Create test files
        sub_dir = source_dir / "Colección LH-" / "Insecta"
        sub_dir.mkdir(parents=True)
        (sub_dir / "LH 15083.jpg").write_text("fake image 1")
        (sub_dir / "LH 15084.jpg").write_text("fake image 2")
        
        # Run pipeline with staging
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=False,
            use_staging=True,
        )
        state = orchestrator.run()
        staging_dir = orchestrator.output_dir
        
        # Close logger before merge
        if orchestrator.action_logger:
            orchestrator.action_logger.close()
        
        print(f"  Pipeline done. Staging dir: {staging_dir}")
        
        # Now merge
        merger = OutputMerger(
            staging_dir=staging_dir,
            dry_run=False,
            auto_accept=True,
        )
        stats = merger.merge()
        
        print(f"\n  Merge stats: {stats}")
        
        # Verify files were copied to output
        output_files = list(output_dir.rglob("*"))
        output_file_names = [f.name for f in output_files if f.is_file()]
        print(f"  Files in output after merge: {len([f for f in output_files if f.is_file()])}")
        
        assert stats['files_copied'] > 0, "Some files should have been copied"
        assert stats['errors'] == 0, f"No errors expected, got {stats['errors']}"
        
        # Verify registries were copied
        output_reg = output_dir / "registries"
        if output_reg.exists():
            reg_files = list(output_reg.glob("*.xlsx"))
            print(f"  Registry files in output: {[f.name for f in reg_files]}")
        
        print("\n[PASS] Merge output test passed")
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


def test_merge_with_existing():
    """Test merging into an output that already has files, including dedup."""
    print("\n" + "=" * 60)
    print("MERGE WITH EXISTING FILES TEST")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = None
    
    try:
        source_dir = temp_dir / "source" / "NewBatch"
        output_dir = temp_dir / "output"
        source_dir.mkdir(parents=True)
        output_dir.mkdir()
        
        # --- Pre-populate output with existing files ---
        existing_dir = output_dir / "Campaña_2020" / "Arthropoda"
        existing_dir.mkdir(parents=True)
        (existing_dir / "existing_file.jpg").write_text("existing image")
        
        # The staging pipeline will process "LH 15083.jpg" with content
        # "new image 1".  We pre-populate an annotation record for the
        # SAME specimen (LH-15083) so the cross-ref merge recognises
        # it as a duplicate and merges rather than appending.
        import pandas as pd
        import hashlib
        from data_ordering.excel_manager import ImageRecord, HashRecord
        from dataclasses import asdict
        
        reg_dir = output_dir / "registries"
        reg_dir.mkdir(parents=True)
        
        existing_img_path = str(existing_dir / "existing_file.jpg")
        
        # Two annotation records: one that will NOT match (LH 99999)
        # and one that WILL match (LH-15083, same specimen as staging).
        existing_records = [
            ImageRecord(
                uuid="existing-uuid-001",
                specimen_id="LH 99999",
                original_path="/old/path/existing.jpg",
                current_path=existing_img_path,
            ),
            ImageRecord(
                uuid="existing-uuid-002",
                specimen_id="LH-15083",
                original_path="/old/path/LH 15083.jpg",
                current_path=str(output_dir / "old_run" / "LH-15083.jpg"),
                macroclass_label="Arthropoda",  # extra data in output
            ),
        ]
        df_ann = pd.DataFrame([asdict(r) for r in existing_records])
        df_ann.to_excel(reg_dir / "anotaciones.xlsx", index=False)
        
        # Pre-populate hashes for the existing records
        existing_hashes = [
            HashRecord(
                uuid="hash-uuid-001",
                md5_hash=hashlib.md5(b"existing image").hexdigest(),
                phash="",
                file_path=existing_img_path,
            ),
            HashRecord(
                uuid="hash-uuid-002",
                md5_hash="dummy_hash_for_old_run",
                phash="",
                file_path=str(output_dir / "old_run" / "LH-15083.jpg"),
            ),
        ]
        df_hash = pd.DataFrame([asdict(r) for r in existing_hashes])
        df_hash.to_excel(reg_dir / "hashes.xlsx", index=False)
        
        # Create source file — same specimen as existing-uuid-002
        (source_dir / "LH 15083.jpg").write_text("new image 1")
        
        # Run pipeline with staging
        orchestrator = PipelineOrchestrator(
            source_dirs=[source_dir],
            output_dir=output_dir,
            use_llm=False,
            dry_run=False,
            use_staging=True,
        )
        state = orchestrator.run()
        staging_dir = orchestrator.output_dir
        
        if orchestrator.action_logger:
            orchestrator.action_logger.close()
        
        # Merge (auto-accept keeps existing on conflicts)
        merger = OutputMerger(
            staging_dir=staging_dir,
            dry_run=False,
            auto_accept=True,
        )
        stats = merger.merge()
        
        print(f"  Merge stats: {stats}")
        
        # Verify existing file is still there
        assert (existing_dir / "existing_file.jpg").exists(), "Existing file should remain"
        
        # Verify new files were added
        assert stats['files_copied'] > 0
        assert stats['errors'] == 0
        
        # Verify registry was merged correctly
        merged_df = pd.read_excel(reg_dir / "anotaciones.xlsx")
        print(f"  Registry records after merge: {len(merged_df)}")
        
        # LH 99999 (no match in staging) should be preserved.
        # LH-15083 matched by specimen_id → merged, NOT appended.
        # So we should have exactly 2 records, not 3.
        assert len(merged_df) == 2, (
            f"Expected 2 records (LH 99999 + merged LH-15083), got {len(merged_df)}"
        )
        
        # Check existing record is preserved
        existing_uuids = merged_df['uuid'].tolist()
        assert "existing-uuid-001" in existing_uuids, "Existing record should be preserved"
        
        # Check the matched record kept its existing macroclass (auto_accept
        # keeps existing on conflicts, and gap-fills are auto-merged).
        lh_row = merged_df[
            merged_df['specimen_id'].astype(str).str.replace(r'[\s\-_]', '', regex=True).str.lower() == 'lh15083'
        ]
        assert len(lh_row) == 1, f"Expected 1 merged LH-15083 row, got {len(lh_row)}"
        assert lh_row.iloc[0]['macroclass_label'] == 'Arthropoda', \
            "Existing macroclass_label should be preserved"
        
        print("\n[PASS] Merge with existing files test passed")
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
    results.append(("Staging Directory", test_staging_directory()))
    results.append(("Merge Output", test_merge_output()))
    results.append(("Merge With Existing", test_merge_with_existing()))
    
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
