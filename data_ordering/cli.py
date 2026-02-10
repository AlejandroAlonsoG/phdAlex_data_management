#!/usr/bin/env python
"""
Data Ordering Tool - Command Line Interface
============================================

CLI for running the data ordering pipeline on fossil image directories.

Usage:
    python cli.py --source /path/to/images --output /path/to/output [options]
    
    # Dry run (no file operations)
    python cli.py --source ./MUPA --output ./output --dry-run
    
    # With LLM analysis (requires GEMINI_API_KEY in .env)
    python cli.py --source ./MUPA --output ./output --use-llm
    
    # Resume interrupted processing
    python cli.py --output ./output --resume
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .main_orchestrator import (
    PipelineOrchestrator, PipelineStage, run_pipeline
)
from .config import config


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Organize fossil image datasets for deep learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (dry run)
  python cli.py --source ./MUPA --output ./output --dry-run
  
  # Process with LLM analysis
  python cli.py --source ./MUPA ./YCLH --output ./organized --use-llm
  
  # Resume interrupted processing
  python cli.py --output ./organized --resume
  
  # Process with custom settings
  python cli.py --source ./images --output ./output --use-llm --phash-threshold 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for organized files and registries'
    )
    
    # Source directories (optional if resuming)
    parser.add_argument(
        '--source', '-s',
        type=Path,
        nargs='+',
        help='Source directories to scan (can specify multiple)'
    )
    
    # Mode flags
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate processing without moving files'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted processing from saved state'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use Gemini LLM for directory analysis (requires GEMINI_API_KEY)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM analysis (use regex patterns only)'
    )
    
    # Interactive mode
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode: pause for manual confirmation on ambiguous cases'
    )
    
    parser.add_argument(
        '--deferred',
        action='store_true',
        help='Deferred mode: move ambiguous files to review folders instead of pausing'
    )
    
    parser.add_argument(
        '--step-by-step', '-S',
        action='store_true',
        dest='step_by_step',
        help='Step-by-step mode: pause at EVERY stage for manual verification (most control)'
    )
    
    # Processing options
    parser.add_argument(
        '--phash-threshold',
        type=int,
        default=8,
        help='Perceptual hash threshold for duplicate detection (default: 8)'
    )
    
    parser.add_argument(
        '--skip-scan',
        action='store_true',
        help='Skip scanning (use existing state)'
    )
    
    parser.add_argument(
        '--skip-hashing',
        action='store_true',
        help='Skip hash computation (faster but no duplicate detection)'
    )
    
    parser.add_argument(
        '--no-staging',
        action='store_true',
        help='Write directly to output directory instead of a staging directory'
    )
    
    # Output control
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Either source or resume must be specified
    if not args.resume and not args.source:
        print("Error: Must specify --source directories or --resume")
        return False
    
    # Check source directories exist
    if args.source:
        for src in args.source:
            if not src.exists():
                print(f"Error: Source directory not found: {src}")
                return False
            if not src.is_dir():
                print(f"Error: Not a directory: {src}")
                return False
    
    # Check resume state exists
    if args.resume:
        state_file = args.output / "pipeline_state.json"
        if not state_file.exists():
            print(f"Error: No saved state found at {state_file}")
            print("Cannot resume without existing state. Start a new run with --source")
            return False
    
    # Conflicting options
    if args.use_llm and args.no_llm:
        print("Error: Cannot specify both --use-llm and --no-llm")
        return False
    
    if args.interactive and args.deferred:
        print("Error: Cannot specify both --interactive and --deferred")
        return False
    
    return True


def print_summary(orchestrator: PipelineOrchestrator):
    """Print pipeline summary."""
    summary = orchestrator.get_summary()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    print(f"\nTotal files processed: {summary['total_files']}")
    print(f"Files organized: {summary['files_organized']}")
    print(f"Directories analyzed: {summary['directories_analyzed']}")
    print(f"Duplicates found: {summary['duplicates']}")
    if summary['errors']:
        print(f"Errors: {summary['errors']}")
    
    print("\nBy collection:")
    for collection, count in summary.get('by_collection', {}).items():
        print(f"  {collection}: {count}")
    
    print("\nBy class:")
    for taxon, count in summary.get('by_taxonomic_class', {}).items():
        name = taxon if taxon else "Unclassified"
        print(f"  {name}: {count}")
    
    print(f"\nOutput directory: {orchestrator.output_base}")
    print(f"Registries: {orchestrator.output_base / 'registries'}")
    print(f"Logs: {orchestrator.output_base / 'logs'}")


def run_cli(args: argparse.Namespace) -> int:
    """Run the pipeline with given arguments."""
    
    # Determine LLM usage
    use_llm = args.use_llm and not args.no_llm
    
    # Determine interaction mode (priority: step_by_step > interactive > deferred > auto)
    if getattr(args, 'step_by_step', False):
        interaction_mode = 'step_by_step'
    elif args.interactive:
        interaction_mode = 'interactive'
    elif args.deferred:
        interaction_mode = 'deferred'
    else:
        interaction_mode = 'interactive'
    
    # Create progress callback
    def progress_callback(stage: str, current: int, total: int):
        if not args.quiet:
            pct = (current / total * 100) if total > 0 else 0
            print(f"\r  {stage}: {current}/{total} ({pct:.1f}%)", end="", flush=True)
            if current == total:
                print()
    
    try:
        # Initialize orchestrator
        if args.verbose:
            print("Initializing pipeline...")
        
        orchestrator = PipelineOrchestrator(
            source_dirs=[p.resolve() for p in args.source] if args.source else [],
            output_base=args.output.resolve(),
            dry_run=args.dry_run,
            use_llm=use_llm,
            phash_threshold=args.phash_threshold,
            progress_callback=None if args.quiet else progress_callback,
            interaction_mode=interaction_mode,
            use_staging=not getattr(args, 'no_staging', False),
        )
        
        # Resume or run fresh
        if args.resume:
            print(f"Resuming from saved state...")
            # Load state to show info before running
            from .main_orchestrator import PipelineState
            state_file = args.output.resolve() / "pipeline_state.json"
            if state_file.exists():
                saved = PipelineState.load(state_file)
                print(f"  Stage: {saved.stage.value}")
                print(f"  Files scanned: {saved.files_scanned}")
        
        # Run pipeline
        if not args.quiet:
            print(f"\nSource directories: {[str(p) for p in (args.source or [])]}")
            print(f"Output directory: {args.output.resolve()}")
            if not getattr(args, 'no_staging', False) and args.source:
                print(f"Staging directory: {orchestrator.output_dir}")
                print(f"  (Results will be written here for validation before merging)")
            print(f"Dry run: {args.dry_run}")
            print(f"Use LLM: {use_llm}")
            print(f"Interaction mode: {interaction_mode}")
            print()
        
        orchestrator.run(resume=args.resume)
        
        # Print summary
        print_summary(orchestrator)
        
        # Show merge instructions if staging was used
        if not getattr(args, 'no_staging', False) and orchestrator.output_dir != orchestrator.final_output_dir:
            print(f"\n{'='*60}")
            print("NEXT STEP: VALIDATE AND MERGE")
            print(f"{'='*60}")
            print(f"Results are in staging directory:")
            print(f"  {orchestrator.output_dir}")
            print(f"\nPlease review the results, then merge into final output:")
            print(f"  python -m data_ordering.merge_output --staging {orchestrator.output_dir}")
            print(f"  (Final output: {orchestrator.final_output_dir})")
            print(f"\nOr merge with explicit target:")
            print(f"  python -m data_ordering.merge_output --staging {orchestrator.output_dir} --output {orchestrator.final_output_dir}")
        
        # Cleanup
        orchestrator.action_logger.close()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved.")
        print("Use --resume to continue from where you left off.")
        return 130
        
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
