"""
Command-line interface for Data Ordering Checker.
"""
import argparse
import sys
from pathlib import Path

from .checker import DataOrderingChecker
from .metrics import MetricsCollector


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Data Ordering Checker - Verify and analyze data_ordering output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display full report in console
  python -m data_ordering_checker.cli --output ./organized_data

  # Export report to JSON
  python -m data_ordering_checker.cli --output ./organized_data --json report.json

  # Export metrics to CSV
  python -m data_ordering_checker.cli --output ./organized_data --csv ./reports

  # Show all available metrics
  python -m data_ordering_checker.cli --output ./organized_data --all-metrics
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=False,
        help='Path to data_ordering output directory (the root folder with collections)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        help='Export detailed report to JSON file'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Export metrics to CSV files in specified directory'
    )
    
    parser.add_argument(
        '--all-metrics',
        action='store_true',
        help='Display all distribution metrics'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (useful with --json or --csv)'
    )

    parser.add_argument(
        '--merge-sources',
        nargs='+',
        help='One or more source output directories to compare before merging'
    )

    parser.add_argument(
        '--merged-dir',
        type=str,
        help='Path to the merged output directory to validate'
    )
    
    args = parser.parse_args()
    
    try:
        # If merge mode requested
        if args.merge_sources or args.merged_dir:
            if not args.merged_dir:
                print("Error: --merged-dir is required when using --merge-sources", file=sys.stderr)
                return 2

            merged = Path(args.merged_dir)
            sources = args.merge_sources or []

            # Use any valid checker instance (we just need the class methods)
            base_checker = DataOrderingChecker(sources[0] if sources else merged)
            result = base_checker.check_merge_sources(sources, merged)

            # Print a concise merge report
            print('\n' + '='*70)
            print('MERGE VALIDATION REPORT')
            print('='*70)
            print(f"  Sources analyzed: {len(result.get('sources', {}))}")
            print(f"  Source unique hashes (union): {result.get('source_unique_hashes_count')}")
            print(f"  Merged hashes: {result.get('merged_hashes_count')}")
            print(f"  Merged duplicate registry entries: {result.get('merged_duplicate_count')}")

            # Print comparison table header
            print('\n  Source comparison:')
            print('  ' + '-'*86)
            hdr = f"  {'Source':30} | {'Main':>6} | {'Duplicates':>9} | {'Text':>6} | {'Other':>6} | {'TotalReg':>8} | {'Hashes':>6}"
            print(hdr)
            print('  ' + '-'*86)

            # Print each source row
            for src, info in result.get('sources', {}).items():
                main_c = info.get('main_registry_count', 0)
                dup_c = info.get('duplicate_registry_count', 0)
                txt_c = info.get('text_registry_count', 0)
                oth_c = info.get('other_registry_count', 0)
                total_reg = main_c + dup_c + txt_c + oth_c
                hashes_c = info.get('hashes_count') if info.get('hashes_count') is not None else ''
                print(f"  {src:30.30} | {main_c:6d} | {dup_c:9d} | {txt_c:6d} | {oth_c:6d} | {total_reg:8d} | {str(hashes_c):>6}")

            print('  ' + '-'*86)

            # Print merged summary row
            ms = result.get('merged_summary', {})
            mm = ms.get('main_registry_count', 0)
            md = ms.get('duplicate_registry_count', 0)
            mt = ms.get('text_registry_count', 0)
            mo = ms.get('other_registry_count', 0)
            mtotal = mm + md + mt + mo
            mh = ms.get('hashes_count') if ms.get('hashes_count') is not None else ''
            print(f"  {'MERGED':30} | {mm:6d} | {md:9d} | {mt:6d} | {mo:6d} | {mtotal:8d} | {str(mh):>6}")
            print('  ' + '-'*86)

            # Print totals comparison (sources sum vs merged)
            sources_totals = result.get('sources_totals', {})
            comparison = result.get('merge_totals_comparison', {})
            if sources_totals and comparison:
                print('\n  Totals comparison:')
                print(f"    Sources (main+duplicates): {sources_totals.get('sum_main_plus_duplicates', 0)}")
                print(f"    Merged  (main+duplicates): {comparison.get('merged_main_plus_duplicates', 0)}")
                eq = comparison.get('equal_totals')
                eq_s = 'YES' if eq else 'NO'
                print(f"    Totals equal?: {eq_s}")
                collisions = comparison.get('collisions_main_minus_sources_main')
                print(f"    Collisions ( sum(sources main)): {collisions:+d}")

            missing = result.get('missing_in_merged', [])
            new = result.get('new_in_merged', [])
            print(f"\n  Missing in merged: {len(missing)}")
            if len(missing) > 0:
                for h in missing[:10]:
                    print(f"    - {h}")
                if len(missing) > 10:
                    print(f"    ... and {len(missing)-10} more")

            print(f"  New in merged: {len(new)}")
            if len(new) > 0:
                for h in new[:10]:
                    print(f"    - {h}")
                if len(new) > 10:
                    print(f"    ... and {len(new)-10} more")

            # Internal merged coherence
            if 'merged_internal' in result:
                mi = result['merged_internal']
                print('\n  Merged internal coherence:')
                print(f"    Files (structure): {mi['file_structure']['total']}")
                print(f"    Files (registries): {mi['registries']['total']}")
                print(f"    Non-dup diff: {mi['discrepancies']['non_duplicate_images_diff']}")
                print(f"    Dup diff:     {mi['discrepancies']['duplicate_images_diff']}")

            print('\n' + '='*70 + '\n')

            # Export merge result if requested
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Merge report exported to: {output_path}")

            return 0

        # Otherwise, normal single-directory check
        if not args.output:
            print('Error: --output is required for single-directory checks', file=sys.stderr)
            return 2

        # Initialize checker
        checker = DataOrderingChecker(args.output)

        # Print report to console (unless quiet)
        if not args.quiet:
            checker.print_report()

            # Show additional metrics if requested
            if args.all_metrics:
                metrics_collector = MetricsCollector(checker.output_dir)
                metrics_collector.print_metrics_summary()

        # Export to JSON if requested
        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            checker.export_report_to_json(output_path)

        # Export to CSV if requested
        if args.csv:
            output_dir = Path(args.csv)
            checker.export_report_to_csv(output_dir)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
