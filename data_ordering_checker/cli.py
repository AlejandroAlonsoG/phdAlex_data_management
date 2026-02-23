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

            if not args.quiet:
                DataOrderingChecker.print_merge_report(result)

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
