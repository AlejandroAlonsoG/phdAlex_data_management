#!/usr/bin/env python3
"""
Generate visual statistics from a merge-pipeline output directory.

Usage:
    cd data_stats
    pip install -r requirements.txt
    python run.py <output_directory>

    # Skip PNG export (HTML only)
    python run.py <output_directory> --no-png

    # Open report in browser after generation
    python run.py <output_directory> --open
"""
import sys
import argparse
import logging
from pathlib import Path

# Ensure this directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from stats_generator import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual dataset statistics from merge-pipeline output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py D:/output_las_hoyas\n"
            "  python run.py D:/output_las_hoyas --open\n"
            "  python run.py D:/output_las_hoyas --no-png\n"
        ),
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the merge-pipeline output directory")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip exporting individual PNG figures")
    parser.add_argument("--open", action="store_true",
                        help="Open the HTML report in the default browser")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.output_dir.is_dir():
        sys.exit(f"ERROR: '{args.output_dir}' is not a directory.")

    print(f"📊  Generating statistics for: {args.output_dir}\n")
    report_path = generate_report(args.output_dir, save_png=not args.no_png)

    if args.open:
        import webbrowser
        webbrowser.open(report_path.as_uri())


if __name__ == "__main__":
    main()
