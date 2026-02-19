"""
Main Data Ordering Checker - Analyzes organized data structure and validates integrity.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict

from .metrics import MetricsCollector


@dataclass
class FileCountMetrics:
    """Metrics for file counts in the organized structure."""
    non_duplicate_images: int = 0
    duplicate_images: int = 0
    text_files: int = 0
    other_files: int = 0
    
    def total(self) -> int:
        """Get total file count."""
        return (self.non_duplicate_images + self.duplicate_images + 
                self.text_files + self.other_files)
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            'non_duplicate_images': self.non_duplicate_images,
            'duplicate_images': self.duplicate_images,
            'text_files': self.text_files,
            'other_files': self.other_files,
            'total': self.total()
        }


@dataclass
class TaxonomyMetrics:
    """Metrics for taxonomic information found in registries."""
    macroclasses: set = field(default_factory=set)
    classes: set = field(default_factory=set)
    determinations: set = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary with sorted lists."""
        def _clean_and_str(items):
            out = []
            for x in items:
                # Exclude obvious missing values
                try:
                    if pd.isna(x):
                        continue
                except Exception:
                    pass
                if x is None:
                    continue
                s = str(x).strip()
                if s == '' or s.lower() in {'nan', 'none'}:
                    continue
                out.append(s)
            return sorted(set(out))

        return {
            'macroclasses': _clean_and_str(self.macroclasses),
            'classes': _clean_and_str(self.classes),
            'determinations': _clean_and_str(self.determinations)
        }


class DataOrderingChecker:
    """
    Verifies the output of data_ordering pipeline and provides detailed metrics.
    
    Analyzes:
    1. File system structure (file counts and types)
    2. Excel registries (metadata validation and taxonomy)
    3. Duplicate tracking
    4. Metadata consistency
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the checker with an output directory.
        
        Args:
            output_dir: Root directory of data_ordering output
        """
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
        
        self.registries_dir = self.output_dir / "registries"
        if not self.registries_dir.exists():
            raise ValueError(f"Registries directory not found: {self.registries_dir}")
        
        # Load main registries
        self.main_registry_path = self.registries_dir / "anotaciones.xlsx"
        self.text_files_path = self.registries_dir / "archivos_texto.xlsx"
        self.other_files_path = self.registries_dir / "archivos_otros.xlsx"
        self.hash_registry_path = self.registries_dir / "hashes.xlsx"
        
        # Load duplicate registry (in Duplicados folder)
        self.duplicados_dir = self.output_dir / "Duplicados"
        self.duplicate_registry_path = self.duplicados_dir / "duplicados_registro.xlsx" if self.duplicados_dir.exists() else None
        
        # Load dataframes
        self.main_df = self._load_registry(self.main_registry_path)
        self.text_df = self._load_registry(self.text_files_path)
        self.other_df = self._load_registry(self.other_files_path)
        self.hash_df = self._load_registry(self.hash_registry_path)
        self.duplicate_df = self._load_registry(self.duplicate_registry_path) if self.duplicate_registry_path else None
        
        self.metrics_collector = MetricsCollector(self.output_dir)
    
    @staticmethod
    def _load_registry(path: Path) -> Optional[pd.DataFrame]:
        """Load an Excel registry file."""
        if not path.exists():
            return None
        try:
            return pd.read_excel(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None
    
    def count_files_in_structure(self) -> FileCountMetrics:
        """
        Count files by type by traversing the organized directory structure.
        
        Expected structure:
        - Las_Hoyas/ (organized non-duplicate images)
        - Otras_Colecciones/ (organized non-duplicate images by collection)
        - Duplicados/ (duplicate images)
        - Archivos_Texto/ (text files)
        - Otros_Archivos/ (other files)
        
        Returns:
            FileCountMetrics with file counts
        """
        metrics = FileCountMetrics()
        
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        text_extensions = {
            '.txt', '.csv', '.md', '.json', '.pdf', '.doc', '.docx',
            '.rtf', '.odt', '.xml', '.yaml', '.yml', '.tsv'
        }
        
        # Check for Duplicados folder (Spanish name used by data_ordering)
        duplicados_dir = self.output_dir / "Duplicados"
        if duplicados_dir.exists():
            for file_path in duplicados_dir.rglob('*'):
                if file_path.is_file():
                    # Skip Excel registry files
                    if file_path.name == "duplicados_registro.xlsx":
                        continue
                    ext = file_path.suffix.lower()
                    if ext in image_extensions:
                        metrics.duplicate_images += 1
                    elif ext in text_extensions:
                        metrics.text_files += 1
                    else:
                        metrics.other_files += 1
        
        # Count non-duplicate images from any top-level collection folders.
        # The pipeline may organize files under various top-level names such as:
        # - Las_Hoyas
        # - Otras_Colecciones
        # - Campaña_2020, Campaña_2019, Sin_Campaña, etc.
        # We treat any top-level directory that is NOT a special folder as a collection root.
        special_names = {
            'Duplicados', 'registries', 'logs', 'Archivos_Texto',
            'Otros_Archivos', 'Revision_Manual', '__pycache__'
        }

        for child in self.output_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in special_names:
                continue
            # Count images recursively in this collection root
            for file_path in child.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in image_extensions:
                        metrics.non_duplicate_images += 1
        
        # Check text files folder
        text_dir = self.output_dir / "Archivos_Texto"
        if text_dir.exists():
            for file_path in text_dir.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in text_extensions:
                        metrics.text_files += 1
                    else:
                        metrics.other_files += 1
        
        # Check other files folder
        other_dir = self.output_dir / "Otros_Archivos"
        if other_dir.exists():
            for file_path in other_dir.rglob('*'):
                if file_path.is_file():
                    metrics.other_files += 1
        
        # Also check Revision_Manual folder for images that need review
        revision_dir = self.output_dir / "Revision_Manual"
        if revision_dir.exists():
            for file_path in revision_dir.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in image_extensions:
                        # These are non-duplicate images flagged for review
                        metrics.non_duplicate_images += 1
        
        return metrics
    
    def count_files_in_registries(self) -> FileCountMetrics:
        """
        Count files by analyzing Excel registries.
        
        The registry structure is:
        - registries/anotaciones.xlsx: Non-duplicate images only
        - registries/archivos_texto.xlsx: Text files
        - registries/archivos_otros.xlsx: Other files
        - Duplicados/duplicados_registro.xlsx: Duplicate images (separate)
        
        Returns:
            FileCountMetrics based on registry data
        """
        metrics = FileCountMetrics()
        
        # Count non-duplicate images from main registry
        if self.main_df is not None:
            metrics.non_duplicate_images = len(self.main_df)
        
        # Count duplicate images from separate duplicates registry
        if self.duplicate_df is not None:
            metrics.duplicate_images = len(self.duplicate_df)
        
        # Count text files
        if self.text_df is not None:
            metrics.text_files = len(self.text_df)
        
        # Count other files
        if self.other_df is not None:
            metrics.other_files = len(self.other_df)
        
        return metrics
    
    def get_taxonomy_info(self) -> TaxonomyMetrics:
        """
        Extract all unique macroclasses, classes, and determinations from ALL registries.
        
        Reads from:
        - anotaciones.xlsx (non-duplicate images)
        - duplicados_registro.xlsx (duplicate images)
        
        Returns:
            TaxonomyMetrics with taxonomy information
        """
        metrics = TaxonomyMetrics()
        
        # Read from main registry (non-duplicates)
        if self.main_df is not None:
            # Get macroclasses
            if 'macroclass_label' in self.main_df.columns:
                metrics.macroclasses.update(self.main_df['macroclass_label'].unique())
            
            # Get classes
            if 'class_label' in self.main_df.columns:
                metrics.classes.update(self.main_df['class_label'].unique())
            
            # Get determinations (genera)
            if 'genera_label' in self.main_df.columns:
                metrics.determinations.update(self.main_df['genera_label'].unique())
        
        # Read from duplicate registry (if exists)
        if self.duplicate_df is not None:
            # Get macroclasses
            if 'macroclass_label' in self.duplicate_df.columns:
                metrics.macroclasses.update(self.duplicate_df['macroclass_label'].unique())
            
            # Get classes
            if 'class_label' in self.duplicate_df.columns:
                metrics.classes.update(self.duplicate_df['class_label'].unique())
            
            # Get determinations (genera)
            if 'genera_label' in self.duplicate_df.columns:
                metrics.determinations.update(self.duplicate_df['genera_label'].unique())
        
        return metrics
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Generate a detailed report comparing file system and registry metrics.
        
        Returns:
            Dictionary with comprehensive metrics and validation info
        """
        file_structure_metrics = self.count_files_in_structure()
        registry_metrics = self.count_files_in_registries()
        taxonomy_metrics = self.get_taxonomy_info()
        
        # Check for discrepancies
        discrepancies = {
            'non_duplicate_images_diff': (file_structure_metrics.non_duplicate_images - 
                                         registry_metrics.non_duplicate_images),
            'duplicate_images_diff': (file_structure_metrics.duplicate_images - 
                                     registry_metrics.duplicate_images),
            'text_files_diff': (file_structure_metrics.text_files - 
                               registry_metrics.text_files),
            'other_files_diff': (file_structure_metrics.other_files - 
                                registry_metrics.other_files),
        }
        
        tax_dict = taxonomy_metrics.to_dict()

        return {
            'file_structure_counts': file_structure_metrics.to_dict(),
            'registry_counts': registry_metrics.to_dict(),
            'discrepancies': discrepancies,
            'taxonomy': tax_dict,
            'summary': {
                'total_files_structure': file_structure_metrics.total(),
                'total_files_registry': registry_metrics.total(),
                'macroclasses_count': len(tax_dict['macroclasses']),
                'classes_count': len(tax_dict['classes']),
                'determinations_count': len(tax_dict['determinations']),
            }
        }
    
    def print_report(self):
        """Print a formatted report to console."""
        report = self.get_detailed_report()
        
        print("\n" + "="*70)
        print("DATA ORDERING CHECKER - VERIFICATION REPORT")
        print("="*70)
        
        # File structure metrics
        print("\n📁 FILE SYSTEM STRUCTURE COUNTS:")
        print("-" * 50)
        fs = report['file_structure_counts']
        print(f"  Non-duplicate images: {fs['non_duplicate_images']}")
        print(f"  Duplicate images:     {fs['duplicate_images']}")
        print(f"  Text files:           {fs['text_files']}")
        print(f"  Other files:          {fs['other_files']}")
        print(f"  {'─' * 45}")
        print(f"  TOTAL:                {fs['total']}")
        
        # Registry metrics
        print("\n📊 EXCEL REGISTRIES COUNTS:")
        print("-" * 50)
        reg = report['registry_counts']
        print(f"  Non-duplicate images: {reg['non_duplicate_images']}")
        print(f"  Duplicate images:     {reg['duplicate_images']}")
        print(f"  Text files:           {reg['text_files']}")
        print(f"  Other files:          {reg['other_files']}")
        print(f"  {'─' * 45}")
        print(f"  TOTAL:                {reg['total']}")
        
        # Discrepancies
        print("\n⚠️  DISCREPANCIES (Structure - Registry):")
        print("-" * 50)
        disc = report['discrepancies']
        all_zero = all(v == 0 for v in disc.values())
        if all_zero:
            print("  ✓ No discrepancies found!")
        else:
            for key, value in disc.items():
                status = "✓" if value == 0 else "✗"
                print(f"  {status} {key}: {value:+d}")
        
        # Taxonomy
        print("\n🔬 TAXONOMY INFORMATION:")
        print("-" * 50)
        tax = report['taxonomy']
        print(f"\n  Macroclasses ({len(tax['macroclasses'])}):")
        for mc in tax['macroclasses'][:10]:  # Show first 10
            print(f"    • {mc}")
        if len(tax['macroclasses']) > 10:
            print(f"    ... and {len(tax['macroclasses']) - 10} more")
        
        print(f"\n  Classes ({len(tax['classes'])}):")
        for c in tax['classes'][:10]:  # Show first 10
            print(f"    • {c}")
        if len(tax['classes']) > 10:
            print(f"    ... and {len(tax['classes']) - 10} more")
        
        print(f"\n  Determinations/Genera ({len(tax['determinations'])}):")
        for d in tax['determinations'][:10]:  # Show first 10
            print(f"    • {d}")
        if len(tax['determinations']) > 10:
            print(f"    ... and {len(tax['determinations']) - 10} more")
        
        # Summary
        print("\n📈 SUMMARY:")
        print("-" * 50)
        summary = report['summary']
        print(f"  Total files (structure):  {summary['total_files_structure']}")
        print(f"  Total files (registry):   {summary['total_files_registry']}")
        print(f"  Unique macroclasses:      {summary['macroclasses_count']}")
        print(f"  Unique classes:           {summary['classes_count']}")
        print(f"  Unique determinations:    {summary['determinations_count']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_report_to_json(self, output_path: Path):
        """Export detailed report to JSON file."""
        import json
        report = self.get_detailed_report()
        
        # Convert sets to lists for JSON serialization
        report['taxonomy'] = {
            k: list(v) if isinstance(v, set) else v 
            for k, v in report['taxonomy'].items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report exported to: {output_path}")
    
    def export_report_to_csv(self, output_dir: Path):
        """Export metrics to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = self.get_detailed_report()
        
        # File counts comparison
        counts_df = pd.DataFrame({
            'File Type': ['Non-duplicate images', 'Duplicate images', 'Text files', 'Other files', 'Total'],
            'File Structure': [
                report['file_structure_counts']['non_duplicate_images'],
                report['file_structure_counts']['duplicate_images'],
                report['file_structure_counts']['text_files'],
                report['file_structure_counts']['other_files'],
                report['file_structure_counts']['total'],
            ],
            'Registry': [
                report['registry_counts']['non_duplicate_images'],
                report['registry_counts']['duplicate_images'],
                report['registry_counts']['text_files'],
                report['registry_counts']['other_files'],
                report['registry_counts']['total'],
            ],
            'Difference': [
                report['discrepancies']['non_duplicate_images_diff'],
                report['discrepancies']['duplicate_images_diff'],
                report['discrepancies']['text_files_diff'],
                report['discrepancies']['other_files_diff'],
                report['file_structure_counts']['total'] - report['registry_counts']['total'],
            ]
        })
        counts_df.to_csv(output_dir / "file_counts.csv", index=False)
        
        # Taxonomy
        max_items = max(
            len(report['taxonomy']['macroclasses']),
            len(report['taxonomy']['classes']),
            len(report['taxonomy']['determinations'])
        )
        
        taxonomy_df = pd.DataFrame({
            'Macroclass': report['taxonomy']['macroclasses'] + [''] * (max_items - len(report['taxonomy']['macroclasses'])),
            'Class': report['taxonomy']['classes'] + [''] * (max_items - len(report['taxonomy']['classes'])),
            'Determination': report['taxonomy']['determinations'] + [''] * (max_items - len(report['taxonomy']['determinations'])),
        })
        taxonomy_df.to_csv(output_dir / "taxonomy.csv", index=False)
        
        print(f"CSV reports exported to: {output_dir}")

    def check_merge_sources(self, source_dirs: List[Path], merged_dir: Path) -> Dict[str, Any]:
        """
        Compare multiple source outputs with a merged output.

        Behavior:
        - For each source, tries to read `registries/hashes.xlsx` to collect MD5 hashes.
        - For the merged directory, reads its hashes and duplicate registry if present.
        - Reports totals per-source, union of unique source hashes, merged totals,
          newly missing items, newly introduced items, and duplicate counts.

        Args:
            source_dirs: List of source output directories (Path or str)
            merged_dir: The merged output directory to validate

        Returns:
            Dictionary containing comparison metrics and lists of differing hashes.
        """
        # Normalize paths
        source_dirs = [Path(p) for p in source_dirs]
        merged_dir = Path(merged_dir)

        def _load_hashes_from(reg_root: Path) -> Optional[Set[str]]:
            all_uuids = set()

            # Load hashes from the main registry
            hashes_path = reg_root / 'registries' / 'hashes.xlsx'
            df_main = DataOrderingChecker._load_registry(hashes_path)
            if df_main is not None and 'uuid' in df_main.columns:
                all_uuids.update(df_main['uuid'].dropna().astype(str).str.strip().values)

            # Load hashes from the duplicate registry
            duplicados_path = reg_root / 'Duplicados' / 'duplicados_registro.xlsx'
            df_dups = DataOrderingChecker._load_registry(duplicados_path)
            if df_dups is not None and 'uuid' in df_dups.columns:
                all_uuids.update(df_dups['uuid'].dropna().astype(str).str.strip().values)
            
            if not all_uuids:
                return None
            return all_uuids

        source_hashes_list = []
        source_entries = {}
        for sd in source_dirs:
            hashes = _load_hashes_from(sd)
            # gather registry-based counts where available
            main_df = DataOrderingChecker._load_registry(Path(sd) / 'registries' / 'anotaciones.xlsx')
            text_df = DataOrderingChecker._load_registry(Path(sd) / 'registries' / 'archivos_texto.xlsx')
            other_df = DataOrderingChecker._load_registry(Path(sd) / 'registries' / 'archivos_otros.xlsx')
            dup_df = DataOrderingChecker._load_registry(Path(sd) / 'Duplicados' / 'duplicados_registro.xlsx')

            entry = {
                'hashes_count': len(hashes) if hashes is not None else None,
                'main_registry_count': len(main_df) if main_df is not None else 0,
                'duplicate_registry_count': len(dup_df) if dup_df is not None else 0,
                'text_registry_count': len(text_df) if text_df is not None else 0,
                'other_registry_count': len(other_df) if other_df is not None else 0,
            }

            # If no hashes found, fall back to filesystem counts as supplemental info
            if hashes is None:
                try:
                    chk = DataOrderingChecker(sd)
                    fs_counts = chk.count_files_in_structure()
                    entry.update({
                        'fs_non_duplicate_images': fs_counts.non_duplicate_images,
                        'fs_duplicate_images': fs_counts.duplicate_images,
                    })
                except Exception:
                    pass

            source_hashes_list.append(hashes if hashes is not None else set())
            source_entries[str(sd)] = entry

        # Union of source unique hashes
        union_hashes: Set[str] = set()
        for s in source_hashes_list:
            union_hashes.update(s)

        # Load merged hashes
        merged_hashes = _load_hashes_from(merged_dir)
        merged_duplicate_df = None
        duplicados_path = merged_dir / 'Duplicados' / 'duplicados_registro.xlsx'
        if duplicados_path.exists():
            merged_duplicate_df = DataOrderingChecker._load_registry(duplicados_path)

        # Prepare comparison
        result = {
            'sources': source_entries,
            'source_unique_hashes_count': len(union_hashes),
            'merged_hashes_count': len(merged_hashes) if merged_hashes is not None else None,
            'merged_duplicate_count': len(merged_duplicate_df) if merged_duplicate_df is not None else 0,
            'missing_in_merged': [],
            'new_in_merged': [],
        }

        if merged_hashes is not None:
            missing = sorted(list(union_hashes - merged_hashes))
            new = sorted(list(merged_hashes - union_hashes))
            result['missing_in_merged'] = missing
            result['new_in_merged'] = new

        # Also run internal coherence check on merged_dir (structure vs registries)
        try:
            merged_checker = DataOrderingChecker(merged_dir)
            fs_metrics = merged_checker.count_files_in_structure()
            reg_metrics = merged_checker.count_files_in_registries()
            taxonomy = merged_checker.get_taxonomy_info().to_dict()

            result['merged_internal'] = {
                'file_structure': fs_metrics.to_dict(),
                'registries': reg_metrics.to_dict(),
                'taxonomy': taxonomy,
                'discrepancies': {
                    'non_duplicate_images_diff': fs_metrics.non_duplicate_images - reg_metrics.non_duplicate_images if reg_metrics.non_duplicate_images is not None else None,
                    'duplicate_images_diff': fs_metrics.duplicate_images - reg_metrics.duplicate_images if reg_metrics.duplicate_images is not None else None,
                }
            }
        except Exception as e:
            result['merged_internal_error'] = str(e)

        # Add merged registry breakdown
        merged_main_df = DataOrderingChecker._load_registry(merged_dir / 'registries' / 'anotaciones.xlsx')
        merged_text_df = DataOrderingChecker._load_registry(merged_dir / 'registries' / 'archivos_texto.xlsx')
        merged_other_df = DataOrderingChecker._load_registry(merged_dir / 'registries' / 'archivos_otros.xlsx')
        merged_dup_df = DataOrderingChecker._load_registry(merged_dir / 'Duplicados' / 'duplicados_registro.xlsx')

        result['merged_summary'] = {
            'main_registry_count': len(merged_main_df) if merged_main_df is not None else 0,
            'duplicate_registry_count': len(merged_dup_df) if merged_dup_df is not None else 0,
            'text_registry_count': len(merged_text_df) if merged_text_df is not None else 0,
            'other_registry_count': len(merged_other_df) if merged_other_df is not None else 0,
            'hashes_count': len(merged_hashes) if merged_hashes is not None else None,
        }

        # Compute totals across sources (main + duplicates) for comparison
        try:
            sum_sources_main = sum(v.get('main_registry_count', 0) or 0 for v in source_entries.values())
            sum_sources_dup = sum(v.get('duplicate_registry_count', 0) or 0 for v in source_entries.values())
        except Exception:
            sum_sources_main = 0
            sum_sources_dup = 0

        result['sources_totals'] = {
            'sum_main_registry_count': sum_sources_main,
            'sum_duplicate_registry_count': sum_sources_dup,
            'sum_main_plus_duplicates': sum_sources_main + sum_sources_dup,
        }

        # Compare merged main+duplicates with sum of sources
        merged_main = result['merged_summary']['main_registry_count']
        merged_dup = result['merged_summary']['duplicate_registry_count']
        result['merge_totals_comparison'] = {
            'merged_main_plus_duplicates': merged_main + merged_dup,
            'sources_main_plus_duplicates': result['sources_totals']['sum_main_plus_duplicates'],
            'equal_totals': (merged_main + merged_dup) == result['sources_totals']['sum_main_plus_duplicates'],
            # collisions as defined: merged main images minus sum of source main images
            'collisions_main_minus_sources_main': merged_main - result['sources_totals']['sum_main_registry_count'],
        }

        return result
