"""
Metrics collector for analyzing data_ordering output.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd


class MetricsCollector:
    """
    Collects and analyzes various metrics from the data_ordering output.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metrics collector.
        
        Args:
            output_dir: Root directory of data_ordering output
        """
        self.output_dir = Path(output_dir)
        self.registries_dir = self.output_dir / "registries"
    
    def get_collection_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of files across collections.
        
        Returns:
            Dictionary with collection names and file counts
        """
        distribution = {}

        special_names = {
            'Duplicados', 'registries', 'logs', 'Archivos_Texto',
            'Otros_Archivos', 'Revision_Manual', '__pycache__'
        }

        # Treat any top-level directory that isn't a special folder as a collection
        for child in self.output_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in special_names:
                # count special folders too (e.g., Duplicados)
                if child.name == 'Duplicados':
                    count = sum(1 for f in child.rglob('*') if f.is_file() and not f.name.endswith('.xlsx'))
                    if count > 0:
                        distribution['Duplicados'] = count
                continue

            # For collection-like folders, count files (skip .xlsx files)
            count = sum(1 for f in child.rglob('*') if f.is_file() and not f.name.endswith('.xlsx'))
            if count > 0:
                distribution[child.name] = count

        # Also include Archivos_Texto and Otros_Archivos if present
        txt_dir = self.output_dir / 'Archivos_Texto'
        if txt_dir.exists():
            count = sum(1 for f in txt_dir.rglob('*') if f.is_file())
            if count > 0:
                distribution['Archivos_Texto'] = count
        other_dir = self.output_dir / 'Otros_Archivos'
        if other_dir.exists():
            count = sum(1 for f in other_dir.rglob('*') if f.is_file())
            if count > 0:
                distribution['Otros_Archivos'] = count

        return distribution
    
    def get_macroclass_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of files across macroclasses.
        
        Directory structure: Collection/Macroclass/Class/...
        
        Returns:
            Dictionary with macroclass names and file counts
        """
        distribution = defaultdict(int)

        special_names = {'Duplicados', 'registries', 'logs', 'Archivos_Texto', 'Otros_Archivos', 'Revision_Manual', '__pycache__'}

        # Iterate over collection-like top-level folders
        for child in self.output_dir.iterdir():
            if not child.is_dir() or child.name in special_names:
                continue
            # macroclass directories are first-level subdirs under collection folders
            for macroclass_dir in child.iterdir():
                if macroclass_dir.is_dir():
                    count = sum(1 for _ in macroclass_dir.rglob('*') if _.is_file())
                    if count > 0:
                        distribution[macroclass_dir.name] += count

        return dict(sorted(distribution.items()))
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of files across classes.
        
        Directory structure: Collection/Macroclass/Class/...
        
        Returns:
            Dictionary with class names and file counts
        """
        distribution = defaultdict(int)

        special_names = {'Duplicados', 'registries', 'logs', 'Archivos_Texto', 'Otros_Archivos', 'Revision_Manual', '__pycache__'}

        for child in self.output_dir.iterdir():
            if not child.is_dir() or child.name in special_names:
                continue
            for macroclass_dir in child.iterdir():
                if macroclass_dir.is_dir():
                    for class_dir in macroclass_dir.iterdir():
                        if class_dir.is_dir():
                            count = sum(1 for _ in class_dir.rglob('*') if _.is_file())
                            if count > 0:
                                distribution[class_dir.name] += count

        return dict(sorted(distribution.items()))
    
    def get_files_per_macroclass_class(self) -> Dict[str, Dict[str, int]]:
        """
        Get file counts organized by macroclass and class.
        
        Directory structure: Collection/Macroclass/Class/...
        
        Returns:
            Nested dictionary: {macroclass: {class: count}}
        """
        distribution = defaultdict(lambda: defaultdict(int))

        special_names = {'Duplicados', 'registries', 'logs', 'Archivos_Texto', 'Otros_Archivos', 'Revision_Manual', '__pycache__'}

        for child in self.output_dir.iterdir():
            if not child.is_dir() or child.name in special_names:
                continue
            for macroclass_dir in child.iterdir():
                if macroclass_dir.is_dir():
                    for class_dir in macroclass_dir.iterdir():
                        if class_dir.is_dir():
                            count = sum(1 for _ in class_dir.rglob('*') if _.is_file())
                            if count > 0:
                                distribution[macroclass_dir.name][class_dir.name] += count

        # Convert to regular dict
        return {k: dict(v) for k, v in distribution.items()}
    
    def get_duplicate_analysis(self) -> Dict[str, int]:
        """
        Analyze the Duplicados folder.
        
        Returns:
            Dictionary with duplicate statistics
        """
        stats = {
            'total_duplicate_files': 0,
            'duplicate_images': 0,
            'duplicate_text_files': 0,
            'duplicate_others': 0,
        }
        
        duplicados_dir = self.output_dir / "Duplicados"
        if not duplicados_dir.exists():
            return stats
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        text_extensions = {
            '.txt', '.csv', '.md', '.json', '.pdf', '.doc', '.docx',
            '.rtf', '.odt', '.xml', '.yaml', '.yml', '.tsv'
        }
        
        for file_path in duplicados_dir.rglob('*'):
            if file_path.is_file():
                # Skip the registry file itself
                if file_path.name == 'duplicados_registro.xlsx':
                    continue
                
                stats['total_duplicate_files'] += 1
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    stats['duplicate_images'] += 1
                elif ext in text_extensions:
                    stats['duplicate_text_files'] += 1
                else:
                    stats['duplicate_others'] += 1
        
        return stats
    
    def print_metrics_summary(self):
        """Print a summary of collected metrics."""
        print("\n" + "="*70)
        print("DISTRIBUTION METRICS")
        print("="*70)
        
        # Collection distribution
        print("\n📂 FILES PER COLLECTION:")
        print("-" * 50)
        collection_dist = self.get_collection_distribution()
        for collection, count in collection_dist.items():
            print(f"  {collection}: {count} files")
        
        # Macroclass distribution
        print("\n🔬 FILES PER MACROCLASS:")
        print("-" * 50)
        macroclass_dist = self.get_macroclass_distribution()
        for macroclass, count in macroclass_dist.items():
            print(f"  {macroclass}: {count} files")
        
        # Class distribution (top 15)
        print("\n📋 FILES PER CLASS (Top 15):")
        print("-" * 50)
        class_dist = self.get_class_distribution()
        for class_name, count in list(class_dist.items())[:15]:
            print(f"  {class_name}: {count} files")
        if len(class_dist) > 15:
            print(f"  ... and {len(class_dist) - 15} more classes")
        
        # Duplicate analysis
        print("\n🔄 DUPLICATE FILES ANALYSIS:")
        print("-" * 50)
        dup_stats = self.get_duplicate_analysis()
        print(f"  Total duplicate files: {dup_stats['total_duplicate_files']}")
        print(f"  Duplicate images:      {dup_stats['duplicate_images']}")
        print(f"  Duplicate text files:  {dup_stats['duplicate_text_files']}")
        print(f"  Duplicate others:      {dup_stats['duplicate_others']}")
        
        print("\n" + "="*70 + "\n")
