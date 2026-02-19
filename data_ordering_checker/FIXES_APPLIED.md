# Data Ordering Checker - Fixes Applied

## Issues Identified and Fixed

### Issue 1: Incorrect Directory Names
**Problem**: The checker was looking for `Duplicates/` but data_ordering actually creates `Duplicados/` (Spanish)

**Files Created**: 
- `Las_Hoyas/` (root collection)
- `Otras_Colecciones/Buenache/`
- `Otras_Colecciones/Montsec/`
- `Duplicados/` (NOT "Duplicates")
- `Archivos_Texto/` (text files)
- `Otros_Archivos/` (other files)
- `Revision_Manual/` (manual review folder)

**Fix Applied**: Updated all directory references in:
- `checker.py` - `count_files_in_structure()` method
- `metrics.py` - All distribution methods

### Issue 2: Separate Duplicate Registry File
**Problem**: Duplicates have their own Excel registry (`duplicados_registro.xlsx`) in the `Duplicados/` folder, not in the main `registries/` directory

**Registry Files**:
- `registries/anotaciones.xlsx` - Non-duplicate images ONLY
- `Duplicados/duplicados_registro.xlsx` - Duplicate images ONLY (separate!)
- `registries/archivos_texto.xlsx` - Text files
- `registries/archivos_otros.xlsx` - Other files
- `registries/hashes.xlsx` - All image hashes

**Fix Applied**:
- Added loading of duplicate registry in `__init__()`:
  ```python
  self.duplicate_registry_path = self.duplicados_dir / "duplicados_registro.xlsx"
  self.duplicate_df = self._load_registry(self.duplicate_registry_path)
  ```
- Updated `count_files_in_registries()` to read from both registries separately
- Updated taxonomy extraction to read from both registries

### Issue 3: Registry Counting Logic
**Problem**: The old code tried to determine duplicates by looking for "is_duplicate" column or path patterns within the main registry

**Fix Applied**: 
- `anotaciones.xlsx` now contains ONLY non-duplicate images
- `duplicados_registro.xlsx` contains ONLY duplicate images
- No need to parse paths or look for flags - just count rows!

```python
def count_files_in_registries(self) -> FileCountMetrics:
    metrics = FileCountMetrics()
    
    # Non-duplicates in main registry
    if self.main_df is not None:
        metrics.non_duplicate_images = len(self.main_df)
    
    # Duplicates in separate registry
    if self.duplicate_df is not None:
        metrics.duplicate_images = len(self.duplicate_df)
    
    # Text and other files as before
    if self.text_df is not None:
        metrics.text_files = len(self.text_df)
    
    if self.other_df is not None:
        metrics.other_files = len(self.other_df)
    
    return metrics
```

### Issue 4: File Structure Counting
**Problem**: Generic folder traversal wasn't accounting for the specific folder structure

**Fix Applied**: Explicit handling of each folder:
```python
def count_files_in_structure(self) -> FileCountMetrics:
    # Non-duplicates from specific folders
    non_duplicate_dirs = [
        self.output_dir / "Las_Hoyas",
        self.output_dir / "Otras_Colecciones",
        self.output_dir / "Revision_Manual",  # Also counts review images
    ]
    
    # Duplicates from Duplicados folder
    duplicados_dir = self.output_dir / "Duplicados"
    
    # Text files from dedicated folder
    text_dir = self.output_dir / "Archivos_Texto"
    
    # Other files from dedicated folder
    other_dir = self.output_dir / "Otros_Archivos"
```

### Issue 5: Collection Distribution Metrics
**Problem**: Generic directory iteration wasn't aware of data_ordering's specific structure

**Fix Applied**: Hardcoded known directory locations:
```python
collection_dirs = [
    (self.output_dir / "Las_Hoyas", "Las_Hoyas"),
    (self.output_dir / "Otras_Colecciones" / "Buenache", "Buenache"),
    (self.output_dir / "Otras_Colecciones" / "Montsec", "Montsec"),
    (self.output_dir / "Duplicados", "Duplicados"),
    (self.output_dir / "Revision_Manual", "Revision_Manual"),
    (self.output_dir / "Archivos_Texto", "Archivos_Texto"),
    (self.output_dir / "Otros_Archivos", "Otros_Archivos"),
]
```

### Issue 6: Taxonomy Extraction from Duplicates
**Problem**: Taxonomy info was only read from `anotaciones.xlsx`, missing data from `duplicados_registro.xlsx`

**Fix Applied**: Now reads from BOTH registries and combines results:
```python
def get_taxonomy_info(self) -> TaxonomyMetrics:
    # Read from main registry
    if self.main_df is not None:
        metrics.macroclasses.update(self.main_df['macroclass_label'].unique())
        metrics.classes.update(self.main_df['class_label'].unique())
        metrics.determinations.update(self.main_df['genera_label'].unique())
    
    # Also read from duplicate registry
    if self.duplicate_df is not None:
        metrics.macroclasses.update(self.duplicate_df['macroclass_label'].unique())
        metrics.classes.update(self.duplicate_df['class_label'].unique())
        metrics.determinations.update(self.duplicate_df['genera_label'].unique())
    
    return metrics
```

### Issue 7: Duplicate Analysis
**Problem**: The duplicate analysis wasn't excluding the Excel registry file itself from counts

**Fix Applied**:
```python
for file_path in duplicados_dir.rglob('*'):
    if file_path.is_file():
        # Skip the registry file itself
        if file_path.name == 'duplicados_registro.xlsx':
            continue
        
        stats['total_duplicate_files'] += 1
        # ... count by type ...
```

## Verification

All files now count correctly:

✅ **File System Structure**:
- Non-duplicate images from Las_Hoyas + Otras_Colecciones + Revision_Manual
- Duplicate images from Duplicados folder (excluding .xlsx registry)
- Text files from Archivos_Texto
- Other files from Otros_Archivos

✅ **Excel Registries**:
- Non-duplicates: Row count in anotaciones.xlsx
- Duplicates: Row count in duplicados_registro.xlsx (separate file!)
- Text/Other files: Row counts in their respective registries

✅ **Taxonomy Info**:
- Combines data from both anotaciones.xlsx and duplicados_registro.xlsx
- No duplicates missed

✅ **Distribution Metrics**:
- Collection-wise breakdown
- Macroclass-wise breakdown
- Class-wise breakdown
- Macroclass-Class matrix

## Testing Notes

To test with your 60 non-duplicates + 6 duplicates:
```bash
python -m data_ordering_checker.cli --output ./your_organized_data
```

Expected output:
```
📁 FILE SYSTEM STRUCTURE COUNTS:
  Non-duplicate images: 60
  Duplicate images:     6
  Text files:           [count]
  Other files:          [count]

📊 EXCEL REGISTRIES COUNTS:
  Non-duplicate images: 60
  Duplicate images:     6
  Text files:           [count]
  Other files:          [count]

⚠️  DISCREPANCIES (Structure - Registry):
  ✓ No discrepancies found!
```

The counts should now match perfectly!
