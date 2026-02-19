# Complete Data Ordering Output Files Reference

## Directories Always Created

### Organized Image Collections
- ✅ `Las_Hoyas/` - Main collection (default, Las Hoyas specimens)
- ✅ `Otras_Colecciones/Buenache/` - Alternative collection
- ✅ `Otras_Colecciones/Montsec/` - Alternative collection

### File Organization
- ✅ `Duplicados/` - Duplicate image files
- ✅ `Revision_Manual/` - Images flagged for manual review
- ✅ `Archivos_Texto/` - Text files (.txt, .csv, .md, .json)
- ✅ `Otros_Archivos/` - Other file types

### Registries and Metadata
- ✅ `registries/` - Excel registries directory

### Logs and State
- ✅ `logs/` - Log files

## Excel Registry Files

### In registries/ directory
- ✅ `anotaciones.xlsx` - Non-duplicate images registry
- ✅ `archivos_texto.xlsx` - Text files registry
- ✅ `archivos_otros.xlsx` - Other files registry
- ✅ `hashes.xlsx` - Image hashes (MD5 + perceptual hash)

### In Duplicados/ directory
- ✅ `duplicados_registro.xlsx` - Duplicate images registry (SEPARATE!)

## JSON/CSV/Log Files

### Root directory
- ✅ `pipeline_state.json` - Current processing state for resume
- ✅ `llm_cache.json` - LLM analysis cache
- ✅ `processing_summary.csv` - Summary of all processed files
- ✅ `deferred_decisions.json` - User decisions during processing (if any)

### logs/ directory
- ✅ `session_*.log` - Detailed processing logs

## What the Checker Reads

### Excel Files (Registries)
- ✅ `registries/anotaciones.xlsx` - For non-duplicate image metadata and taxonomy
- ✅ `registries/archivos_texto.xlsx` - For text file count
- ✅ `registries/archivos_otros.xlsx` - For other file count
- ✅ `registries/hashes.xlsx` - For validation (not used in current checks)
- ✅ `Duplicados/duplicados_registro.xlsx` - For duplicate metadata and taxonomy

### Directory Trees (File Counts)
- ✅ `Las_Hoyas/` - For non-duplicate image count
- ✅ `Otras_Colecciones/` - For non-duplicate image count
- ✅ `Revision_Manual/` - For images flagged for review
- ✅ `Duplicados/` - For duplicate file count (excluding .xlsx file)
- ✅ `Archivos_Texto/` - For text file count validation
- ✅ `Otros_Archivos/` - For other file count validation

## What the Checker DOESN'T Use (Currently)

### Could be useful in future versions
- `llm_cache.json` - LLM analysis results cache
- `processing_summary.csv` - Alternative to registry check
- `deferred_decisions.json` - User decisions log
- `pipeline_state.json` - Progress tracking
- `hashes.xlsx` - More detailed duplicate analysis

## Directory Structure for Non-Duplicate Images

```
Las_Hoyas/
├── Arthropoda/
│   ├── Insecta/
│   │   └── IMG_001.jpg
│   ├── Arachnida/
│   │   └── IMG_002.jpg
│   └── ...
├── Mollusca/
│   ├── Cephalopoda/
│   │   └── IMG_003.jpg
│   └── ...
└── ...

Otras_Colecciones/
├── Buenache/
│   ├── Arthropoda/
│   │   ├── Insecta/
│   │   │   └── IMG_004.jpg
│   │   └── ...
│   └── ...
├── Montsec/
│   ├── Arthropoda/
│   │   └── ...
│   └── ...
└── ...
```

## Directory Structure for Duplicates

```
Duplicados/
├── Arthropoda/
│   ├── Insecta/
│   │   ├── IMG_005.jpg
│   │   └── IMG_006.jpg
│   └── ...
├── Mollusca/
│   └── ...
└── duplicados_registro.xlsx
```

Note: Duplicates maintain the same taxonomic structure as non-duplicates!

## Validation Matrix

| Component | Checker Reads | Validates Against | Status |
|-----------|---------------|-------------------|--------|
| Non-duplicate images | File system: Las_Hoyas + Otras_Colecciones + Revision_Manual | anotaciones.xlsx row count | ✅ |
| Duplicate images | File system: Duplicados/*.jpg | duplicados_registro.xlsx row count | ✅ |
| Text files | File system: Archivos_Texto | archivos_texto.xlsx row count | ✅ |
| Other files | File system: Otros_Archivos | archivos_otros.xlsx row count | ✅ |
| Taxonomy | anotaciones.xlsx + duplicados_registro.xlsx | Extracted macroclasses, classes, genera | ✅ |
| Collections | File system directories | Distribution across Las_Hoyas, Buenache, Montsec | ✅ |
| Macroclasses | File system structure | Distribution across taxonomy groups | ✅ |
| Classes | File system structure | Distribution within macroclasses | ✅ |
| Duplicates detail | duplicados_registro.xlsx | Comments field shows "duplicate_of" info | ✅ |

## Critical Facts About File Organization

1. **Non-duplicates and duplicates are SEPARATE**
   - Non-duplicates in `anotaciones.xlsx`
   - Duplicates in `Duplicados/duplicados_registro.xlsx`
   - They are NOT mixed in the main registry!

2. **Duplicates maintain taxonomic structure**
   - Found in `Duplicados/Arthropoda/Insecta/etc`
   - Same structure as non-duplicates
   - But isolated in their own folder

3. **Text and other files are separated**
   - Text files → `Archivos_Texto/`
   - Other files → `Otros_Archivos/`
   - Not mixed with images

4. **Revision flagged images are separate**
   - Found in `Revision_Manual/`
   - Same taxonomic structure
   - Counted as non-duplicates (they are unique images, just needing review)

5. **Order of operations**
   1. Files move first (ORGANIZING stage)
   2. Registries created AFTER move (REGISTRY_GENERATION stage)
   3. So `current_path` in registries = final destination

## Testing Your Output

Run checker to verify your data:

```bash
python -m data_ordering_checker.cli --output ./your_organized_data --all-metrics
```

This will:
1. Count all files by type in file system
2. Read all Excel registries
3. Compare file system vs registry counts
4. Report any discrepancies
5. Extract and list all taxonomy information
6. Show distribution metrics

Expected (60 non-dup + 6 dup example):
- Non-duplicate images: 60 (registry) = 60 (files)
- Duplicate images: 6 (registry) = 6 (files in Duplicados)
- ✓ No discrepancies
