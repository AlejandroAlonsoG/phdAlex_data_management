# Data Ordering Output Structure - Complete Reference

## Directory Structure

After running `data_ordering`, the output directory contains:

```
output_base/
├── Las_Hoyas/                          # Default collection (Las Hoyas specimens)
│   ├── Arthropoda/                     # Macroclass
│   │   ├── Insecta/                    # Class
│   │   │   ├── specimen_123.jpg
│   │   │   ├── specimen_124.jpg
│   │   │   └── ...
│   │   ├── Arachnida/
│   │   │   └── ...
│   │   └── ...
│   ├── Mollusca/
│   │   └── ...
│   └── ...
│
├── Otras_Colecciones/
│   ├── Buenache/                       # Other collection
│   │   ├── Arthropoda/
│   │   │   ├── Insecta/
│   │   │   │   └── *.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── Montsec/                        # Another collection
│   │   ├── Arthropoda/
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── Duplicados/                         # Duplicate images (same structure as above)
│   ├── Arthropoda/
│   │   ├── Insecta/
│   │   │   ├── duplicate_001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── duplicados_registro.xlsx        # Separate registry for duplicates
│
├── Archivos_Texto/                     # Text files (.txt, .csv, .md, .json)
│   ├── file_001.txt
│   ├── file_002.csv
│   └── ...
│
├── Otros_Archivos/                     # Other file types (non-image, non-text)
│   ├── document.pdf
│   ├── video.mp4
│   └── ...
│
├── Revision_Manual/                    # Images flagged for manual review
│   ├── Arthropoda/
│   │   ├── Insecta/
│   │   │   ├── uncertain_001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── registries/                         # Excel registries with metadata
│   ├── anotaciones.xlsx                # Non-duplicate images registry
│   ├── archivos_texto.xlsx             # Text files registry
│   ├── archivos_otros.xlsx             # Other files registry
│   └── hashes.xlsx                     # Image hashes (MD5 + perceptual hash)
│
├── logs/
│   └── session_*.log                   # Detailed processing logs
│
├── pipeline_state.json                 # State file for resumable processing
├── llm_cache.json                      # LLM analysis cache
├── processing_summary.csv              # Summary of all processed files
└── deferred_decisions.json             # User decisions during processing
```

## Excel Registries Format

### anotaciones.xlsx (Non-Duplicate Images Registry)

| Column | Type | Description |
|--------|------|-------------|
| uuid | str | Unique identifier for the image |
| specimen_id | str | Specimen ID extracted from filename/path |
| original_path | str | Original path(s) - if multiple duplicates, semicolon-separated |
| current_path | str | Final destination path in organized structure |
| macroclass_label | str | Major taxonomic group (Arthropoda, Mollusca, etc.) |
| class_label | str | Taxonomic class (Insecta, Osteichthyes, etc.) |
| genera_label | str | Genus-level determination |
| campaign_year | str | Year/campaign of collection |
| fuente | str | Source/collection code (MUPA, YCLH, MCCM, etc.) |
| comentarios | str | Comments and auto-generated notes |
| created_at | str | ISO timestamp of creation |

**Row count**: Total non-duplicate images processed

### duplicados_registro.xlsx (Duplicate Images Registry)

Located in: `Duplicados/duplicados_registro.xlsx`

Same columns as anotaciones.xlsx with additional info in `comentarios` field:
- `[AUTO] duplicate_of=<original_file_path>` - indicates which file this is a duplicate of
- `[REVIEW] <reason>` - if flagged for manual review

**Row count**: Total duplicate images identified

### archivos_texto.xlsx (Text Files Registry)

| Column | Type | Description |
|--------|------|-------------|
| id | str | Generated ID (TXT000001, TXT000002, etc.) |
| original_path | str | Original file path |
| current_path | str | Destination path in Archivos_Texto/ |
| original_filename | str | Original filename before moving |
| file_type | str | File extension (.txt, .csv, .md, .json) |
| processed | bool | Whether content was processed for metadata extraction |
| extracted_info | str | Any extracted metadata from file |
| created_at | str | ISO timestamp |

### archivos_otros.xlsx (Other Files Registry)

| Column | Type | Description |
|--------|------|-------------|
| id | str | Generated ID (OTH000001, OTH000002, etc.) |
| original_path | str | Original file path |
| current_path | str | Destination path in Otros_Archivos/ |
| original_filename | str | Original filename |
| file_type | str | File extension |
| created_at | str | ISO timestamp |

### hashes.xlsx (Image Hashes Registry)

| Column | Type | Description |
|--------|------|-------------|
| uuid | str | Image UUID (references anotaciones.xlsx or duplicados_registro.xlsx) |
| md5_hash | str | MD5 checksum for exact duplicate detection |
| phash | str | Perceptual hash for near-duplicate detection |
| file_path | str | Current file path |
| created_at | str | ISO timestamp |

**Row count**: Total images processed (includes both non-duplicates AND duplicates)

## File Counting Logic

### Non-Duplicate Images
Counted from:
- `Las_Hoyas/` folder (recursive)
- `Otras_Colecciones/` folder (recursive)
- `Revision_Manual/` folder (recursive)
- Should match row count in `anotaciones.xlsx`

### Duplicate Images
Counted from:
- `Duplicados/` folder (excluding `duplicados_registro.xlsx` file itself)
- Should match row count in `duplicados_registro.xlsx`

### Text Files
Counted from:
- `Archivos_Texto/` folder
- Extensions: .txt, .csv, .md, .json
- Should match row count in `archivos_texto.xlsx`

### Other Files
Counted from:
- `Otros_Archivos/` folder
- Any file not matching image or text extensions
- Should match row count in `archivos_otros.xlsx`

## Collection Codes

Recognized collections and their folder locations:

| Code | Full Name | Location |
|------|-----------|----------|
| LH, MUPA, YCLH, MCCM, etc. | Las Hoyas | `Las_Hoyas/` |
| BUE, K-BUE, CER-BUE | Buenache | `Otras_Colecciones/Buenache/` |
| MON | Montsec | `Otras_Colecciones/Montsec/` |

**Default**: If collection cannot be determined, file goes to `Las_Hoyas/`

## Deduplication Process

The pipeline identifies duplicates in two stages:

1. **Exact Duplicates** (MD5 hash match)
   - Same binary content
   - High confidence of duplication
   
2. **Near Duplicates** (Perceptual hash match)
   - Slightly different images (compression, rotation, crops)
   - Lower confidence - may be flagged for review

Duplicates are:
- Moved to `Duplicados/` maintaining taxonomic structure
- Recorded in `duplicados_registro.xlsx`
- NOT included in `anotaciones.xlsx`
- Can still be referenced via original_path in main registry

## Processing Summary CSV

Located at: `processing_summary.csv`

Quick CSV with all files and their processing results:
- original_path, filename, specimen_id, collection
- taxonomic_class, determination, campaign_year
- is_duplicate, destination_path, md5_hash

Useful for Excel analysis and filtering.

## State and Resume

### pipeline_state.json

Records current processing state for resume capability:
- Current stage (SCANNING, LLM_ANALYSIS, HASHING, ORGANIZING, etc.)
- Total files scanned
- Files organized so far
- Last directory analyzed
- All processed file metadata
- Decision log entries

### Stages

1. INIT - Initialization
2. SCANNING - Discover files
3. LLM_ANALYSIS - Extract metadata with LLM
4. PATTERN_EXTRACTION - Extract from filenames
5. METADATA_RECONCILIATION - Resolve conflicts
6. HASHING - Compute MD5/perceptual hashes
7. DEDUPLICATION - Identify duplicates
8. REGISTRY_GENERATION - Create Excel files
9. ORGANIZING - Move/copy files to final locations
10. COMPLETED - Finished

Note: Files are moved BEFORE registries are created, so `current_path` in registries reflects final location.

## Manual Intervention Points

User may be asked to review/decide:

1. **Unknown Collection** - File doesn't match known collection codes
2. **Collection Discrepancy** - Different sources suggest different collections
3. **Duplicate Decision** - Marginal duplicates may be reviewed
4. **Metadata Conflicts** - Multiple sources disagree on taxonomy

Decisions are logged in `deferred_decisions.json`.

## Data Validation with Checker

Run `data_ordering_checker` to verify:

```bash
python -m data_ordering_checker.cli --output ./organized_data --all-metrics
```

This will:
- Count files by type in each location
- Compare against registry counts
- Report discrepancies
- Extract all taxonomy information
- Show distribution across collections and classes
