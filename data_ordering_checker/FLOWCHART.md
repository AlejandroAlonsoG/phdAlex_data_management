# 📊 Data Ordering Checker - Flowchart & Decision Trees

## Checker Execution Flowchart

```
START
│
├─→ Initialize Checker
│   ├─ Load: anotaciones.xlsx (non-dup images)
│   ├─ Load: Duplicados/duplicados_registro.xlsx (dup images)
│   ├─ Load: archivos_texto.xlsx (text files)
│   ├─ Load: archivos_otros.xlsx (other files)
│   └─ Load: hashes.xlsx (validation)
│
├─→ Count Files on Disk (File System)
│   ├─ Traverse Las_Hoyas/** → non_dup_count
│   ├─ Traverse Otras_Colecciones/** → non_dup_count (add)
│   ├─ Traverse Revision_Manual/** → non_dup_count (add)
│   ├─ Traverse Duplicados/** (skip .xlsx) → dup_count
│   ├─ Traverse Archivos_Texto/** → text_count
│   └─ Traverse Otros_Archivos/** → other_count
│
├─→ Count Records in Registries (Excel Files)
│   ├─ len(anotaciones.xlsx) → non_dup_registry
│   ├─ len(duplicados_registro.xlsx) → dup_registry
│   ├─ len(archivos_texto.xlsx) → text_registry
│   └─ len(archivos_otros.xlsx) → other_registry
│
├─→ Calculate Discrepancies
│   ├─ non_dup_file_count == non_dup_registry? YES → ✅ / NO → ⚠️
│   ├─ dup_file_count == dup_registry? YES → ✅ / NO → ⚠️
│   ├─ text_file_count == text_registry? YES → ✅ / NO → ⚠️
│   └─ other_file_count == other_registry? YES → ✅ / NO → ⚠️
│
├─→ Extract Taxonomy
│   ├─ Read macroclasses from anotaciones.xlsx
│   ├─ Read macroclasses from duplicados_registro.xlsx
│   ├─ Combine and deduplicate
│   ├─ Same for classes
│   └─ Same for determinations
│
├─→ Calculate Distribution Metrics
│   ├─ Files per collection
│   ├─ Files per macroclass
│   ├─ Files per class
│   └─ Macroclass-Class matrix
│
├─→ Format Output
│   ├─ Console report (formatted)
│   ├─ JSON export (if requested)
│   └─ CSV exports (if requested)
│
└─→ END (with results)
```

## Directory Traversal Decision Tree

```
Starting at: output_base/

├─ Is "Las_Hoyas" folder?
│  YES → Traverse recursively → Count all image files
│
├─ Is "Otras_Colecciones" folder?
│  YES → Enter it
│       ├─ Is "Buenache" folder?
│       │  YES → Traverse recursively → Add to non-dup count
│       └─ Is "Montsec" folder?
│          YES → Traverse recursively → Add to non-dup count
│
├─ Is "Duplicados" folder?
│  YES → Traverse recursively
│       └─ For each file:
│           ├─ Is "duplicados_registro.xlsx"?
│           │  YES → SKIP
│           │  NO → Is image? → Count as duplicate
│                  Is text? → Count as duplicate text
│                  Other? → Count as duplicate other
│
├─ Is "Archivos_Texto" folder?
│  YES → Traverse recursively
│       └─ For each file:
│           └─ Count in text_count
│
├─ Is "Otros_Archivos" folder?
│  YES → Traverse recursively
│       └─ For each file:
│           └─ Count in other_count
│
└─ Is "Revision_Manual" folder?
   YES → Traverse recursively
        └─ For each image file:
            └─ Add to non_dup_count (these are review-flagged, not duplicates)
```

## Registry Matching Decision Tree

```
For each file type:

NON-DUPLICATE IMAGES:
├─ File system count = anotaciones.xlsx row count?
│  ├─ YES: ✅ Match - Display green check
│  └─ NO: ⚠️ Discrepancy
│         ├─ More files on disk? → Files not registered yet
│         └─ More in registry? → Files deleted from disk

DUPLICATE IMAGES:
├─ File system count = Duplicados/duplicados_registro.xlsx row count?
│  ├─ YES: ✅ Match - Display green check
│  └─ NO: ⚠️ Discrepancy
│         ├─ More files on disk? → Registry incomplete
│         └─ More in registry? → Files deleted from disk

TEXT FILES:
├─ File system count = archivos_texto.xlsx row count?
│  ├─ YES: ✅ Match
│  └─ NO: ⚠️ Discrepancy

OTHER FILES:
├─ File system count = archivos_otros.xlsx row count?
│  ├─ YES: ✅ Match
│  └─ NO: ⚠️ Discrepancy
```

## Taxonomy Extraction Logic

```
BUILDING TAXONOMY SET:

macroclasses = {}

├─ From anotaciones.xlsx:
│  ├─ Read column: macroclass_label
│  └─ Add all unique, non-null values → macroclasses.add()
│
├─ From duplicados_registro.xlsx:
│  ├─ Read column: macroclass_label
│  └─ Add all unique, non-null values → macroclasses.add()
│
└─ Result: Combined set of ALL macroclasses

SAME PROCESS FOR:
├─ classes (from class_label column)
└─ determinations (from genera_label column)

OUTPUT: Three sorted lists with all unique taxonomy terms
```

## Error Handling Flow

```
LOADING EACH REGISTRY:

Try:
├─ Open .xlsx file
├─ Read into dataframe
└─ Return dataframe

Except:
├─ File not found → Return None
├─ Read error → Return None
├─ Corruption → Return None
└─ Permission denied → Return None

IN COUNTING LOGIC:

For each registry:
├─ If df is None:
│  └─ Treat as 0 records (graceful)
├─ If df is empty:
│  └─ len(df) = 0 (correct)
└─ If df has records:
   └─ len(df) = number of rows (correct)

RESULT: Never crashes, always gives best possible count
```

## Output Format Decision

```
User requests checker with options:

checker(output_dir)
    ├─ No arguments
    │  └─ Print to console only
    │     ├─ File counts
    │     ├─ Registry counts
    │     ├─ Discrepancies
    │     ├─ Taxonomy
    │     └─ Summary
    │
    ├─ --json output.json
    │  ├─ Generates: JSON file
    │  └─ Format: {file_structure_counts, registry_counts, discrepancies, taxonomy, summary}
    │
    ├─ --csv ./reports/
    │  ├─ Generates: file_counts.csv
    │  │              ├─ File Type | File Structure | Registry | Difference
    │  │              └─ Rows for each type
    │  └─ Generates: taxonomy.csv
    │               ├─ Macroclass | Class | Determination
    │               └─ All unique values
    │
    ├─ --all-metrics
    │  ├─ Adds to console:
    │  │  ├─ Collection distribution
    │  │  ├─ Macroclass distribution
    │  │  ├─ Class distribution
    │  │  └─ Duplicate analysis
    │  │
    │  └─ And all normal output
    │
    └─ Combinations:
       └─ --json + --csv + --all-metrics = Everything!
```

## Discrepancy Report Logic

```
CALCULATING DIFFERENCES:

For each file type:
├─ diff = file_system_count - registry_count
├─ If diff == 0:
│  └─ Print: ✓ Type: 0 (match)
├─ If diff > 0:
│  └─ Print: ✗ Type: +diff (more files than registry)
├─ If diff < 0:
│  └─ Print: ✗ Type: diff (fewer files than registry)
│
└─ Final check:
   ├─ If all diffs == 0:
   │  └─ "✓ No discrepancies found!" (green)
   ├─ Else:
   │  └─ List all mismatches (red)

INTERPRETATION:
├─ Discrepancy = 0 → ✅ Pipeline completed successfully
├─ Discrepancy > 0 → Files added after pipeline
├─ Discrepancy < 0 → Files deleted after pipeline
└─ Multiple diffs → Data consistency issue
```

## File Counting Algorithm (Pseudocode)

```
FUNCTION count_files_in_structure():
    metrics = FileCountMetrics()
    
    # Non-duplicate images
    FOR dir IN [Las_Hoyas, Otras_Colecciones, Revision_Manual]:
        IF dir.exists():
            FOR file IN dir.rglob('*'):
                IF file.is_file() AND file.extension in [.jpg, .png, ...]:
                    metrics.non_duplicate_images += 1
    
    # Duplicate images
    IF Duplicados.exists():
        FOR file IN Duplicados.rglob('*'):
            IF file.is_file():
                IF file.name == 'duplicados_registro.xlsx':
                    CONTINUE  # Skip registry file
                ELSE IF file.extension in [.jpg, .png, ...]:
                    metrics.duplicate_images += 1
    
    # Text files
    IF Archivos_Texto.exists():
        FOR file IN Archivos_Texto.rglob('*'):
            IF file.is_file() AND file.extension in [.txt, .csv, .md, .json]:
                metrics.text_files += 1
    
    # Other files
    IF Otros_Archivos.exists():
        FOR file IN Otros_Archivos.rglob('*'):
            IF file.is_file():
                metrics.other_files += 1
    
    RETURN metrics
```

## Registry Counting Algorithm (Pseudocode)

```
FUNCTION count_files_in_registries():
    metrics = FileCountMetrics()
    
    # Load registries
    main_df = load('registries/anotaciones.xlsx')
    dup_df = load('Duplicados/duplicados_registro.xlsx')
    text_df = load('registries/archivos_texto.xlsx')
    other_df = load('registries/archivos_otros.xlsx')
    
    # Count rows
    IF main_df is not None:
        metrics.non_duplicate_images = len(main_df)
    
    IF dup_df is not None:
        metrics.duplicate_images = len(dup_df)
    
    IF text_df is not None:
        metrics.text_files = len(text_df)
    
    IF other_df is not None:
        metrics.other_files = len(other_df)
    
    RETURN metrics
```

---

This flowchart shows exactly how the checker works, making it easy to understand the logic! 📊
