# Data Ordering Output - Visual Structure

## Directory Tree (After data_ordering completes)

```
output_base/
│
├─ 📁 Las_Hoyas/                        [Non-duplicate images - Collection 1]
│  ├─ 📁 Arthropoda/
│  │  ├─ 📁 Insecta/
│  │  │  ├─ 🖼️ IMG_001.jpg
│  │  │  ├─ 🖼️ IMG_002.jpg
│  │  │  └─ ...
│  │  ├─ 📁 Arachnida/
│  │  │  └─ 🖼️ ...
│  │  └─ ...
│  └─ 📁 Mollusca/
│     └─ ...
│
├─ 📁 Otras_Colecciones/                [Alternative collections]
│  ├─ 📁 Buenache/                      [Collection 2]
│  │  ├─ 📁 Arthropoda/
│  │  │  └─ ...
│  │  └─ ...
│  └─ 📁 Montsec/                       [Collection 3]
│     ├─ 📁 Arthropoda/
│     │  └─ ...
│     └─ ...
│
├─ 📁 Duplicados/                       [Duplicate images - SEPARATE]
│  ├─ 📁 Arthropoda/                    [Same structure as above]
│  │  ├─ 📁 Insecta/
│  │  │  ├─ 🖼️ DUP_001.jpg
│  │  │  ├─ 🖼️ DUP_002.jpg
│  │  │  └─ ...
│  │  └─ ...
│  └─ 📄 duplicados_registro.xlsx       [Separate registry file!]
│
├─ 📁 Archivos_Texto/                   [Text files]
│  ├─ 📄 file_001.txt
│  ├─ 📄 file_002.csv
│  └─ ...
│
├─ 📁 Otros_Archivos/                   [Other file types]
│  ├─ 📄 document.pdf
│  ├─ 🎥 video.mp4
│  └─ ...
│
├─ 📁 Revision_Manual/                  [Images flagged for review]
│  ├─ 📁 Arthropoda/
│  │  └─ 📁 Insecta/
│  │     ├─ 🖼️ uncertain_001.jpg
│  │     └─ ...
│  └─ ...
│
├─ 📁 registries/                       [Main metadata registries]
│  ├─ 📊 anotaciones.xlsx               [Non-duplicate images]
│  ├─ 📊 archivos_texto.xlsx            [Text files]
│  ├─ 📊 archivos_otros.xlsx            [Other files]
│  └─ 📊 hashes.xlsx                    [Image hashes]
│
├─ 📁 logs/
│  └─ 📋 session_*.log
│
├─ 🔧 pipeline_state.json               [Processing state]
├─ 🔧 llm_cache.json                    [LLM cache]
├─ 📊 processing_summary.csv            [Quick summary]
└─ 🔧 deferred_decisions.json           [User decisions]
```

## Data Flow Diagram

```
BEFORE data_ordering:
┌─────────────────┐
│ Source Files    │
│ (MUPA, YCLH, etc)
└────────┬────────┘
         │
         ▼
    [SCANNING]
         │
         ▼
    [LLM ANALYSIS]
         │
    [PATTERN EXTRACTION]
         │
    [HASHING]
         │
    [DEDUPLICATION]
         │
         ▼
    [ORGANIZING] ◄── Files moved to organized structure
         │
         ▼
┌─────────────────────────────────────────┐
│ Organized Output (THIS IS WHAT YOU GET) │
├─────────────────────────────────────────┤
│ • Las_Hoyas/                            │
│ • Otras_Colecciones/                    │
│ • Duplicados/                           │
│ • Archivos_Texto/                       │
│ • Otros_Archivos/                       │
│ • Revision_Manual/                      │
│ • registries/                           │
└─────────────────────────────────────────┘
         │
         ▼
    [REGISTRY GENERATION] ◄── Excel files created AFTER move
         │
         ▼
┌─────────────────────────────────────────┐
│ Final Output (COMPLETE)                 │
├─────────────────────────────────────────┤
│ • anotaciones.xlsx                      │
│ • Duplicados/duplicados_registro.xlsx   │
│ • archivos_texto.xlsx                   │
│ • archivos_otros.xlsx                   │
│ • hashes.xlsx                           │
└─────────────────────────────────────────┘
```

## Registry Relationship Diagram

```
REGISTRIES (Excel files):
│
├─ registries/anotaciones.xlsx
│  ├─ Columns: uuid, specimen_id, original_path, current_path,
│  │            macroclass_label, class_label, genera_label, ...
│  └─ Rows: 60 (non-duplicate images)
│
├─ Duplicados/duplicados_registro.xlsx (SEPARATE FILE)
│  ├─ Same columns as anotaciones.xlsx
│  └─ Rows: 6 (duplicate images)
│
├─ registries/archivos_texto.xlsx
│  ├─ Columns: id, original_path, current_path, file_type, ...
│  └─ Rows: N (text files)
│
├─ registries/archivos_otros.xlsx
│  ├─ Columns: id, original_path, current_path, file_type, ...
│  └─ Rows: M (other files)
│
└─ registries/hashes.xlsx
   ├─ Columns: uuid, md5_hash, phash, file_path, ...
   └─ Rows: 66 (ALL images: 60 non-dup + 6 dup)
```

## File Counting Logic

```
CHECKER COUNTS:

Total = 66 (in your example)
│
├─ NON-DUPLICATE IMAGES: 60
│  ├─ From file system:
│  │  ├─ Las_Hoyas/** (recursive)
│  │  ├─ Otras_Colecciones/** (recursive)
│  │  └─ Revision_Manual/** (recursive)
│  └─ From registry:
│     └─ Row count of anotaciones.xlsx
│
├─ DUPLICATE IMAGES: 6
│  ├─ From file system:
│  │  └─ Duplicados/** (excluding .xlsx)
│  └─ From registry:
│     └─ Row count of Duplicados/duplicados_registro.xlsx
│
├─ TEXT FILES: N
│  ├─ From file system:
│  │  └─ Archivos_Texto/**
│  └─ From registry:
│     └─ Row count of archivos_texto.xlsx
│
└─ OTHER FILES: M
   ├─ From file system:
   │  └─ Otros_Archivos/**
   └─ From registry:
      └─ Row count of archivos_otros.xlsx
```

## Expected Match Matrix

```
Component              │ File System │ Registry │ Match │ Note
───────────────────────┼─────────────┼──────────┼───────┼─────────────────
Non-duplicate images   │     60      │    60    │  ✅   │ anotaciones.xlsx
Duplicate images       │      6      │     6    │  ✅   │ duplicados_registro.xlsx
Text files             │      N      │     N    │  ✅   │ archivos_texto.xlsx
Other files            │      M      │     M    │  ✅   │ archivos_otros.xlsx
───────────────────────┼─────────────┼──────────┼───────┼─────────────────
TOTAL                  │    60+6+N+M │  60+6+N+M│  ✅   │ All should match
```

## Checker Validation Process

```
START
 │
 ├─ Load registries
 │  ├─ anotaciones.xlsx
 │  ├─ duplicados_registro.xlsx (from Duplicados/)
 │  ├─ archivos_texto.xlsx
 │  ├─ archivos_otros.xlsx
 │  └─ hashes.xlsx
 │
 ├─ Traverse file system
 │  ├─ Count Las_Hoyas/** → non-dup count
 │  ├─ Count Otras_Colecciones/** → non-dup count
 │  ├─ Count Duplicados/** → dup count (skip .xlsx)
 │  ├─ Count Archivos_Texto/** → text count
 │  └─ Count Otros_Archivos/** → other count
 │
 ├─ Compare
 │  ├─ File system non-dup == anotaciones.xlsx rows?
 │  ├─ File system dup == duplicados_registro.xlsx rows?
 │  ├─ File system text == archivos_texto.xlsx rows?
 │  └─ File system other == archivos_otros.xlsx rows?
 │
 ├─ Extract taxonomy
 │  ├─ Read macroclasses from anotaciones.xlsx
 │  ├─ Read macroclasses from duplicados_registro.xlsx
 │  └─ Combine (dedup)
 │
 └─ Report results
    ├─ File counts ✅
    ├─ Registry counts ✅
    ├─ Discrepancies ✅
    └─ Taxonomy ✅

END
```

## Key Insights

🎯 **Two Registry Files**
```
anotaciones.xlsx          Duplicados/duplicados_registro.xlsx
─────────────────────     ──────────────────────────────────
Non-duplicate images      Duplicate images
(60 in your example)      (6 in your example)
Main registry/            Separate registry in
registries/folder         Duplicados/folder
```

🎯 **Two Separate Physical Locations**
```
Las_Hoyas/                Duplicados/
Otras_Colecciones/        (separate folder)
───────────────────────   
Non-duplicate images      Duplicate images
(60 files)                (6 files)
```

🎯 **Registry Created AFTER File Organization**
```
1. Files organized
   ├─ Move to Las_Hoyas/
   ├─ Move to Otras_Colecciones/
   ├─ Move to Duplicados/
   └─ etc.

2. THEN registries created
   ├─ Write anotaciones.xlsx
   ├─ Write duplicados_registro.xlsx
   ├─ Write archivos_texto.xlsx
   └─ Write archivos_otros.xlsx
```

---

This explains why counts must always match if pipeline completed successfully! ✅
