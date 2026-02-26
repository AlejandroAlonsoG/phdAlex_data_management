# Annotation Validator

Validates annotations produced by the data ordering pipeline's merge output.

## What It Does

Randomly samples annotation records and presents them for human review, checking:

1. **Field Extraction** — Do the extracted fields (specimen_id, macroclass, class, year, source) match what's visible in the original file paths?
2. **Duplicate Detection** — Are the identified duplicates actually duplicates? Do their fields align?
3. **Image–Field Consistency** — Do the annotation fields make sense when looking at the actual image?

## Usage

```sh
cd annotation_validator
pip install -r requirements.txt
python app.py
```

Then select the merge output directory (the one containing `registries/` or `registros/` and optionally `Duplicados/`).

## Interface

| Area | Description |
|------|-------------|
| **Left panel** | Main image display + duplicate thumbnails (click to enlarge) |
| **Right panel** | Annotation fields, original paths, path analysis comparison, hash info, duplicate field comparison |
| **Bottom bar** | Notes field + verdict buttons (Valid / Invalid / Uncertain / Skip) |
| **Header** | Stats, filters (has duplicates, missing fields), save session |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` or `V` | Mark as Valid |
| `2` or `I` | Mark as Invalid |
| `3` or `U` | Mark as Uncertain |
| `→` or `N` | Skip (no verdict) |
| `M` | Back to main image (when viewing a duplicate) |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Open directory |

## Output

Validation sessions are saved as Excel files (`validation_session_YYYYMMDD_HHMMSS.xlsx`) in the output directory with columns:

- `uuid`, `specimen_id`, `original_path`, `current_path`
- `macroclass_label`, `class_label`, `genera_label`, `campaign_year`, `fuente`
- `n_duplicates`, `verdict`, `notes`, `reviewed_at`

## Path Analysis

The tool extracts clues from original file paths and compares them with annotation fields:

- **Specimen IDs**: Patterns like `LH-12345`, `MCCM-LH-001`, standalone 5–8 digit numbers
- **Years**: 4-digit numbers in 1900–2029 range
- **Sources**: Keywords like `MUPA`, `YCLH`
- **Taxa**: Known taxonomic class names (e.g., `Insecta`, `Osteichthyes`)
- **Macroclasses**: Inferred from detected taxa

Status symbols: `✓` match · `~` partial · `✗` mismatch · `—` not in path · `⚠` empty annotation

## Requirements

- Python 3.9+
- Pillow (image display)
- pandas + openpyxl (Excel reading)
- Optional: rawpy (for RAW camera formats like .orf, .nef, .cr2)
