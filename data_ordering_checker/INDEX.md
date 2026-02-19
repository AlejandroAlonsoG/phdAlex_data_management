# 📚 Data Ordering Checker - Documentation Index

## Quick Start
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Commands and expected output
- **[README.md](README.md)** - Full user guide

## Understanding the Fixes
- **[SUMMARY.md](SUMMARY.md)** - What was wrong and what was fixed
- **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - Detailed fix explanations

## Reference Documentation
- **[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)** - Complete data_ordering output structure
- **[COMPLETE_REFERENCE.md](COMPLETE_REFERENCE.md)** - Exhaustive file and registry reference

## Source Code
- **[__init__.py](__init__.py)** - Package initialization
- **[__main__.py](__main__.py)** - Module entry point
- **[checker.py](checker.py)** - Main verification logic
- **[metrics.py](metrics.py)** - Distribution metrics collector
- **[cli.py](cli.py)** - Command-line interface

---

## The Problem (What You Reported)

You had:
- 60 non-duplicate images
- 6 duplicate images

But the checker reported:
- 66 non-duplicate images ❌
- 0 duplicate images ❌

## The Solution (What Was Fixed)

The checker was looking for the wrong directories and reading from the wrong registries:

1. **Directory Names**: Looking for `Duplicates/` but data_ordering creates `Duplicados/` ✅
2. **Registry Split**: Duplicates have their own registry in `Duplicados/duplicados_registro.xlsx` ✅
3. **Folder Structure**: Need to count specific folders, not traverse generically ✅
4. **Taxonomy**: Must read from BOTH registries ✅

## How to Use

### Basic Check
```bash
python -m data_ordering_checker.cli --output ./organized_data
```

### With All Metrics
```bash
python -m data_ordering_checker.cli --output ./organized_data --all-metrics
```

### Export Reports
```bash
python -m data_ordering_checker.cli --output ./organized_data \
  --json report.json \
  --csv ./reports
```

## Expected Output (Your Example)

```
📁 FILE SYSTEM STRUCTURE COUNTS:
  Non-duplicate images: 60 ✅
  Duplicate images:     6 ✅
  
📊 EXCEL REGISTRIES COUNTS:
  Non-duplicate images: 60 ✅
  Duplicate images:     6 ✅

⚠️  DISCREPANCIES:
  ✓ No discrepancies found! ✅
```

## Documentation Guide

### For Users
1. Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands
2. Check [README.md](README.md) for full features
3. Use [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) to understand what gets counted

### For Understanding the Fix
1. Read [SUMMARY.md](SUMMARY.md) for overview
2. Check [FIXES_APPLIED.md](FIXES_APPLIED.md) for technical details
3. See [COMPLETE_REFERENCE.md](COMPLETE_REFERENCE.md) for full reference

### For Developers
1. Look at [checker.py](checker.py) for main logic
2. Check [metrics.py](metrics.py) for distribution calculations
3. See [cli.py](cli.py) for command-line interface

## Key Points

✅ **Correct Directory Names**
- `Duplicados/` not `Duplicates/`
- `Archivos_Texto/` for text files
- `Otros_Archivos/` for other files
- `Las_Hoyas/` + `Otras_Colecciones/` for organized images

✅ **Separate Registries**
- Non-duplicates: `registries/anotaciones.xlsx`
- Duplicates: `Duplicados/duplicados_registro.xlsx` (separate!)
- Text files: `registries/archivos_texto.xlsx`
- Other files: `registries/archivos_otros.xlsx`

✅ **Accurate Counting**
- Non-duplicates from file system should match anotaciones.xlsx
- Duplicates from file system should match duplicados_registro.xlsx
- No discrepancies if pipeline completed correctly

✅ **Complete Taxonomy**
- Reads from BOTH anotaciones.xlsx and duplicados_registro.xlsx
- Combines macroclasses, classes, and genera from both sources

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Counts don't match | Check if pipeline stage is COMPLETED |
| Duplicates = 0 | Verify `Duplicados/duplicados_registro.xlsx` exists |
| Can't find registries | Ensure output path has `registries/` subfolder |
| Wrong directory names | Update path to data_ordering output root |

## Testing

With your 60+6 file example:
```bash
python -m data_ordering_checker.cli --output ./your_data
```

Should show:
- Non-duplicate: 60 / 60 ✅
- Duplicate: 6 / 6 ✅
- Discrepancies: 0 ✅

---

**Version**: 0.1.0 (corrected)  
**Status**: ✅ All issues fixed  
**Last Updated**: February 18, 2026

For questions or issues, refer to the relevant documentation file above.
