# Data Ordering Tool

A Python tool for organizing and preparing fossil image datasets for deep learning training.

## Features

- **Pattern Extraction**: Automatically extracts specimen IDs, taxonomy, and campaign dates from file paths and names
- **Duplicate Detection**: Uses perceptual hashing (pHash) and MD5 checksums to identify exact and near-duplicate images
- **LLM Integration**: Optional Gemini AI analysis for automatic taxonomy classification and specimen ID regex generation
- **Collection Detection**: Automatically categorizes files by geographical collection (Las Hoyas, Buenache, Montsec)
- **Excel Registries**: Generates organized Excel files with all image metadata
- **Resumable Processing**: Save and resume long-running operations

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` (or create `.env`):
```
GEMINI_API_KEY=your_api_key_here
```

2. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)

## Usage

### Command Line Interface

```bash
# Dry run (preview without moving files)
python cli.py --source ./MUPA --output ./output --dry-run

# Process with LLM analysis
python cli.py --source ./MUPA ./YCLH --output ./organized --use-llm

# Resume interrupted processing
python cli.py --output ./organized --resume

# Verbose output
python cli.py --source ./images --output ./output --verbose
```

### Python API

```python
from data_ordering import (
    PipelineOrchestrator,
    run_pipeline,
    PatternExtractor,
    FileScanner,
)

# Quick run
run_pipeline(
    source_dirs=["./MUPA", "./YCLH"],
    output_base="./output",
    dry_run=True,
    use_llm=True,
)

# Manual control
orchestrator = PipelineOrchestrator(
    source_dirs=["./MUPA"],
    output_base="./output",
    dry_run=False,
    use_llm=True,
)
orchestrator.run()
summary = orchestrator.get_summary()
print(summary)
```

## Output Structure

```
output/
├── Las_Hoyas/
│   ├── Insecta/
│   │   ├── Coleoptera/
│   │   │   └── LH 15083.jpg
│   │   └── ...
│   ├── Crustacea/
│   │   └── ...
│   └── ...
├── Buenache/
│   └── ...
├── Montsec/
│   └── ...
├── Unknown_Collection/
│   └── Unsorted/
│       └── ...
├── Duplicates/
│   └── ...
├── registries/
│   ├── anotaciones.xlsx
│   └── summary.csv
├── logs/
│   └── session_*.log
├── pipeline_state.json
└── llm_cache.json
```

## Specimen ID Formats

The tool recognizes these specimen ID formats:

| Collection | Prefixes | Examples |
|------------|----------|----------|
| Las Hoyas | LH, MUPA, YCLH, MCCM, MCCM-LH, ADL, MDCLM | `LH-12345678-a`, `MUPA 8765` |
| Buenache | K-BUE, CER-BUE, PB (3-5 digits) | `K-Bue 085`, `PB 7005b` |

Camera codes (DSC, IMG, P1, PA, PC, etc.) are automatically filtered out.

## Architecture

```
data_ordering/
├── __init__.py           # Package exports
├── config.py             # Configuration and settings
├── logger_module.py      # Structured logging
├── excel_manager.py      # Excel registry management
├── file_utils.py         # File system utilities
├── pattern_extractor.py  # Regex pattern extraction
├── image_hasher.py       # Perceptual hashing
├── file_scanner.py       # File scanning and categorization
├── llm_integration.py    # Gemini LLM integration
├── main_orchestrator.py  # Pipeline orchestration
├── cli.py                # Command line interface
└── tests/
    ├── test_phase1.py
    ├── test_phase2.py
    ├── test_phase3.py
    ├── test_phase4.py
    ├── test_phase5.py
    └── test_phase6.py
```

## Pipeline Stages

1. **SCANNING**: Traverse source directories, classify files by type
2. **LLM_ANALYSIS**: Analyze directories with Gemini for taxonomy/collection info
3. **PATTERN_EXTRACTION**: Extract specimen IDs and metadata from paths/filenames
4. **HASHING**: Compute MD5 and perceptual hashes for duplicate detection
5. **DEDUPLICATION**: Identify and group duplicate images
6. **ORGANIZING**: Copy files to organized directory structure
7. **REGISTRY_GENERATION**: Create Excel registries with all metadata

## Running Tests

```bash
# Run all tests
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
python test_phase5.py
python test_phase6.py

# Or all at once
python -c "import subprocess; [subprocess.run(['python', f'test_phase{i}.py']) for i in range(1,7)]"
```

## Version History

- **0.7.0**: Main orchestrator and CLI complete
- **0.6.0**: LLM integration with structured output
- **0.5.0**: File scanner with duplicate detection
- **0.4.0**: Perceptual hashing
- **0.3.0**: Pattern extraction
- **0.2.0**: Core infrastructure

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests before submitting PR
4. Update documentation as needed
