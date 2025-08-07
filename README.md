# ProxyFeatureExtraction

A Python project for extracting and analyzing proxy-related features from datasets, with support for parallel processing and modular feature extraction.

## Project Structure

```
ProxyFeatureExtraction/
├── configs/                                    # Configuration files (YAML, JSON) for feature extraction
├── scripts/                                    # Main scripts to run feature extraction
│   ├── extract_all_features_parallel.py        # Main entry point (parallel execution)
│   └── extract_all_features.py                 # (Optional) Sequential version
├── src/
│   └── feature_extraction/
│       ├── data_io.py                          # Data input/output utilities
│       ├── preprocessing.py                    # Data preprocessing functions
│       └── extractors/                         # Feature extractor modules
│           ├── base_extractor.py
│           ├── corr_extractor.py
│           ├── hayes_usenix2019_features.py
│           ├── host_feature_helpers.py
│           └── ta_extractor.py
├── tests/                                      # Unit tests for core modules
└── temp/                                       # Temporary files and test outputs
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (see below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ProxyFeatureExtraction
   ```

2. **Install dependencies:**
   - (If you have a `requirements.txt`, run: `pip install -r requirements.txt`)
   - Otherwise, manually install required packages as needed.

### Configuration

- Edit the configuration files in the `configs/` directory to specify feature extraction parameters, input data locations, and other settings.

## Usage

### Main Entry Point

To run the feature extraction in parallel, use the following command from the project root:

```bash
PYTHONPATH=src python scripts/extract_all_features_parallel.py
```

- Edit `configs/ta_exp.yaml` with your desired configuration file.
- The script will orchestrate the entire feature extraction process, leveraging modules in `src/feature_extraction/`.

### Alternative: Sequential Extraction

For sequential processing (mainly for testing or debugging):

```bash
PYTHONPATH=src python scripts/extract_all_features.py
```

## Testing

Unit tests are located in the `tests/` directory. To run all tests:

```bash
PYTHONPATH=src pytest tests/
```

## Extending Feature Extraction

- Add new feature extractors in `src/feature_extraction/extractors/` by subclassing `base_extractor.py`.
- Update configuration files as needed to include new features.

## Directory Details

- **configs/**: YAML/JSON files for experiment and feature extraction configuration.
- **scripts/**: Entry-point scripts for running feature extraction.
- **src/feature_extraction/**: Core logic for data I/O, preprocessing, and feature extraction.
- **tests/**: Unit tests for core modules.
- **temp/**: Temporary files and outputs (not version-controlled).

