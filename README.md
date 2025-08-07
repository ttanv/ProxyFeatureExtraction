# ProxyFeatureExtraction

A Python-based proxy feature extraction and classification system for analyzing network traffic patterns. The project extracts features from proxy traffic datasets and classifies different types of network attacks using various machine learning models.

## Two-Stage Pipeline Architecture

1. **Feature Extraction**: Processes network traffic data and applies various attacks/defenses to extract features
2. **Classification**: Trains and evaluates ML models (XGBoost, CNN, Transformer) on extracted features

## Project Structure

```
ProxyFeatureExtraction/
├── configs/                                    # Configuration files
│   ├── background_distributions.json           # Background traffic distributions
│   ├── classification_config.yaml             # Classification model settings
│   ├── extraction_config.yaml                 # Feature extraction parameters
│   ├── gateway_classification_config.yaml     # Gateway experiment settings
│   └── ta_exp.yaml                            # Timing analysis experiment config
├── scripts/                                    # Main execution scripts
│   ├── extract_all_features_parallel.py        # Parallel feature extraction
│   ├── extract_all_features.py                 # Sequential feature extraction
│   ├── run_classification.py                   # Binary classification experiments
│   ├── run_gateway_classification.py           # Gateway classification experiments
│   ├── run_extraction.py                       # Main feature extraction entry point
│   └── run_gateway_extraction.py               # Gateway-specific extraction
├── src/
│   ├── classification/                         # Classification pipeline
│   │   ├── data.py                            # Data loading and preprocessing
│   │   ├── evaluate.py                        # Model evaluation utilities
│   │   ├── train.py                           # Training pipeline
│   │   └── models/                            # ML model implementations
│   │       ├── xgboost.py                     # XGBoost classifier
│   │       ├── cnn.py                         # CNN for sequence analysis
│   │       ├── transformer.py                 # Transformer model
│   │       ├── corr_transformer.py            # Correlation-based transformer
│   │       └── deep_coffea.py                 # DeepCoFFEA implementation
│   └── feature_extraction/                    # Feature extraction pipeline
│       ├── data_io.py                         # Data input/output utilities
│       ├── preprocessing.py                   # Data preprocessing functions
│       └── extractors/                        # Feature extractor modules
│           ├── base_extractor.py              # Base extractor interface
│           ├── corr_extractor.py              # Correlation features
│           ├── ta_extractor.py                # Timing analysis features
│           ├── slt_extractor.py               # Statistical features
│           ├── thesis_extractor.py            # Thesis-specific features
│           └── hayes_usenix2019_features.py   # USENIX 2019 features
├── data/                                       # Extracted feature datasets
├── final_data/                                 # Final processed datasets
├── results/                                    # Experiment results
│   ├── classification/                        # Binary classification results
│   └── gateway_classification/                # Gateway experiment results
├── tests/                                      # Unit tests
│   ├── test_feature_extraction/               # Feature extraction tests
│   └── test_classification/                   # Classification tests
└── notebooks/                                  # Jupyter notebooks for analysis
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

Edit the configuration files in the `configs/` directory to specify experiment parameters:

- **`extraction_config.yaml`**: Controls feature extraction experiments, attack parameters, and data paths
- **`classification_config.yaml`**: Specifies model type, training parameters, and data preprocessing options
- **`gateway_classification_config.yaml`**: Gateway experiment settings (multi-class + binary classification)
- **`background_distributions.json`**: Background traffic distributions for decorrelation attacks

## Usage

### Feature Extraction

#### Main Entry Point (Recommended)
```bash
# Run feature extraction with parallel processing
PYTHONPATH=src python scripts/run_extraction.py
```

#### Alternative Methods
```bash
# Parallel extraction (alternative script)
PYTHONPATH=src python scripts/extract_all_features_parallel.py

# Sequential processing (for debugging)
PYTHONPATH=src python scripts/extract_all_features.py
```

### Classification

#### Binary Classification (Relay vs Background)
```bash
PYTHONPATH=src python scripts/run_classification.py --config configs/classification_config.yaml
```

#### Gateway Classification Experiments
```bash
# Run gateway classification experiments (multi-class + binary)
PYTHONPATH=src python scripts/run_gateway_classification.py --config configs/gateway_classification_config.yaml
```

The gateway classification includes:
1. **Multi-class Classification**: Gateway vs Relay vs Background (3 classes)
   - Labels: 0=background, 1=relay, 2=gateway
2. **Binary Classification**: Gateway vs (Relay + Background) (2 classes)
   - Labels: 0=background+relay, 1=gateway

**Data Sources**:
- Background and Relay data: `final_data/br/` folder
- Gateway data: `final_data/none/` folder

## Testing

Unit tests are located in the `tests/` directory. To run all tests:

```bash
# Run all tests
PYTHONPATH=src pytest tests/

# Run specific test module
PYTHONPATH=src pytest tests/test_feature_extraction/test_corr_extractor.py
```

## Model Types

- **XGBoost**: Primary model for tabular feature classification
- **MultiClassXGBoost**: Specialized XGBoost for 3-class classification (gateway vs relay vs background)
- **CNN**: For sequence-based feature analysis
- **Transformer**: For attention-based feature learning
- **CorrTransformer**: Correlation-based transformer with CorrTransform
- **DeepCoFFEA**: Deep learning implementation for network traffic analysis

Configure model selection and hyperparameters in `configs/classification_config.yaml` or `configs/gateway_classification_config.yaml`.

## Key Components

### Feature Extractors
Modular extractors in `src/feature_extraction/extractors/` for different feature types:
- **Correlation features** (`corr_extractor.py`): Timing correlations and statistical relationships
- **Timing analysis** (`ta_extractor.py`): Inter-packet delays and timing patterns
- **Statistical features** (`slt_extractor.py`): Basic statistical measurements
- **Research features** (`thesis_extractor.py`, `hayes_usenix2019_features.py`): Academic implementations

### Data Processing
Handles attack simulation via `DataProcessor` class:
- Bias removal
- Decorrelation attacks
- Padding and reshaping
- Jitter injection

### Configuration-Driven Architecture
YAML configs control:
- Experiment parameters
- Data paths and sources
- Model settings and hyperparameters
- Attack simulation parameters

## Data Structure

The project expects structured data directories:
- Raw data in train/test/val splits organized by attack type
- Output features saved in CSV format with batch processing
- Results stored in `results/` directory with model artifacts and evaluation metrics

## Extending the System

### Adding New Feature Extractors
1. Create new extractor in `src/feature_extraction/extractors/` by inheriting from `BaseExtractor`
2. Implement the `process_df()` method
3. Update configuration files to include new features

### Adding New Models
1. Implement new model in `src/classification/models/`
2. Follow existing model interfaces
3. Update classification configuration to support new model type

## Development Notes

- **PYTHONPATH**: Always set `PYTHONPATH=src` when running scripts to ensure proper module imports
- **Multiprocessing**: Feature extraction uses ProcessPoolExecutor with 'spawn' method for parallel processing
- **Batch Processing**: Data is processed in configurable batch sizes to manage memory usage
- **Attack Simulation**: The `DataProcessor` class applies various network attacks for robustness testing

