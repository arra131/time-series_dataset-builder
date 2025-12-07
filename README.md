# Time Series Dataset Builder

This project implements a HuggingFace `datasets` builder for ~8000 time-series datasets.  
It loads metadata from a configuration CSV, downloads datasets from Kaggle, standardizes schema (date column, value columns), and produces train/test splits.

Key features:
- Automatic Kaggle download + local caching
- Support for univariate and multivariate time series
- Robust date parsing and schema validation
- Modular structure using HuggingFace `GeneratorBasedBuilder`
- Reproducible 80/20 train-test split

