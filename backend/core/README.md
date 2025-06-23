# core

This folder contains the core logic and data processing modules for the backend. It is responsible for the main data operations, business logic, and feature engineering.

- `data/`: Contains submodules for data storage, data fetching (from external APIs or databases), and data processing (feature extraction, transformation, and modeling).
    - `fetchers/`: Modules for retrieving data from various sources, such as match history, tokens, and upcoming matches.
    - `processors/`: Modules for processing and transforming raw data, including feature engineering and model preparation.
    - `storage.py`: Handles data storage and retrieval logic, abstracting the underlying storage mechanism.

The `core` folder is the heart of the backend, implementing the essential algorithms and data workflows that power the application's main features.
