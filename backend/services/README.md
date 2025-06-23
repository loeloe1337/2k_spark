# services

This folder contains service modules that provide various backend functionalities, acting as intermediaries between the core logic and external interfaces (APIs, databases, etc.).

- `data_service.py`: Handles data-related operations and services, such as data retrieval, updates, and synchronization.
- `enhanced_prediction_service.py`: Provides advanced prediction features, possibly leveraging machine learning models or additional data sources.
- `live_scores_service.py`: Manages live scores, real-time updates, and related data feeds.
- `match_prediction_service.py`: Implements the logic for generating match predictions, including model inference and result formatting.
- `supabase_service.py`: Integrates with Supabase for data storage, retrieval, and possibly authentication or other backend services.

The `services` folder encapsulates the main service logic, ensuring modularity and separation of concerns between business logic and service orchestration.
