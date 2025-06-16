# 2K Flash - NBA 2K25 eSports Match Prediction System.

2K Flash is a comprehensive prediction system for NBA 2K25 eSports matches in the H2H GG League. The system collects real data from the H2H GG League API, processes player statistics, and uses advanced machine learning models to predict match winners and scores with high accuracy.

## Features

- Real-time data collection from H2H GG League API
- Comprehensive player statistics calculation with advanced metrics
- Machine learning models with Bayesian optimization for accurate predictions
- Cross-validated winner prediction model with feature selection
- Score prediction model with low MAE (Mean Absolute Error)
- RESTful API for frontend-backend communication
- Modern Next.js frontend with Shadcn UI components in dark mode
- Automated model training, validation, and optimization

## Project Structure

```
2k_spark/
├── backend/                   # Backend code
│   ├── app/                   # Application entry points
│   │   ├── api.py             # API server implementation
│   │   ├── cli.py             # Command-line interface
│   │   ├── optimize_score_model.py  # Score model optimization
│   │   ├── optimize_winner_model.py # Winner model optimization
│   │   └── clean_model_registry.py  # Model registry maintenance
│   ├── config/                # Configuration management
│   ├── core/                  # Core business logic
│   │   ├── data/              # Data access and processing
│   │   ├── models/            # Prediction models
│   │   │   ├── base.py        # Base model class
│   │   │   ├── winner_prediction.py # Winner prediction model
│   │   │   ├── score_prediction.py  # Score prediction model
│   │   │   ├── registry.py    # Model registry
│   │   │   └── feature_engineering.py # Feature engineering
│   │   └── optimization/      # Model optimization
│   ├── services/              # Service layer
│   ├── utils/                 # Utility functions
│   └── tests/                 # Tests
├── frontend/                  # Frontend code
│   ├── app/                   # Next.js app directory
│   ├── components/            # React components
│   ├── lib/                   # Utility libraries
│   ├── hooks/                 # Custom React hooks
│   └── public/                # Static assets
├── output/                    # Output data files
├── models/                    # Trained models
├── logs/                      # Log files
└── memory-bank/               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Chrome browser (for Selenium)
- ChromeDriver (for Selenium)

### Backend Setup

1. Install Python dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```
   python app/api.py
   ```

### CLI Usage

The CLI provides various commands for data fetching, model training, optimization, and maintenance:

```
# Data Collection
python app/cli.py fetch-token        # Fetch authentication token
python app/cli.py fetch-history      # Fetch match history
python app/cli.py fetch-upcoming     # Fetch upcoming matches
python app/cli.py calculate-stats    # Calculate player statistics

# Model Training
python app/cli.py train-winner-model # Train winner prediction model
python app/cli.py train-score-model  # Train score prediction model
python app/cli.py list-models        # List trained models

# Model Optimization
python app/cli.py optimize-score-model    # Optimize score prediction model with Bayesian optimization
python app/cli.py optimize-winner-model   # Optimize winner prediction model with Bayesian optimization

# Model Maintenance
python app/cli.py clean-model-registry    # Clean model registry by removing problematic models
```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   cd frontend
   npm install
   ```

2. Run the development server:
   ```
   npm run dev
   ```

## API Endpoints

- `GET /api/predictions`: Get predictions for upcoming matches
- `GET /api/score-predictions`: Get score predictions for upcoming matches
- `GET /api/prediction-history`: Get historical predictions with filtering
- `GET /api/stats`: Get prediction statistics and metrics
- `POST /api/refresh`: Trigger data refresh and prediction update

## Deployment

This application can be deployed for free using Vercel (frontend) and Render (backend).

### Frontend Deployment (Vercel)

1. Fork or clone this repository to your GitHub account
2. Sign up for a free account at [Vercel](https://vercel.com)
3. Create a new project in Vercel and import your GitHub repository
4. Configure the following settings:
   - Framework Preset: Next.js
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Environment Variables:
     - `NEXT_PUBLIC_API_URL`: Your backend URL (e.g., https://2k-spark-backend.onrender.com)
5. Click "Deploy"

### Backend Deployment (Render)

1. Sign up for a free account at [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the following settings:
   - Name: 2k-spark-backend (or your preferred name)
   - Root Directory: `backend`
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app/api.py`
   - Environment Variables:
     - `CORS_ORIGINS`: Your frontend URL (e.g., https://2k-spark.vercel.app)
5. Click "Create Web Service"

### Notes

- The free tier of Render will spin down after periods of inactivity, which may cause a slight delay on the first request
- The application is configured to refresh data periodically
- Make sure to set the environment variables correctly to ensure proper communication between frontend and backend

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- H2H GG League for providing the data API
- NBA 2K25 eSports community
