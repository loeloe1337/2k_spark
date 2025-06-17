# Testing Strategy & Implementation

## Overview

This document outlines the testing strategy for the 2K Spark project, including unit tests, integration tests, and end-to-end testing approaches. The project uses pytest for backend testing and React Testing Library for frontend testing.

## 🎯 Testing Philosophy

### Goals
- **Reliability**: Ensure the system works correctly under various conditions
- **Maintainability**: Make code changes with confidence
- **Documentation**: Tests serve as living documentation
- **Quality**: Catch bugs before they reach production

### Testing Pyramid
```
    /\
   /  \    E2E Tests (Few)
  /____\   
 /      \   Integration Tests (Some)
/__________\ Unit Tests (Many)
```

## 🔧 Backend Testing (Python)

### Setup

#### Install Testing Dependencies
```bash
cd backend
pip install pytest pytest-cov pytest-mock requests-mock
```

#### Test Configuration
Create `backend/pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --verbose
```

### Directory Structure
```
backend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests
│   │   ├── test_models.py
│   │   ├── test_data_processors.py
│   │   └── test_utils.py
│   ├── integration/             # Integration tests
│   │   ├── test_api_endpoints.py
│   │   ├── test_data_pipeline.py
│   │   └── test_services.py
│   └── fixtures/                # Test data
│       ├── sample_matches.json
│       └── sample_players.json
```

### Unit Testing Examples

#### Test Model Training
```python
# tests/unit/test_models.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from core.models.winner_prediction import WinnerPredictionModel

class TestWinnerPredictionModel:
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data for testing."""
        return {
            'features': pd.DataFrame({
                'home_win_rate': [0.7, 0.6, 0.8],
                'away_win_rate': [0.5, 0.7, 0.3],
                'home_avg_score': [20.5, 18.2, 22.1],
                'away_avg_score': [18.1, 19.5, 16.8]
            }),
            'targets': [1, 0, 1]  # 1 = home win, 0 = away win
        }
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = WinnerPredictionModel()
        assert model is not None
        assert hasattr(model, 'model')
    
    def test_model_training(self, sample_data):
        """Test model training with sample data."""
        model = WinnerPredictionModel()
        
        # Train the model
        metrics = model.train(
            sample_data['features'], 
            sample_data['targets']
        )
        
        # Check that training completed and returned metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert metrics['accuracy'] >= 0.0
        assert metrics['accuracy'] <= 1.0
    
    def test_model_prediction(self, sample_data):
        """Test model prediction functionality."""
        model = WinnerPredictionModel()
        model.train(sample_data['features'], sample_data['targets'])
        
        # Make predictions
        predictions = model.predict(sample_data['features'])
        
        assert len(predictions) == len(sample_data['features'])
        assert all(p in [0, 1] for p in predictions)
    
    @patch('core.models.winner_prediction.joblib.dump')
    def test_model_saving(self, mock_dump, sample_data):
        """Test model saving functionality."""
        model = WinnerPredictionModel()
        model.train(sample_data['features'], sample_data['targets'])
        
        # Save the model
        model.save('test_model.pkl')
        
        # Verify save was called
        mock_dump.assert_called_once()
```

#### Test Data Processing
```python
# tests/unit/test_data_processors.py
import pytest
from unittest.mock import Mock, patch
from core.data.processors.player_stats import PlayerStatsProcessor

class TestPlayerStatsProcessor:
    
    @pytest.fixture
    def sample_matches(self):
        """Sample match data for testing."""
        return [
            {
                'match_id': '1',
                'home_team': 'PLAYER1',
                'away_team': 'PLAYER2',
                'home_score': 21,
                'away_score': 18,
                'winner': 'PLAYER1',
                'date': '2025-06-01'
            },
            {
                'match_id': '2',
                'home_team': 'PLAYER2',
                'away_team': 'PLAYER3',
                'home_score': 15,
                'away_score': 22,
                'winner': 'PLAYER3',
                'date': '2025-06-02'
            }
        ]
    
    def test_calculate_win_rate(self, sample_matches):
        """Test win rate calculation."""
        processor = PlayerStatsProcessor()
        stats = processor.calculate_stats(sample_matches)
        
        # PLAYER1: 1 win, 0 losses = 100% win rate
        assert stats['PLAYER1']['win_rate'] == 1.0
        
        # PLAYER2: 0 wins, 2 losses = 0% win rate
        assert stats['PLAYER2']['win_rate'] == 0.0
        
        # PLAYER3: 1 win, 0 losses = 100% win rate
        assert stats['PLAYER3']['win_rate'] == 1.0
    
    def test_calculate_average_score(self, sample_matches):
        """Test average score calculation."""
        processor = PlayerStatsProcessor()
        stats = processor.calculate_stats(sample_matches)
        
        # PLAYER1: scored 21 in 1 game
        assert stats['PLAYER1']['avg_score'] == 21.0
        
        # PLAYER2: scored 18 + 15 in 2 games = 16.5 average
        assert stats['PLAYER2']['avg_score'] == 16.5
```

### Integration Testing Examples

#### Test API Endpoints
```python
# tests/integration/test_api_endpoints.py
import pytest
import json
from unittest.mock import patch, Mock
from app.api import app

class TestAPIEndpoints:
    
    @pytest.fixture
    def client(self):
        """Flask test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data."""
        return [
            {
                'match_id': '123',
                'home_team': 'PLAYER1',
                'away_team': 'PLAYER2',
                'predicted_winner': 'PLAYER1',
                'home_win_probability': 0.75,
                'away_win_probability': 0.25
            }
        ]
    
    @patch('builtins.open')
    @patch('json.load')
    def test_get_predictions_success(self, mock_json_load, mock_open, 
                                   client, sample_predictions):
        """Test successful prediction retrieval."""
        mock_json_load.return_value = sample_predictions
        
        response = client.get('/api/predictions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['match_id'] == '123'
    
    def test_get_predictions_no_file(self, client):
        """Test prediction endpoint when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            response = client.get('/api/predictions')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['predictions'] == []
    
    def test_refresh_endpoint(self, client):
        """Test refresh endpoint."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            response = client.post('/api/refresh')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
```

### Running Backend Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run tests matching pattern
pytest -k "test_model"

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

## 🌐 Frontend Testing (React)

### Setup

#### Install Testing Dependencies
```bash
cd frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest-environment-jsdom
```

#### Test Configuration
Create `frontend/jest.config.js`:
```javascript
const nextJest = require('next/jest')

const createJestConfig = nextJest({
  dir: './',
})

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapping: {
    '^@/components/(.*)$': '<rootDir>/src/components/$1',
    '^@/hooks/(.*)$': '<rootDir>/src/hooks/$1',
  },
  testEnvironment: 'jest-environment-jsdom',
}

module.exports = createJestConfig(customJestConfig)
```

### Directory Structure
```
frontend/
├── __tests__/                  # Test files
│   ├── components/
│   │   ├── PredictionCard.test.tsx
│   │   └── ScoreCard.test.tsx
│   ├── hooks/
│   │   └── use-predictions.test.ts
│   └── pages/
│       └── predictions.test.tsx
├── jest.config.js
└── jest.setup.js
```

### Component Testing Examples

#### Test Prediction Card Component
```typescript
// __tests__/components/PredictionCard.test.tsx
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import { PredictionCard } from '@/components/ui/prediction-card'

describe('PredictionCard', () => {
  const mockPrediction = {
    match_id: '123',
    home_team: 'PLAYER1',
    away_team: 'PLAYER2',
    predicted_winner: 'PLAYER1',
    home_win_probability: 0.75,
    away_win_probability: 0.25,
    match_date: '2025-06-18 20:00:00'
  }

  test('renders prediction information correctly', () => {
    render(<PredictionCard prediction={mockPrediction} />)
    
    expect(screen.getByText('PLAYER1')).toBeInTheDocument()
    expect(screen.getByText('PLAYER2')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
  })

  test('highlights predicted winner', () => {
    render(<PredictionCard prediction={mockPrediction} />)
    
    const winnerElement = screen.getByTestId('predicted-winner')
    expect(winnerElement).toHaveTextContent('PLAYER1')
    expect(winnerElement).toHaveClass('font-bold')
  })

  test('displays match date correctly', () => {
    render(<PredictionCard prediction={mockPrediction} />)
    
    expect(screen.getByText(/June 18, 2025/)).toBeInTheDocument()
  })
})
```

#### Test Custom Hook
```typescript
// __tests__/hooks/use-predictions.test.ts
import { renderHook, waitFor } from '@testing-library/react'
import { usePredictions } from '@/hooks/use-predictions'

// Mock fetch
global.fetch = jest.fn()

describe('usePredictions', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  test('fetches predictions successfully', async () => {
    const mockPredictions = [
      {
        match_id: '123',
        home_team: 'PLAYER1',
        away_team: 'PLAYER2',
        predicted_winner: 'PLAYER1'
      }
    ]

    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ predictions: mockPredictions })
    })

    const { result } = renderHook(() => usePredictions())

    await waitFor(() => {
      expect(result.current.predictions).toEqual(mockPredictions)
      expect(result.current.loading).toBe(false)
      expect(result.current.error).toBe(null)
    })
  })

  test('handles fetch error', async () => {
    ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'))

    const { result } = renderHook(() => usePredictions())

    await waitFor(() => {
      expect(result.current.predictions).toEqual([])
      expect(result.current.loading).toBe(false)
      expect(result.current.error).toBe('Failed to fetch predictions')
    })
  })
})
```

### Running Frontend Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test PredictionCard.test.tsx

# Run tests matching pattern
npm test -- --testNamePattern="prediction"
```

## 🔄 End-to-End Testing

### Setup Playwright

```bash
npm install --save-dev @playwright/test
npx playwright install
```

### E2E Test Examples

```typescript
// e2e/predictions.spec.ts
import { test, expect } from '@playwright/test'

test.describe('Predictions Page', () => {
  test('loads predictions successfully', async ({ page }) => {
    // Start the application
    await page.goto('http://localhost:3000/predictions')
    
    // Wait for predictions to load
    await expect(page.getByTestId('predictions-list')).toBeVisible()
    
    // Check for prediction cards
    const predictionCards = page.getByTestId('prediction-card')
    await expect(predictionCards.first()).toBeVisible()
  })

  test('can refresh predictions', async ({ page }) => {
    await page.goto('http://localhost:3000/predictions')
    
    // Click refresh button
    await page.getByRole('button', { name: /refresh/i }).click()
    
    // Check for loading state
    await expect(page.getByText(/loading/i)).toBeVisible()
    
    // Wait for predictions to reload
    await expect(page.getByTestId('predictions-list')).toBeVisible()
  })
})
```

## 📊 Test Coverage Goals

### Backend Coverage Targets
- **Overall**: > 80%
- **Core Models**: > 90%
- **API Endpoints**: > 85%
- **Data Processing**: > 85%
- **Utils**: > 90%

### Frontend Coverage Targets
- **Components**: > 80%
- **Hooks**: > 90%
- **Utils**: > 90%
- **Pages**: > 70%

## 🚀 Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r ../requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          cd backend
          pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      - name: Run tests
        run: |
          cd frontend
          npm test -- --coverage --watchAll=false
```

## 🛠️ Testing Best Practices

### General Principles
1. **Test Behavior, Not Implementation**: Focus on what the code does, not how
2. **Arrange, Act, Assert**: Structure tests clearly
3. **One Assertion Per Test**: Keep tests focused
4. **Descriptive Test Names**: Make the purpose clear
5. **Use Fixtures**: Share test data efficiently

### Backend Testing Tips
- Mock external dependencies (APIs, databases)
- Test edge cases and error conditions
- Use parametrized tests for multiple scenarios
- Test both success and failure paths

### Frontend Testing Tips
- Test user interactions, not implementation details
- Mock API calls consistently
- Test accessibility and responsive behavior
- Use data-testid for reliable element selection

---

**Last Updated**: June 17, 2025
