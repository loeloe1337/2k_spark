# Quick Start Guide

Get up and running with 2K Spark in minutes! This guide provides the fastest path to a working development environment.

## 🚀 Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **Node.js 18+** installed
- **Chrome browser** (for data fetching)
- **Git** for version control

## ⚡ 5-Minute Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd 2k_spark
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt

# Test the installation
python app/api.py
```

**Expected Output**: API server running on `http://localhost:5000`

### 3. Frontend Setup

```bash
# Open new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Expected Output**: Frontend running on `http://localhost:3000`

### 4. Verify Installation

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **API Health**: http://localhost:5000/api/stats

## 🔧 Common Issues & Solutions

### Python Virtual Environment Issues

**Problem**: `python -m venv` not working
```bash
# Alternative method
pip install virtualenv
virtualenv .venv
```

**Problem**: Package installation fails
```bash
# Upgrade pip first
python -m pip install --upgrade pip
pip install -r ../requirements.txt
```

### Node.js Issues

**Problem**: `npm install` fails
```bash
# Clear npm cache
npm cache clean --force
npm install
```

**Problem**: Port 3000 already in use
```bash
# Use different port
npm run dev -- -p 3001
```

### Chrome/Selenium Issues

**Problem**: ChromeDriver not found
```bash
# Install webdriver-manager (already in requirements.txt)
pip install webdriver-manager
```

## 📋 Next Steps

Now that you have the basic setup working:

1. **Explore the API**: Check out [API Documentation](./api-docs.md)
2. **Understand the Code**: Review [Project Structure](#project-structure)
3. **Run Tests**: Follow the [Testing Guide](./testing.md)
4. **Make Changes**: See [Development Workflow](#development-workflow)

## 🏗️ Project Structure

```
2k_spark/
├── backend/                 # Python API backend
│   ├── app/                # Main application files
│   ├── core/               # Business logic
│   ├── services/           # Service layer
│   └── config/             # Configuration
├── frontend/               # Next.js React frontend
│   ├── src/app/           # App router pages
│   ├── src/components/    # React components
│   └── src/hooks/         # Custom hooks
├── docs/                  # Documentation
├── logs/                  # Application logs
└── output/                # Generated data files
```

## 🔄 Development Workflow

### Making Backend Changes

1. **Edit Code**: Make changes in `backend/` directory
2. **Test Locally**: Run `python app/api.py` to test
3. **Check Logs**: Monitor logs in `logs/` directory
4. **Validate**: Use API endpoints to verify changes

### Making Frontend Changes

1. **Edit Components**: Modify files in `frontend/src/`
2. **Hot Reload**: Changes appear automatically in browser
3. **Check Console**: Monitor browser console for errors
4. **Test Responsive**: Check mobile and desktop views

### Data Pipeline Testing

```bash
# Fetch fresh data
cd backend
python app/cli.py fetch-token
python app/cli.py fetch-history
python app/cli.py calculate-stats

# Train models
python app/cli.py train-winner-model
python app/cli.py train-score-model

# Generate predictions
python app/cli.py refresh
```

## 🐛 Troubleshooting

### API Not Responding

1. Check if backend is running: `http://localhost:5000/api/stats`
2. Verify virtual environment is activated
3. Check logs in `logs/api.log`

### Frontend Not Loading

1. Ensure Node.js is installed: `node --version`
2. Clear browser cache
3. Check browser console for errors

### Data Fetching Issues

1. Verify Chrome is installed
2. Check internet connection
3. Review token fetching logs

## 📖 Additional Resources

- **Full Setup Guide**: [Environment Setup](./environment-setup.md)
- **API Reference**: [API Documentation](./api-docs.md)
- **Component Guide**: [Frontend Components](../components/frontend/)
- **Troubleshooting**: [Maintenance Guide](../operations/maintenance.md)

---

**Need Help?** Check the logs in the `logs/` directory or review the detailed documentation for specific components.

**Last Updated**: June 17, 2025
