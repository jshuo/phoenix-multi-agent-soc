#!/bin/bash

# Battery Analytics Service - Quick Start Script

set -e

echo "🚀 Battery Analytics Service - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env file with your configuration:"
    echo "   - Database credentials"
    echo "   - Weather API key (optional)"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Run tests
echo "Running tests..."
pytest --quiet tests/
if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "⚠️  Some tests failed (this is okay if database is not configured)"
fi
echo ""

# Start the service
echo "Starting Battery Analytics Service..."
echo "=========================================="
echo ""
echo "Service will be available at:"
echo "  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

python main.py
