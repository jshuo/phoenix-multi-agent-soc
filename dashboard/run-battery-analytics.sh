#!/bin/bash
# Battery Analytics - Python Setup and Run Script

echo "================================================================"
echo "  Battery Analytics - Python Setup"
echo "  Professional Kalman Filtering with FilterPy"
echo "================================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "📦 Setting up virtual environment..."
if [ ! -d "venv-battery" ]; then
    python3 -m venv venv-battery
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv-battery/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r examples/requirements-python.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Run the analysis
echo "================================================================"
echo "  🚀 Running Battery Analytics"
echo "================================================================"
echo ""

python3 examples/battery-analytics-python.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "  ✅ SUCCESS!"
    echo "================================================================"
    echo ""
    echo "Generated files are in the current directory:"
    echo "  📊 battery_kalman_analysis.png"
    echo "  📊 battery_noise_reduction.png"
    echo "  📊 battery_zscore_analysis.png"
    echo "  📄 battery_analytics_results.csv"
    echo "  📄 battery_analytics_summary.json"
    echo ""
    echo "To run again: source venv-battery/bin/activate && python3 examples/battery-analytics-python.py"
    echo ""
else
    echo ""
    echo "❌ Analysis failed. Check error messages above."
    exit 1
fi
