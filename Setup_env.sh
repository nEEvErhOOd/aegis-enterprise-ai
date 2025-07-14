#!/bin/bash
# Industrial-grade environment setup script

# Create virtual environment
python3 -m venv gemini_env
source gemini_env/bin/activate

# Install core requirements
pip install --upgrade pip
pip install tensorflow transformers scikit-learn pandas flask gunicorn

# Install security monitoring dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip install psutil
elif [[ "$OSTYPE" == "darwin"* ]]; then
    pip install psutil
elif [[ "$OSTYPE" == "win32" ]]; then
    pip install psutil
fi

# Create directory structure
mkdir -p logs
mkdir -p data/processed
mkdir -p models/checkpoints

# Download security threat database
curl -o data/threat_signatures.json https://example.com/threats/latest

# Set environment variables
export FLASK_ENV=production
export COGNITIVE_LOAD_LIMIT=0.85
export SECURITY_LEVEL=8

echo "Environment setup complete. Activate with: source gemini_env/bin/activate"
