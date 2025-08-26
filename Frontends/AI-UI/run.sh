#!/bin/bash

# AI-UI Run Script
# This script activates the virtual environment, installs requirements, and runs the application

echo "🚀 Starting AI-UI Application..."

# Check if venv directory exists
if [ ! -d "venv" ]; then
    echo "❌ Error: venv directory not found. Please create a virtual environment first:"
    echo "    python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  Warning: .env file not found. Copying from .env.example..."
        cp .env.example .env
        echo "📝 Please edit the .env file with your actual configuration values."
    else
        echo "❌ Error: Neither .env nor .env.example found. Please create a .env file."
        exit 1
    fi
fi

# Run the application
echo "🎯 Starting the application..."
python main.py