#!/bin/bash

# Dataset Compiler Runner Script
# This script activates the virtual environment and allows interactive folder selection

echo "=== Dataset Compiler ==="
echo "Activating virtual environment..."

# Check if venv exists
if [ ! -d "../venv" ]; then
    echo "Error: Virtual environment '../venv' not found!"
    echo "Please run: python3 -m venv ../venv && source ../venv/bin/activate && pip install opencv-python numpy"
    exit 1
fi

# Activate virtual environment
source ../venv/bin/activate

echo "Virtual environment activated."
echo ""

# List available CSV files (excluding hidden files)
echo "Available CSV files in current directory:"
echo ""

# Create array of CSV files
csv_files=()
counter=1

for file in *.csv; do
    # Skip hidden files and specific files we don't want to process
    if [[ "$file" != .* ]]; then
        csv_files+=("$file")
        echo "  $counter) $file"
        ((counter++))
    fi
done

# Check if any CSV files were found
if [ ${#csv_files[@]} -eq 0 ]; then
    echo "No CSV files found in current directory."
    exit 1
fi

echo ""
echo "Enter the number of the CSV file you want to clean (or 'q' to quit):"
read -p "Selection: " selection

# Handle quit
if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
    echo "Exiting..."
    exit 0
fi

# Validate selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#csv_files[@]} ]; then
    echo "Invalid selection. Please enter a number between 1 and ${#csv_files[@]}."
    exit 1
fi

# Get selected file
selected_file="${csv_files[$((selection-1))]}"

# Generate output filename
output_file="${selected_file%.*}_cleaned.csv"

echo ""
echo "Processing file: $selected_file"
echo "Output file will be: $output_file"
echo ""

echo "Running dataset cleaner..."
echo ""

# Run the dataset cleaner
python clean_dataset.py "$selected_file" "$output_file"

echo ""
echo "Done! Check the generated file:"
echo "  - Cleaned CSV file: $output_file"