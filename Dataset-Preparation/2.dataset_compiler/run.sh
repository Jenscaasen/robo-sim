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

# List available folders (excluding venv and hidden folders)
echo "Available folders in current directory:"
echo ""

# Create array of folders
folders=()
counter=1

for dir in */; do
    # Skip venv and hidden folders
    if [[ "$dir" != "venv/" && "$dir" != .* ]]; then
        folder_name="${dir%/}"  # Remove trailing slash
        folders+=("$folder_name")
        echo "  $counter) $folder_name"
        ((counter++))
    fi
done

# Check if any folders were found
if [ ${#folders[@]} -eq 0 ]; then
    echo "No dataset folders found in current directory."
    exit 1
fi

echo ""
echo "Enter the number of the folder you want to process (or 'q' to quit):"
read -p "Selection: " selection

# Handle quit
if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
    echo "Exiting..."
    exit 0
fi

# Validate selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#folders[@]} ]; then
    echo "Invalid selection. Please enter a number between 1 and ${#folders[@]}."
    exit 1
fi

# Get selected folder
selected_folder="${folders[$((selection-1))]}"

echo ""
echo "Processing folder: $selected_folder"
echo ""

# Ask if user wants to generate debug images
echo "Do you want to generate debug images showing detected tips? (y/n):"
read -p "Debug images: " debug_choice

echo ""
echo "Running dataset compiler..."
echo ""

# Run the dataset compiler with or without debug images
if [[ "$debug_choice" == "y" || "$debug_choice" == "Y" ]]; then
    echo "Generating debug images..."
    python dataset_compiler.py "$selected_folder" --debug-images
    echo ""
    echo "Done! Check the generated files:"
    echo "  - CSV file: ${selected_folder}.csv"
    echo "  - Debug images: ${selected_folder}_debug_images/"
else
    python dataset_compiler.py "$selected_folder"
    echo ""
    echo "Done! Check the generated CSV file: ${selected_folder}.csv"
fi