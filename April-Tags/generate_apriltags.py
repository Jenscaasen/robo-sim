#!/usr/bin/env python3
"""
Script to generate random April tag-like patterns as JPG images.
"""

import cv2
import numpy as np
import random
import os
from PIL import Image

def generate_april_tag_image(tag_id, size=200):
    """
    Generate an April tag-like image for a given tag ID.

    Args:
        tag_id (int): The tag ID to generate
        size (int): Size of the output image in pixels

    Returns:
        numpy.ndarray: The generated tag image
    """
    # Create white background
    img = np.ones((size, size), dtype=np.uint8) * 255

    # April tags have a specific structure:
    # - Black border
    # - White border inside
    # - Data area with black and white cells

    # Define margins and sizes
    outer_margin = size // 20  # 5% margin
    border_width = size // 20  # Border width
    data_area_size = size - 2 * (outer_margin + border_width)

    # Draw outer black border
    cv2.rectangle(img, (outer_margin, outer_margin),
                  (size - outer_margin, size - outer_margin), 0, -1)

    # Draw inner white border (background)
    inner_border_start = outer_margin + border_width
    inner_border_end = size - outer_margin - border_width
    cv2.rectangle(img, (inner_border_start, inner_border_start),
                  (inner_border_end, inner_border_end), 255, -1)

    # Draw data area background (black)
    data_start = inner_border_start + border_width
    data_end = inner_border_end - border_width
    cv2.rectangle(img, (data_start, data_start), (data_end, data_end), 0, -1)

    # Create data pattern (6x6 grid for AprilTag 36h11 family)
    cell_size = data_area_size // 6

    # Set random seed based on tag_id for reproducible patterns
    random.seed(tag_id)

    # Generate pattern data (avoid all-white or all-black for better detection)
    pattern_data = []
    for i in range(6):
        row = []
        for j in range(6):
            # Ensure we don't get all same color
            value = random.choice([0, 255])
            row.append(value)
        pattern_data.append(row)

    # Ensure at least one black and one white cell per row for better detection
    for i in range(6):
        if all(cell == 255 for cell in pattern_data[i]):  # All white
            pattern_data[i][random.randint(0, 5)] = 0  # Add one black
        elif all(cell == 0 for cell in pattern_data[i]):  # All black
            pattern_data[i][random.randint(0, 5)] = 255  # Add one white

    # Draw the pattern
    for i in range(6):
        for j in range(6):
            if pattern_data[i][j] == 255:  # White cell
                x1 = data_start + j * cell_size
                y1 = data_start + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)

    # Add tag ID as text overlay (optional, for identification)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(tag_id)
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (size - text_size[0]) // 2
    text_y = size - outer_margin - 10

    # Draw text background
    cv2.rectangle(img,
                  (text_x - 3, text_y - text_size[1] - 3),
                  (text_x + text_size[0] + 3, text_y + 3),
                  255, -1)

    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, 0.5, 0, 1, cv2.LINE_AA)

    return img

def main():
    """Generate 20 random April tag-like patterns and save them as JPG files."""
    # Create output directory if it doesn't exist
    output_dir = "April-Tags"
    os.makedirs(output_dir, exist_ok=True)

    # Generate 20 random tag IDs
    random_ids = random.sample(range(1000), 20)  # Random IDs from 0-999

    print(f"Generating {len(random_ids)} April tag-like patterns...")

    for i, tag_id in enumerate(random_ids):
        print(f"Generating April tag {tag_id} ({i+1}/{len(random_ids)})")

        # Generate the tag image
        tag_image = generate_april_tag_image(tag_id)

        # Save as JPG using PIL (more reliable than OpenCV for JPG)
        filename = os.path.join(output_dir, f"{tag_id:03d}.jpg")
        pil_image = Image.fromarray(tag_image)
        pil_image.save(filename, 'JPEG')

        print(f"Saved: {filename}")

    print(f"\nSuccessfully generated {len(random_ids)} April tag-like patterns in {output_dir}/")

if __name__ == "__main__":
    main()