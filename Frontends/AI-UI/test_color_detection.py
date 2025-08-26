#!/usr/bin/env python3
"""
Test script for the new color detection tool in Simulator client.
"""

import asyncio
from simulator_client import SimulatorClient

async def test_color_detection():
    """Test the color detection functionality"""

    # Initialize simulator client
    simulator = SimulatorClient(host="127.0.0.1", port=5000)

    print("Testing color detection tool...")

    # Test the detect_color_position method directly
    try:
        print("\n1. Testing direct simulator method with red:")
        result = await simulator.detect_color_position("red")
        print(f"Result: {result}")

        print("\n2. Testing with different colors:")
        colors_to_test = ["blue", "green", "yellow", "orange", "purple"]
        for color in colors_to_test:
            print(f"\nTesting {color}:")
            result = await simulator.detect_color_position(color)
            print(f"Result: {result}")

        print("\n3. Testing with invalid color:")
        result = await simulator.detect_color_position("invalid_color")
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await simulator.close()

if __name__ == "__main__":
    asyncio.run(test_color_detection())