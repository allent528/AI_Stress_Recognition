import sys

try:
    with open("debug_output_5.txt", "r", encoding="utf-16") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading file: {e}")
    # Try default encoding just in case
    try:
        with open("debug_output_5.txt", "r") as f:
            print(f.read())
    except Exception as e2:
        print(f"Error reading file with default encoding: {e2}")
