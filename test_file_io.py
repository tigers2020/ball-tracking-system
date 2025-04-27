"""
Test script for file_io module.
"""
import os
import tempfile
from pathlib import Path

from src.utils.file_io import JsonFileIO, XmlFileIO, CsvFileIO, FileIOFactory

def test_json_file_io():
    """Test JsonFileIO class."""
    print("Testing JsonFileIO...")
    
    # Create test data
    test_data = {
        "name": "Ball Tracking System",
        "version": "1.0.0",
        "settings": {
            "hsv": {
                "hue_min": 30,
                "hue_max": 90,
                "saturation_min": 100,
                "saturation_max": 255,
                "value_min": 100,
                "value_max": 255
            },
            "roi": {
                "left": {
                    "x": 100,
                    "y": 100,
                    "width": 300,
                    "height": 300
                },
                "right": {
                    "x": 100,
                    "y": 100,
                    "width": 300,
                    "height": 300
                }
            }
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create JsonFileIO instance
        json_io = JsonFileIO()
        
        # Save data
        result = json_io.save(test_data, temp_path)
        print(f"  Save result: {result}")
        
        # Load data
        loaded_data = json_io.load(temp_path)
        print(f"  Load result: {loaded_data is not None}")
        
        # Verify data
        if loaded_data == test_data:
            print("  Data verification: PASS")
        else:
            print("  Data verification: FAIL")
            print(f"  Expected: {test_data}")
            print(f"  Actual: {loaded_data}")
    finally:
        # Clean up
        os.unlink(temp_path)

def test_xml_file_io():
    """Test XmlFileIO class."""
    print("Testing XmlFileIO...")
    
    # Create test data
    test_data = {
        "root_tag": "TrackingData",
        "name": "Ball Tracking System",
        "version": "1.0.0",
        "settings": {
            "hsv": {
                "hue_min": 30,
                "hue_max": 90,
                "saturation_min": 100,
                "saturation_max": 255,
                "value_min": 100,
                "value_max": 255
            }
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create XmlFileIO instance
        xml_io = XmlFileIO()
        
        # Save data
        result = xml_io.save(test_data, temp_path)
        print(f"  Save result: {result}")
        
        # Load data
        loaded_data = xml_io.load(temp_path)
        print(f"  Load result: {loaded_data is not None}")
        
        # Print loaded data
        print(f"  Loaded data: {loaded_data}")
    finally:
        # Clean up
        os.unlink(temp_path)

def test_csv_file_io():
    """Test CsvFileIO class."""
    print("Testing CsvFileIO...")
    
    # Create test data
    test_data = [
        [1, "John", 30],
        [2, "Alice", 25],
        [3, "Bob", 35]
    ]
    headers = ["ID", "Name", "Age"]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create CsvFileIO instance
        csv_io = CsvFileIO()
        
        # Save data
        result = csv_io.save(test_data, temp_path, headers)
        print(f"  Save result: {result}")
        
        # Load data without headers
        loaded_data_no_headers = csv_io.load(temp_path)
        print(f"  Load result (no headers): {loaded_data_no_headers is not None}")
        print(f"  Loaded data (no headers): {loaded_data_no_headers}")
        
        # Load data with headers
        loaded_data_with_headers = csv_io.load(temp_path, has_headers=True)
        print(f"  Load result (with headers): {loaded_data_with_headers is not None}")
        print(f"  Loaded data (with headers): {loaded_data_with_headers}")
    finally:
        # Clean up
        os.unlink(temp_path)

def test_file_io_factory():
    """Test FileIOFactory class."""
    print("Testing FileIOFactory...")
    
    # Test JSON file
    json_io = FileIOFactory.create("test.json")
    print(f"  JSON file: {type(json_io).__name__}")
    
    # Test XML file
    xml_io = FileIOFactory.create("test.xml")
    print(f"  XML file: {type(xml_io).__name__}")
    
    # Test CSV file
    csv_io = FileIOFactory.create("test.csv")
    print(f"  CSV file: {type(csv_io).__name__}")
    
    # Test unsupported file
    unsupported_io = FileIOFactory.create("test.txt")
    print(f"  Unsupported file: {unsupported_io}")

if __name__ == "__main__":
    print("File I/O Module Test")
    print("=" * 50)
    
    test_json_file_io()
    print("\n" + "-" * 50 + "\n")
    
    test_xml_file_io()
    print("\n" + "-" * 50 + "\n")
    
    test_csv_file_io()
    print("\n" + "-" * 50 + "\n")
    
    test_file_io_factory()
    
    print("\n" + "=" * 50)
    print("Test Complete") 