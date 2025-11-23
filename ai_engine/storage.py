# FILE: ai_engine/storage.py
import os
import csv
from typing import List, Dict, Any

class CSVStorage:
    """
    Handles saving input data, predictions, and feedback to a CSV file.
    """
    def __init__(self, path: str):
        """
        Initialize the storage with a file path. Creates the file if it doesn't exist.
        
        Args:
            path (str): Path to the CSV file.
        """
        self.path = path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """
        Creates the directory and file if they don't exist.
        """
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # We don't create the file here with headers because we don't know the headers yet.
        # We'll handle header creation on the first write.

    def save_record(self, data: Dict[str, Any]):
        """
        Appends a single record to the CSV file.
        
        Args:
            data (dict): Dictionary containing data to save. Keys become headers.
        """
        file_exists = os.path.exists(self.path)
        
        try:
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                
                if not file_exists or os.path.getsize(self.path) == 0:
                    writer.writeheader()
                
                writer.writerow(data)
        except IOError as e:
            raise IOError(f"Failed to write to CSV file {self.path}: {e}")

    def save_batch(self, list_of_dicts: List[Dict[str, Any]]):
        """
        Appends multiple records to the CSV file.
        
        Args:
            list_of_dicts (list): List of dictionaries to save.
        """
        if not list_of_dicts:
            return

        file_exists = os.path.exists(self.path)
        keys = list_of_dicts[0].keys()
        
        try:
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                
                if not file_exists or os.path.getsize(self.path) == 0:
                    writer.writeheader()
                
                writer.writerows(list_of_dicts)
        except IOError as e:
            raise IOError(f"Failed to write batch to CSV file {self.path}: {e}")
