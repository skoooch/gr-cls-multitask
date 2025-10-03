import os
import time

directory = os.path.dirname(os.path.abspath(__file__))

while True:
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    time.sleep(5)  # Check every 5 seconds