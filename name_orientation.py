import os
import csv

def get_rgb_filenames(directory="data/top_5_compressed"):
    """
    Scans the specified directory and returns a list of filenames 
    that contain the substring "_RGB".
    """
    rgb_files = []
    classes = []
    # Check if the directory exists to avoid errors
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if "_RGB" in filename:
                # Construct the full path or just keep the filename depending on preference
                # The original code just appended the filename, but usually with recursion 
                # you want the full path or relative path. 
                # Based on the original snippet returning just filenames, I will stick to that
                # but usually os.path.join(root, filename) is more useful.
                # However, to strictly match the "recurse" request while keeping the list flat:
                rgb_files.append(filename)
                classes.append(root.split("/")[3])
            
    return rgb_files, classes

if __name__ == "__main__":
    # Example usage
    files, classes = get_rgb_filenames()
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class', 'filename', 'label'])  # Optional header
        for i, f in enumerate(files):
            writer.writerow([classes[i], f, 0])