import os
import numpy as np

def check_npy_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter for .npy files
    npy_files = [f for f in files if f.endswith('.npy')]
    
    # Initialize a list to store files that can't be read
    unreadable_files = []

    # Iterate through each .npy file
    for npy_file in npy_files:
        file_path = os.path.join(directory, npy_file)
        try:
            # Try reading the file
            data = np.load(file_path)
            print(f"Successfully read: {npy_file}")
        except Exception as e:
            # If there is an error, add the file to the list
            unreadable_files.append(npy_file)
            print(f"Error reading {npy_file}: {e}")
    
    # Return the list of unreadable files
    if unreadable_files:
        print("\nUnreadable files:")
        for file in unreadable_files:
            print(file)
    else:
        print("\nAll .npy files can be read successfully.")

# Example usage
directory_path = '/scratch/data/repos/Classical-Sound-Source-Localization-Algorithms-in-Spherical-Domain/output_music_starss_batch_reshaped/'
check_npy_files(directory_path)

