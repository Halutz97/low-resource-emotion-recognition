import os
import shutil

def copy_wav_files(source_folder, destination_folder):
    """
    Copies all WAV files from the source folder to the destination folder.
    Creates the destination folder if it does not exist.

    Args:
    source_folder (str): Path to the folder where WAV files are located.
    destination_folder (str): Path to the folder where WAV files will be copied.
    """
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory {destination_folder}")
    
    # Count of copied files
    files_copied = 0

    # Loop through each file in the source folder
    for filename in os.listdir(source_folder):
        if files_copied >= 150:
            break
        # Check if the file is a WAV file
        if filename.endswith('.wav'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Copy the file to the destination folder
            shutil.copy2(source_file, destination_file)
            files_copied += 1
            print(f"Copied {filename} to {destination_folder}")

    # If no WAV files found, inform the user
    if files_copied == 0:
        print("No WAV files found to copy.")
    else:
        print(f"Total of {files_copied} WAV files copied to {destination_folder}.")

# Example usage
source_dir = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD testing"
destination_dir = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_test_data"

copy_wav_files(source_dir, destination_dir)
