import os

def delete_mp4_files(folder_path):
    """
    Deletes all MP4 files in the specified folder.

    Args:
    folder_path (str): Path to the folder to search for MP4 files.
    """
    # Count of deleted files
    files_deleted = 0

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an MP4 file
        if filename.endswith('.mp4'):
            file_path = os.path.join(folder_path, filename)
            
            # Attempt to delete the file
            try:
                os.remove(file_path)
                files_deleted += 1
                print(f"Deleted {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

    # If no MP4 files found, inform the user
    if files_deleted == 0:
        print("No MP4 files found to delete.")
    else:
        print(f"Total of {files_deleted} MP4 files deleted.")

# Example usage
folder = "C:\MyDocs\DTU\MSc\Thesis\Data\MELD\MELD_fine_tune_v1_train_data"
delete_mp4_files(folder)