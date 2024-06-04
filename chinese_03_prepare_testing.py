import os
import shutil
import pandas as pd

num_files_per_class = 200

source_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4"
dest_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\VideoMP4_testing"

source_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\label_corrected.csv"
dest_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\CH-SIMS-RAW\label_corrected_testing.csv"

# Get the list of files in the source directory
# files = os.listdir(source_dir)

# Read the source CSV file into a DataFrame
df = pd.read_csv(source_csv)

# rows (filenames) to remove, if they are present in the dataframe
# video_0045_0029
# video_0056_0026
# video_0060_0041

# create list of filenames to remove
filenames_to_remove = ['video_0045_0029', 'video_0056_0026', 'video_0060_0041']

df = df[df['filename'] != filenames_to_remove[0]]
df = df[df['filename'] != filenames_to_remove[1]]
df = df[df['filename'] != filenames_to_remove[2]]

# Group the DataFrame by the 'Emotion' column
grouped = df.groupby('Emotion')

# For each group, check if the group size is less than num_files_per_class
for emotion, group in grouped:
    if len(group) < num_files_per_class:
        print(f"Warning: There are only {len(group)} files for emotion {emotion}, less than the desired {num_files_per_class}.")
        num_files_per_class = len(group)

# Randomly select num_files_per_class rows from each group
selected_df = grouped.apply(lambda x: x.sample(min(len(x), num_files_per_class)))

# Reset the index of the selected DataFrame
selected_df.reset_index(drop=True, inplace=True)

# Sort the DataFrame by the first column
selected_df.sort_values('filename', inplace=True)

# Get the list of selected files
selected_files = selected_df['filename'].tolist()

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    print('Destination directory created')

# Check if the destination directory is empty
if os.listdir(dest_dir):
    print('Warning: Destination directory is not empty')
else:
    for file in selected_files:
        file = file + '.mp4'
        shutil.copy2(os.path.join(source_dir, file), dest_dir)
    print("Copied files: ", len(selected_files))


# Check if the order of the rows on the CSV file match the order of the files in the destination directory
final_files = os.listdir(dest_dir)
exception = False
for i, file in enumerate(final_files):
    filename = selected_files[i]
    if file != (filename + '.mp4'):
        print(i)
        print("File:", file + " - " + "Filename:", filename + '.mp4')
        print("Warning: The order of the rows on the CSV file does not match the order of the files in the destination directory")
        exception = True
        break

# Write the selected DataFrame to the destination CSV file
if exception == False:
    selected_df.to_csv(dest_csv, index=False)
    print('CSV file created')
        

    





        




