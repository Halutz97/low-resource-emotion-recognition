# Cut 100 randomly selected wav audios from a directory and save them to another directory

import os
import random
import shutil
import csv

# Set the source and destination directories
source_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio'
dest_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing'

# Set the number of files to cut
num_files = 100

print('Cutting', num_files, 'files from', source_dir, 'to', dest_dir)

# Get the list of files in the source directory
files = os.listdir(source_dir)

# Randomly select num_files files
selected_files = random.sample(files, num_files)

# Copy the selected files to the destination directory
for file in selected_files:
    shutil.copy(os.path.join(source_dir, file), dest_dir)
    print('Copied', file)

print('Done')

# Creating a csv file with the data of the transfered wav files from the source csv file
source_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_corrected.csv'
dest_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv'

# Open the source CSV file
with open(source_csv, 'r') as source:
    reader = csv.reader(source)

    # Open the destination CSV file
    with open(dest_csv, 'w', newline='') as destination:
        writer = csv.writer(destination)

        # Write the header (assuming the first row is the header)
        header = next(reader)
        writer.writerow(header)

        # For each row in the source CSV file
        for row in reader:
            # If the filename (assuming it's in the first column) is in the list of transferred files
            if (row[0]+".wav") in selected_files:
                # Write the row to the destination CSV file
                writer.writerow(row)

print('CSV file created')

