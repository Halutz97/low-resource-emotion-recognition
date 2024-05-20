# Cut 100 randomly selected wav audios from a directory and save them to another directory

import os
import random
import shutil
import csv

attributes = False
print('Attributes:', attributes)

# Set the source and destination directories
source_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio'
dest_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing'
train_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_training'
train_dir_att = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_training_att'
dest_dir_att = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing_att'

# Set the number of files to cut
num_files = 300

if attributes == False:

    print('Cutting', num_files, 'files from', source_dir, 'to', dest_dir)

    # Get the list of files in the source directory
    files = os.listdir(source_dir)

    # Randomly select num_files files
    selected_files = random.sample(files, num_files)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        print('Destination directory created')

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)
        print('Copied', file)

    print('Done')

    # Creating a csv file with the data of the transferred wav files from the source csv file
    source_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_corrected.csv'
    dest_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv'
    train_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_training.csv'

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

    # # Search the new CSV file for entries with emotions "fru", "oth", and "exc"
    # emotions_to_remove = ["fru", "oth", "exc"]
    # rows_to_remove = []

    # # Open the new CSV file
    # with open(dest_csv, 'r') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)

    #     # Find the rows with emotions to remove
    #     for row in reader:
    #         if row[1] in emotions_to_remove:
    #             rows_to_remove.append(row)

    # # Remove the corresponding wav files from the folder and the new CSV file
    # for row in rows_to_remove:
    #     filename = row[0]+".wav"
    #     filepath = os.path.join(dest_dir, filename)
    #     if os.path.exists(filepath):
    #         os.remove(filepath)
    #         print('Removed', filename)
    #     else:
    #         print('File not found:', filename)

    # # Remove the rows from the new CSV file
    # with open(dest_csv, 'r') as file:
    #     reader = csv.reader(file)
    #     rows = list(reader)

    # with open(dest_csv, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)
    #     for row in rows:
    #         if row not in rows_to_remove:
    #             writer.writerow(row)

    # print('Wav files and corresponding rows removed from the new CSV file')

    # # Actualize values from the labels column in the new CSV file following the new my_encoding_dict
    # # Emotion-Label codifier
    # my_encoding_dict = {'ang': 0, 'cal': 1, 'dis': 2, 'fea': 3, 'hap': 4, 'neu': 5, 'sad': 6, 'sur': 7}

    # with open(dest_csv, 'r') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)
    #     header2 = next(reader)
    #     rows = list(reader)

    # with open(dest_csv, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)
    #     for row in rows:
    #         row[2] = my_encoding_dict[row[1]]
    #         writer.writerow(row)

    # print('Values in the labels column updated in the new CSV file')




    # Remove extracted files from the source csv files and directory

    # Remove the corresponding wav files from the folder and the new CSV file
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print('Training directory created')

    for file in files:
        if file not in selected_files:
            shutil.copy(os.path.join(source_dir, file), train_dir)
            print('Copied', file)


    # Open the source CSV file
    with open(source_csv, 'r') as source:
        reader = csv.reader(source)

        # Open the destination CSV file
        with open(train_csv, 'w', newline='') as destination:
            writer = csv.writer(destination)

            # Write the header (assuming the first row is the header)
            header = next(reader)
            writer.writerow(header)

            # For each row in the source CSV file
            for row in reader:
                # If the filename (assuming it's in the first column) is in the list of transferred files
                if (row[0]+".wav") not in selected_files:
                    # Write the row to the destination CSV file
                    writer.writerow(row)

    print('CSV for training created')

if attributes == True:

    print('Cutting', num_files, 'files from', source_dir, 'to', dest_dir_att)

    # Get the list of files in the source directory
    files = os.listdir(source_dir)

    # Randomly select num_files files
    selected_files = random.sample(files, num_files)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir_att):
        os.makedirs(dest_dir_att, exist_ok=True)
        print('Destination directory created')

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(source_dir, file), dest_dir_att)
        print('Copied', file)

    print('Done')

    # Creating a csv file with the data of the transferred wav files from the source csv file
    source_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_attributes_corrected.csv'
    dest_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\attributes_testing.csv'
    train_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\attributes_training.csv'

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

    if not os.path.exists(train_dir_att):
        os.makedirs(train_dir_att, exist_ok=True)
        print('Training directory created')

    for file in files:
        if file not in selected_files:
            shutil.copy(os.path.join(source_dir, file), train_dir_att)
            print('Copied', file)


    # Open the source CSV file
    with open(source_csv, 'r') as source:
        reader = csv.reader(source)

        # Open the destination CSV file
        with open(train_csv, 'w', newline='') as destination:
            writer = csv.writer(destination)

            # Write the header (assuming the first row is the header)
            header = next(reader)
            writer.writerow(header)

            # For each row in the source CSV file
            for row in reader:
                # If the filename (assuming it's in the first column) is in the list of transferred files
                if (row[0]+".wav") not in selected_files:
                    # Write the row to the destination CSV file
                    writer.writerow(row)

    print('CSV for training created')



