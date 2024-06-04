import os
import shutil
import pandas as pd

attributes = False
print('Attributes:', attributes)

# Set the dataset and the number of files to cut
dataset = "CREMA-D-voted"
num_files_per_class = 300


# Set the source and destination directories
if dataset == "IEMOCAP":
    # source_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio'
    # dest_dir = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing'
    source_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\audio"
    dest_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\audio_testing"

elif dataset == "CREMA-D":
    source_dir = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio'
    dest_dir = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio_testing'

elif dataset == "CREMA-D-voted":
    # source_dir = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio'
    # dest_dir = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio_v_testing'
    audio_source_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAV"
    video_source_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoMP4"
    audio_dest_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\AudioWAV_multimodal"
    video_dest_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoMP4_multimodal"

elif dataset == "EMO-DB":
    source_dir = r'C:\Users\DANIEL\Desktop\thesis\EmoDB\audio'
    dest_dir = r'C:\Users\DANIEL\Desktop\thesis\EmoDB\audio_testing'

elif dataset == "ShEMO":
    source_dir = r'C:\Users\DANIEL\Desktop\thesis\ShEMO\audio'
    dest_dir = r'C:\Users\DANIEL\Desktop\thesis\ShEMO\audio_testing'


train_dir_att = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_training_att'
dest_dir_att = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\audio_testing_att'


if attributes == False:

    # Creating a csv file with the data of the transferred wav files from the source csv file
    if dataset == "IEMOCAP":
        # source_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_corrected.csv'
        # dest_csv = r'C:\Users\DANIEL\Desktop\thesis\IEMOCAP_full_release\labels_testing.csv'
        source_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\labels_corrected.csv"
        dest_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\labels_testing.csv"
        

    elif dataset == "CREMA-D":
        source_csv = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_corrected.csv'
        dest_csv = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_testing.csv'

    elif dataset == "CREMA-D-voted":
        # source_csv = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\voted_labels_corrected.csv'
        # dest_csv = r'C:\Users\DANIEL\Desktop\thesis\CREMA-D\labels_v_testing.csv'
        source_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\voted_combined_labels_corrected.csv"
        dest_csv = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\voted_combined_labels_corrected_multimodal.csv"

    elif dataset == "EMO-DB":
        source_csv = r'C:\Users\DANIEL\Desktop\thesis\EmoDB\labels_corrected.csv'
        dest_csv = r'C:\Users\DANIEL\Desktop\thesis\EmoDB\labels_testing.csv'

    elif dataset == "ShEMO":
        source_csv = r'C:\Users\DANIEL\Desktop\thesis\ShEMO\labels_corrected.csv'
        dest_csv = r'C:\Users\DANIEL\Desktop\thesis\ShEMO\labels_testing.csv'


    # Get the list of files in the source directory
    # files = os.listdir(source_dir)

    # Read the source CSV file into a DataFrame
    df = pd.read_csv(source_csv)

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
    if dataset == "CREMA-D-voted":
        if not os.path.exists(audio_dest_dir):
            os.makedirs(audio_dest_dir, exist_ok=True)
            print('Audio destination directory created')
        if not os.path.exists(video_dest_dir):
            os.makedirs(video_dest_dir, exist_ok=True)
            print('Video destination directory created')
    else:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            print('Destination directory created')

    # Check if the destination directory is empty
    if dataset == "CREMA-D-voted":
        if os.listdir(audio_dest_dir) or os.listdir(video_dest_dir):
            print('Warning: Destination directory is not empty')
        else:
            for file in selected_files:
                shutil.copy(os.path.join(audio_source_dir, file + '.wav'), audio_dest_dir)
                shutil.copy(os.path.join(video_source_dir, file + '.mp4'), video_dest_dir)
    else:
        if os.listdir(dest_dir):
            print('Warning: Destination directory is not empty')
        else:
            for file in selected_files:
                if file[-4:] != '.wav':
                    file = file + '.wav'
                shutil.copy(os.path.join(source_dir, file), dest_dir)

    print("Copied files: ", len(selected_files))


    # Check if the order of the rows on the CSV file match the order of the files in the destination directory
    if dataset == "CREMA-D-voted":
        final_files = os.listdir(audio_dest_dir)
    else:
        final_files = os.listdir(dest_dir)
    exception = False
    for i, file in enumerate(final_files):
        filename = selected_files[i].split('.')[0]
        if file != (filename + '.wav'):
            print(i)
            print("File:", file + " - " + "Filename:", filename + '.wav')
            print("Warning: The order of the rows on the CSV file does not match the order of the files in the destination directory")
            exception = True
            break

    # Write the selected DataFrame to the destination CSV file
    if exception == False:
        selected_df.to_csv(dest_csv, index=False)
        print('CSV file created')



        




