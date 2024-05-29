from util import extract_audio_from_video
from util import detect_broken_audio_files
from util import renaming_MELD_files
from util import match_labels
import os
import shutil
import pandas as pd



def handle_MELD(directory):
    renaming_MELD_files.rename_audio_files(directory)

def handle_CREMA_D(directory):
    # Extracting the labels from the file names
    index = []
    filename = []
    speaker = []
    line = []
    emotions = []
    intensity = []

    for i, file in enumerate(os.listdir(directory)):
        index.append(i)
        filename.append(file)
        speaker.append(file.split("_")[0])
        line.append(file.split("_")[1])
        emotions.append(file.split("_")[2])
        intensity.append(file.split("_")[3][:2])

    # Creating a csv file with the extracted labels
    labels = pd.DataFrame(list(zip(index, filename, speaker, line, emotions, intensity)), columns = ["Index", "filename", "Speaker", "Line", "Emotion", "Intensity"])
    labels.to_csv(os.path.join(os.path.dirname(directory), "labels.csv"), index=False)

    return

def handle_CREMA_D_v(directory, source_labels):
    # Specify the columns you want to read
    cols = ["FileName", "VoiceVote", "VoiceLevel"]
    # Uncomment the version you need (Voice / Face)
    # cols = ["FileName", "FaceVote", "FaceLevel"]

    # Read the specified columns from the source CSV file
    labels = pd.read_csv(source_labels, usecols=cols)

    # Rename the columns
    labels = labels.rename(columns={
        "FileName": "filename",
        "VoiceVote": "Emotion",
        "VoiceLevel": "Level"
    })
    # Uncomment the version you need (Voice / Face)
    # labels = labels.rename(columns={
    #     "FileName": "filename",
    #     "FaceVote": "Emotion",
    #     "FaceLevel": "Level"
    # })

     # Filter rows where 'Emotion' has more than one letter
    labels = labels[labels['Emotion'].str.len() == 1]

    # Creating a csv file with the extracted labels
    labels.to_csv(os.path.join(os.path.dirname(directory), "voted_voice_labels.csv"), index=False)
    # Uncomment the version you need (Voice / Face)
    # labels.to_csv(os.path.join(os.path.dirname(directory), "voted_face_labels.csv"), index=False)

    return

def handle_EmoDB(directory):
    # Extracting the labels from the file names
    filename = []
    speaker = []
    line = []
    emotions = []
    version = []

    for file in os.listdir(directory):
        filename.append(file)
        speaker.append(file[0:2])
        line.append(file[2:5])
        emotions.append(file[5])
        version.append(file[6])

    # Creating a csv file with the extracted labels
    labels = pd.DataFrame(list(zip(filename, speaker, line, emotions, version)), columns = ["filename", "Speaker", "Line", "Emotion", "Version"])
    labels.to_csv(os.path.join(os.path.dirname(directory), "labels.csv"), index=False)

    return

def handle_ShEMO(directory):
    # Extracting the labels from the file names
    filename = []
    gender = []
    speaker = []
    emotions = []
    number = []

    for file in os.listdir(directory):
        filename.append(file)
        gender.append(file[0])
        speaker.append(file[0:3])
        emotions.append(file[3])
        number.append(file[4:6])

    # Creating a csv file with the extracted labels
    labels = pd.DataFrame(list(zip(filename, speaker, gender, emotions, number)), columns = ["filename", "Speaker", "Gender", "Emotion", "Number"])
    labels.to_csv(os.path.join(os.path.dirname(directory), "labels.csv"), index=False)

    return


def handle_IEMOCAP(labels_path, source_directories):
    # Initialize an empty DataFrame
    first_lines = []

    # Iterate over each source directory
    for source_directory in source_directories:
        # Walk through subdirectories
        for dirpath, dirnames, filenames in os.walk(os.path.join(source_directory, "dialog/EmoEvaluation")):
            # We search for only files in the "EmoEvaluation" directory
            if dirpath.endswith("EmoEvaluation"):
                # Iterate over each file
                for filename in filenames:
                    # If the file is a .txt file and doesn't start with a "."
                    if filename.endswith(".txt") and not filename.startswith("."):
                        # Define file path
                        file_path = os.path.join(dirpath, filename)
                        # Open the file in read mode
                        with open(file_path, 'r') as file:
                            # Initialize a flag to indicate whether we're at the start of a paragraph
                            start_of_paragraph = True
                            # Read the file line by line
                            lines=file.readlines()
                            for line in lines:
                                # If the line is not empty and we're at the start of a paragraph
                                if line.strip() and start_of_paragraph and not line.startswith('%'):
                                    # Process the line and add it to the list
                                    first_lines.append(line.split('\t'))
                                    # Update the flag to indicate that we're no longer at the start of a paragraph
                                    start_of_paragraph = False
                                # If the line is empty, update the flag to indicate that we're at the start of a paragraph
                                elif not line.strip():
                                    start_of_paragraph = True
    
    # Define the column names
    column_names = ["Time", "filename", "Emotion", "Attributes"]

    # Convert the list to a DataFrame
    df = pd.DataFrame(first_lines, columns=column_names)

    # Save the DataFrame to a .csv file
    df.to_csv(labels_path, index=False)



def default_case():
    print("No dataset match found.")

dataset_switch = {
    "MELD": handle_MELD,
    "CREMA-D": handle_CREMA_D,
    "IEMOCAP": handle_IEMOCAP
}

def switch_case(dataset, *args, **kwargs):
    if dataset in dataset_switch:
        dataset_switch[dataset](*args, **kwargs)
    else:
        default_case()

def main():
    toggle_controls = [True, True, True, True]
    dataset = "IEMOCAP"
    attributes = True
    audio_directory = ""
    corrected_labels_path = ""
    labels_path = ""
    video_directory = ""


    if dataset == "MELD":
        video_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\videos"
        labels_path = r"C:\MyDocs\DTU\MSc\Thesis\Data\MELD\Automation_testing\dev_sent_emo.csv"

        toggle_controls = [True, True, True, True]

        audio_directory = os.path.join(os.path.dirname(video_directory), os.path.basename(video_directory) + "_audio")
        corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")

    elif dataset == "CREMA-D":
        old_audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWAV"
        audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio"
        labels_path = os.path.join(os.path.dirname(audio_directory), "labels.csv")

        # Create the destination directory if it doesn't exist
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory, exist_ok=True)

            # Copy all files from the source to the destination directory
            for filename in os.listdir(old_audio_directory):
                source_path = os.path.join(old_audio_directory, filename)
                destination_path = os.path.join(audio_directory, filename)
                shutil.copy2(source_path, destination_path)

        if not os.path.exists(labels_path):
            handle_CREMA_D(audio_directory)

        toggle_controls = [False, False, False, True]
        corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")

    elif dataset == "CREMA-D-voted":
        old_audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWAV"
        audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\audio"
        source_labels = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\processedResults\summaryTable.csv"
        labels_path = os.path.join(os.path.dirname(audio_directory), "voted_labels.csv")

        # Create the destination directory if it doesn't exist
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory, exist_ok=True)

            # Copy all files from the source to the destination directory
            for filename in os.listdir(old_audio_directory):
                source_path = os.path.join(old_audio_directory, filename)
                destination_path = os.path.join(audio_directory, filename)
                shutil.copy2(source_path, destination_path)

        if not os.path.exists(labels_path):
            handle_CREMA_D_v(audio_directory,source_labels)

        toggle_controls = [False, False, False, True]
        corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")


    elif dataset == "EMO-DB":
        audio_directory = r"C:\Users\DANIEL\Desktop\thesis\EmoDB\audio"
        labels_path = os.path.join(os.path.dirname(audio_directory), "labels.csv")

        if not os.path.exists(labels_path):
            handle_EmoDB(audio_directory)

        toggle_controls = [False, False, False, True]
        corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")
    
    elif dataset == "ShEMO":
        audio_directory = r"C:\Users\DANIEL\Desktop\thesis\ShEMO\audio"
        labels_path = os.path.join(os.path.dirname(audio_directory), "labels.csv")

        if not os.path.exists(labels_path):
            handle_ShEMO(audio_directory)

        toggle_controls = [False, False, False, True]
        corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")
     

    elif dataset == "IEMOCAP":

        # old_audio_directory = r"C:\Users\DANIEL\Desktop\thesis\CREMA-D\AudioWAV"
        source_directories = [os.path.join(r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release", f"Session{i}") for i in range(1, 7)]
        audio_directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\IEMOCAP\IEMOCAP_full_release\audio"
        labels_path = os.path.join(os.path.dirname(audio_directory), "labels.csv")

        # Create the destination directory if it doesn't exist
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory, exist_ok=True)

            # Iterate over each source directory
            for source_directory in source_directories:
                # Walk through subdirectories
                for dirpath, dirnames, filenames in os.walk(os.path.join(source_directory, "sentences/wav")):
                    print("dirpath: ", dirpath)
                    # If we're in the directory containing .wav files
                    if not dirpath.endswith("wav"):
                        print("dirpath: ", dirpath)
                        # Iterate over each file
                        for filename in filenames:
                            # If the file is a .wav file and doesn't start with a "."
                            if filename.endswith(".wav") and not filename.startswith("."):
                                # Define source file path
                                source_file_path = os.path.join(dirpath, filename)
                                # Define destination file path
                                destination_file_path = os.path.join(audio_directory, filename)
                                # Copy the file
                                shutil.copy2(source_file_path, destination_file_path)

        if not os.path.exists(labels_path):
            handle_IEMOCAP(labels_path, source_directories)

        toggle_controls = [False, False, False, True]
        if attributes == False:
            corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_corrected.csv")
        else:
            corrected_labels_path = os.path.join(os.path.dirname(labels_path), os.path.basename(labels_path)[:-4] + "_attributes_corrected.csv")



    if attributes == True and dataset != "IEMOCAP":
        attributes = False
        print("Attributes can only be extracted from the IEMOCAP dataset.")

    extract_audio_files_from_video = toggle_controls[0]
    rename_files = toggle_controls[1]
    detect_broken_files = toggle_controls[2]
    extract_corrected_labels = toggle_controls[3]
    



    if extract_audio_files_from_video:
        print("Extracting audio files from video files...")
        if not os.path.exists(audio_directory):
            os.makedirs(audio_directory)
        if not os.listdir(audio_directory):
            extract_audio_from_video.extract_files(video_directory, audio_directory, num_files_to_process=50)
        else:
            print("Audio file folder not empty.")

        print("Done.")

    if rename_files:
        print("Renaming files...")
        switch_case(dataset, audio_directory)
        print("Done.")
    
    if detect_broken_files:
        print("Detecting broken audio files...")
        detect_broken_audio_files.process_files(audio_directory)
        print("Done.")
    
    if extract_corrected_labels:
        print("Extracting corrected labels...")
        match_labels.match_emotion_labels(labels_path,  corrected_labels_path, audio_directory, dataset, attributes)
        print("Done.")

if __name__ == "__main__":
    main()